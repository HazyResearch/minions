import torch
from typing import List, Dict
from rank_bm25 import BM25Plus
from abc import ABC, abstractmethod
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("SentenceTransformer not installed")

try:
    import faiss
except ImportError:
    faiss = None
    print("faiss not installed")


def bm25_retrieve_top_k_chunks(
    keywords: List[str],
    chunks: List[str] = None,
    weights: Dict[str, float] = None,
    k: int = 10,
) -> List[str]:
    """
    Retrieves top k chunks using BM25 with weighted keywords.
    """

    weights = {keyword: weights.get(keyword, 1.0) for keyword in keywords}
    bm25_retriever = BM25Plus(chunks)

    final_scores = np.zeros(len(chunks))
    for keyword, weight in weights.items():
        scores = bm25_retriever.get_scores(keyword)
        final_scores += weight * scores

    top_k_indices = sorted(
        range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
    )[:k]
    top_k_indices = sorted(top_k_indices)
    relevant_chunks = [chunks[i] for i in top_k_indices]

    return relevant_chunks


class BaseEmbeddingModel(ABC):
    """
    Abstract base class defining interface for embedding models.
    """

    @abstractmethod
    def get_model(self, **kwargs):
        """Get or initialize the embedding model."""
        pass

    @abstractmethod
    def encode(self, texts, **kwargs) -> np.ndarray:
        """Encode texts to create embeddings."""
        pass


class EmbeddingModel(BaseEmbeddingModel):
    """
    Implementation of embedding model using SentenceTransformer.
    """

    _instances = {}  # Dictionary to store instances by model name
    _default_model_name = "intfloat/multilingual-e5-large-instruct"

    def __new__(cls, model_name=None):
        model_name = model_name or cls._default_model_name
        
        # Check if we already have an instance for this model
        if model_name not in cls._instances:
            instance = super(EmbeddingModel, cls).__new__(cls)
            instance.model_name = model_name
            instance._model = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                instance._model = instance._model.to(torch.device("cuda"))
            cls._instances[model_name] = instance
        
        return cls._instances[model_name]

    def get_model(self):
        return self._model

    def encode(self, texts) -> np.ndarray:
        return self._model.encode(texts).astype("float32")

    @classmethod
    def get_model_by_name(cls, model_name=None):
        """Get model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.get_model()

    @classmethod
    def encode_by_name(cls, texts, model_name=None) -> np.ndarray:
        """Encode texts using model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.encode(texts)


def embedding_retrieve_top_k_chunks(
    queries: List[str],
    chunks: List[str] = None,
    k: int = 10,
    embedding_model: BaseEmbeddingModel = None,
) -> List[str]:
    """
    Retrieves top k chunks using dense vector embeddings and FAISS similarity search

    Args:
        queries: List of query strings
        chunks: List of text chunks to search through
        k: Number of top chunks to retrieve
        embedding_model: Optional embedding model to use (defaults to EmbeddingModel)

    Returns:
        List of top k relevant chunks
    """
    # Check if FAISS is available
    if faiss is None:
        raise ImportError(
            "FAISS is not installed. Please install it with: pip install faiss-cpu"
        )

    # Check if SentenceTransformer is available  
    if SentenceTransformer is None:
        raise ImportError(
            "SentenceTransformer is not installed. Please install it with: pip install sentence-transformers"
        )

    # Use the provided embedding model or default to EmbeddingModel
    if embedding_model is None:
        model = EmbeddingModel()
    else:
        model = embedding_model

    chunk_embeddings = model.encode(chunks).astype("float32")

    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(chunk_embeddings)

    aggregated_scores = np.zeros(len(chunks))

    for query in queries:
        query_embedding = model.encode([query]).astype("float32")
        cur_scores, cur_indices = index.search(query_embedding, k)
        np.add.at(aggregated_scores, cur_indices[0], cur_scores[0])

    top_k_indices = np.argsort(aggregated_scores)[::-1][:k]

    relevant_chunks = [chunks[i] for i in top_k_indices]

    return relevant_chunks
