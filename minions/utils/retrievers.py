import torch
from typing import List, Dict
from rank_bm25 import BM25Plus
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

    
def bm25_retrieve_top_k_chunks(
    keywords: List[str], chunks: List[str] = None, weights: Dict[str, float] = None, k: int = 10
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
    Singleton implementation of embedding model using SentenceTransformer.
    """
    _instance = None
    _model = None
    _default_model_name = 'intfloat/multilingual-e5-large-instruct'
    
    def __new__(cls, model_name=None):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            model_name = model_name or cls._default_model_name
            cls._model = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                cls._model = cls._model.to(torch.device("cuda"))
        return cls._instance
    
    @classmethod
    def get_model(cls, model_name=None):
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._model
    
    @classmethod
    def encode(cls, texts, model_name=None) -> np.ndarray:
        model = cls.get_model(model_name)
        return model.encode(texts).astype('float32')


def embedding_retrieve_top_k_chunks(
    queries: List[str], chunks: List[str] = None, k: int = 10
) -> List[str]:
    """
    Retrieves top k chunks using dense vector embeddings and FAISS similarity search
    """
    
    chunk_embeddings = EmbeddingModel.encode(chunks).astype('float32')
    
    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(chunk_embeddings)
    
    aggregated_scores = np.zeros(len(chunks))
    
    for query in queries:
        query_embedding = EmbeddingModel.encode([query]).astype('float32')
        cur_scores, cur_indices = index.search(query_embedding, k)
        np.add.at(aggregated_scores, cur_indices[0], cur_scores[0])
    
    top_k_indices = np.argsort(aggregated_scores)[::-1][:k]
    
    relevant_chunks = [chunks[i] for i in top_k_indices]

    return relevant_chunks