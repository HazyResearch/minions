from typing import List, Dict, Union
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

# MLX Embeddings support
try:
    import mlx.core as mx
    from mlx_embeddings.utils import load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


### EMBEDDING MODELS ###

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


class SentenceTransformerEmbeddings(BaseEmbeddingModel):
    """
    Implementation of embedding model using SentenceTransformer.
    """

    _instances = {}  # Dictionary to store instances by model name
    _default_model_name = "Qwen/Qwen3-Embedding-0.6B"

    def __new__(cls, model_name=None):
        model_name = model_name or cls._default_model_name
        print(f"Using SentenceTransformer model: {model_name}")

        try:
            import torch 
        except ImportError:
            print("torch not installed")
        
        # Check if we already have an instance for this model
        if model_name not in cls._instances:
            instance = super(SentenceTransformerEmbeddings, cls).__new__(cls)
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


# For backward compatibility
EmbeddingModel = SentenceTransformerEmbeddings


class MLXEmbeddings(BaseEmbeddingModel):
    """
    Implementation of embedding model using MLX Embeddings.

    This class provides an interface to use MLX-based embedding models
    with the existing retrieval system.
    """

    _instance = None
    _model = None
    _tokenizer = None
    _default_model_name = "mlx-community/all-MiniLM-L6-v2-4bit"

    def __new__(cls, model_name=None, **kwargs):
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX and mlx-embeddings are required to use MLXEmbeddings. "
                "Please install them with: pip install mlx mlx-embeddings"
            )

        if cls._instance is None:
            cls._instance = super(MLXEmbeddings, cls).__new__(cls)
            model_name = model_name or cls._default_model_name
            cls._model, cls._tokenizer = load(model_name, **kwargs)
        return cls._instance

    @classmethod
    def get_model(cls, model_name=None, **kwargs):
        """Get or initialize the MLX embedding model and tokenizer."""
        if cls._instance is None:
            cls._instance = cls(model_name, **kwargs)
        return cls._model, cls._tokenizer

    @classmethod
    def encode(
        cls,
        texts: Union[str, List[str]],
        model_name=None,
        max_length: int = 1024,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts to create embeddings using MLX model.

        Args:
            texts: Single text or list of texts to encode
            model_name: Optional model name to use
            normalize: Whether to normalize embeddings (default: True)
            batch_size: Batch size for encoding (default: 32)
            max_length: Maximum sequence length (default: 512)
            **kwargs: Additional arguments to pass to the model

        Returns:
            Numpy array of embeddings
        """
        model, tokenizer = cls.get_model(model_name)

        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        inputs = tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Get embeddings
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        # Get the text embeddings (already normalized if the model does that)
        embeddings = outputs.text_embeds
        
        # Convert MLX array to NumPy array for compatibility with existing code
        if hasattr(embeddings, 'numpy'):
            # If it's an MLX array with numpy() method
            embeddings = embeddings.numpy()
        elif hasattr(embeddings, '__array__'):
            # If it has __array__ method (for array-like objects)
            embeddings = np.array(embeddings)
        else:
            # Fallback: try to convert directly
            embeddings = np.array(embeddings)
        
        return embeddings

### RETRIEVERS ###

def bm25_retrieve_top_k_chunks(
    keywords: List[str],
    chunks: List[str] = None,
    weights: Dict[str, float] = None,
    k: int = 10,
) -> List[str]:
    """
    Retrieves top k chunks using BM25 with weighted keywords.
    """

    # Handle case where weights is None
    if weights is None:
        weights = {}
    
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

def embedding_retrieve_top_k_chunks(
    queries: List[str],
    chunks: List[str] = None,
    k: int = 10,
    embedding_model: BaseEmbeddingModel = None,
    embedding_model_name: str = None,
) -> List[str]:
    """
    Retrieves top k chunks using dense vector embeddings and FAISS similarity search

    Args:
        queries: List of query strings
        chunks: List of text chunks to search through
        k: Number of top chunks to retrieve
        embedding_model: Optional embedding model to use (defaults to SentenceTransformerEmbeddings)

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

    # Use the provided embedding model or default to SentenceTransformerEmbeddings
    if embedding_model is None:
        model = SentenceTransformerEmbeddings(embedding_model_name)
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