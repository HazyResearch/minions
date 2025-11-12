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

# Gemini Embeddings support
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# OpenRouter Embeddings support
try:
    from minions.clients.openrouter import OpenRouterClient
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Liquid AI ColBERT support
try:
    from pylate import models, indexes, retrieve, rank
    LIQUID_COLBERT_AVAILABLE = True
except ImportError:
    LIQUID_COLBERT_AVAILABLE = False

# Ollama Embeddings support
try:
    from ollama import embed as ollama_embed
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


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
    _default_model_name = "granite-embedding-english-r2"

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


class GeminiEmbeddings(BaseEmbeddingModel):
    """
    Implementation of embedding model using Google Gemini Embeddings.
    
    This class provides an interface to use Gemini-based embedding models
    with the existing retrieval system.
    """

    _instances = {}  # Dictionary to store instances by model name
    _default_model_name = "text-embedding-004"

    def __new__(cls, model_name=None):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai is required to use GeminiEmbeddings. "
                "Please install it with: pip install google-genai"
            )

        model_name = model_name or cls._default_model_name
        print(f"Using Gemini embedding model: {model_name}")

        # Check if we already have an instance for this model
        if model_name not in cls._instances:
            instance = super(GeminiEmbeddings, cls).__new__(cls)
            instance.model_name = model_name
            instance.api_key = cls._get_api_key()
            instance._client = genai.Client(api_key=instance.api_key)
            cls._instances[model_name] = instance
        
        return cls._instances[model_name]

    @staticmethod
    def _get_api_key():
        """Get API key from environment variables."""
        import os
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        return api_key

    def get_model(self):
        """Get the Gemini client."""
        return self._client

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts to create embeddings using Gemini model.

        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments (currently unused)

        Returns:
            Numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        embeddings_list = []
        
        # Process texts in batches to handle API limits
        batch_size = kwargs.get('batch_size', 100)  # Gemini has limits on batch size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Use the embed_content method from Gemini API
                result = self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts,
                )
                
                # Extract embeddings from the result
                batch_embeddings = []
                if hasattr(result, 'embeddings'):
                    # Handle multiple embeddings
                    if isinstance(result.embeddings, list):
                        for embedding in result.embeddings:
                            if hasattr(embedding, 'values'):
                                batch_embeddings.append(embedding.values)
                            else:
                                batch_embeddings.append(embedding)
                    else:
                        # Single embedding
                        if hasattr(result.embeddings, 'values'):
                            batch_embeddings.append(result.embeddings.values)
                        else:
                            batch_embeddings.append(result.embeddings)
                else:
                    raise ValueError("No embeddings found in Gemini API response")
                
                embeddings_list.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Create zero embeddings as fallback
                fallback_dim = 768  # Default embedding dimension
                for _ in batch_texts:
                    embeddings_list.append([0.0] * fallback_dim)

        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings

    @classmethod
    def get_model_by_name(cls, model_name=None):
        """Get model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.get_model()

    @classmethod
    def encode_by_name(cls, texts, model_name=None, **kwargs) -> np.ndarray:
        """Encode texts using model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.encode(texts, **kwargs)


class OpenRouterEmbeddings(BaseEmbeddingModel):
    """
    Implementation of embedding model using OpenRouter Embeddings API.
    
    This class provides an interface to use OpenRouter-based embedding models
    with the existing retrieval system. OpenRouter supports various embedding
    models from different providers.
    """

    _instances = {}  # Dictionary to store instances by model name
    _default_model_name = "openai/text-embedding-3-small"

    def __new__(cls, model_name=None, api_key=None, **kwargs):
        if not OPENROUTER_AVAILABLE:
            raise ImportError(
                "OpenRouter client is required to use OpenRouterEmbeddings. "
                "Please ensure minions.clients.openrouter is available."
            )

        model_name = model_name or cls._default_model_name
        print(f"Using OpenRouter embedding model: {model_name}")

        # Check if we already have an instance for this model
        if model_name not in cls._instances:
            instance = super(OpenRouterEmbeddings, cls).__new__(cls)
            instance.model_name = model_name
            instance.api_key = api_key
            instance._client = OpenRouterClient(
                model_name=model_name,
                api_key=api_key,
                **kwargs
            )
            cls._instances[model_name] = instance
        
        return cls._instances[model_name]

    @staticmethod
    def _get_api_key():
        """Get API key from environment variables."""
        import os
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable."
            )
        return api_key

    def get_model(self):
        """Get the OpenRouter client."""
        return self._client

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts to create embeddings using OpenRouter model.

        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments passed to embed() method

        Returns:
            Numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        embeddings_list = []
        
        # Process texts in batches to handle API limits
        batch_size = kwargs.get('batch_size', 100)  # Reasonable batch size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Use the embed method from OpenRouterClient
                batch_embeddings = self._client.embed(
                    content=batch_texts,
                    model=self.model_name,
                    **{k: v for k, v in kwargs.items() if k != 'batch_size'}
                )
                
                # Convert to list of lists if needed
                if isinstance(batch_embeddings, list):
                    embeddings_list.extend(batch_embeddings)
                else:
                    embeddings_list.append(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Create zero embeddings as fallback
                # Try to get dimension from first successful embedding or use default
                fallback_dim = 1536  # Default for text-embedding-3-small
                if embeddings_list:
                    fallback_dim = len(embeddings_list[0])
                for _ in batch_texts:
                    embeddings_list.append([0.0] * fallback_dim)

        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings

    @classmethod
    def get_model_by_name(cls, model_name=None, api_key=None, **kwargs):
        """Get model by name (for backward compatibility)"""
        instance = cls(model_name, api_key, **kwargs)
        return instance.get_model()

    @classmethod
    def encode_by_name(cls, texts, model_name=None, api_key=None, **kwargs) -> np.ndarray:
        """Encode texts using model by name (for backward compatibility)"""
        instance = cls(model_name, api_key, **kwargs)
        return instance.encode(texts, **kwargs)


class OllamaEmbeddings(BaseEmbeddingModel):
    """
    Implementation of embedding model using Ollama Embeddings.
    
    This class provides an interface to use Ollama-based embedding models
    with the existing retrieval system. Ollama supports various local embedding
    models like llama3.2, nomic-embed-text, mxbai-embed-large, etc.
    """

    _instances = {}  # Dictionary to store instances by model name
    _default_model_name = "llama3.2"

    def __new__(cls, model_name=None):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama is required to use OllamaEmbeddings. "
                "Please install it with: pip install ollama"
            )

        model_name = model_name or cls._default_model_name
        print(f"Using Ollama embedding model: {model_name}")

        # Check if we already have an instance for this model
        if model_name not in cls._instances:
            instance = super(OllamaEmbeddings, cls).__new__(cls)
            instance.model_name = model_name
            cls._instances[model_name] = instance
        
        return cls._instances[model_name]

    def get_model(self):
        """Get the model name (Ollama doesn't use a client object)."""
        return self.model_name

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts to create embeddings using Ollama model.

        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments (currently unused)

        Returns:
            Numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        embeddings_list = []
        
        for text in texts:
            try:
                # Use the embed function from Ollama
                response = ollama_embed(
                    model=self.model_name,
                    input=text
                )
                
                # Extract embedding from response
                # Ollama returns {'embeddings': [[...values...]]} for batch or single input
                if 'embeddings' in response:
                    embedding = response['embeddings']
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # If it's a list of embeddings, take the first one
                        embeddings_list.append(embedding[0] if isinstance(embedding[0], list) else embedding)
                    else:
                        embeddings_list.append(embedding)
                elif 'embedding' in response:
                    # Fallback for singular 'embedding' key
                    embeddings_list.append(response['embedding'])
                else:
                    raise ValueError("No embeddings found in Ollama API response")
                
            except Exception as e:
                print(f"Error generating embedding for text: {e}")
                # Create zero embedding as fallback
                fallback_dim = 4096  # Default embedding dimension for llama3.2
                if embeddings_list:
                    fallback_dim = len(embeddings_list[0])
                embeddings_list.append([0.0] * fallback_dim)

        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings

    @classmethod
    def get_model_by_name(cls, model_name=None):
        """Get model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.get_model()

    @classmethod
    def encode_by_name(cls, texts, model_name=None, **kwargs) -> np.ndarray:
        """Encode texts using model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.encode(texts, **kwargs)


class LiquidAIColBERTEmbeddings(BaseEmbeddingModel):
    """
    Implementation of embedding model using Liquid AI's LFM2-ColBERT-350M.
    
    This is a late-interaction retriever that provides excellent multilingual
    performance and supports both retrieval and reranking tasks.
    
    Note: ColBERT models output multi-vector embeddings (one per token) rather
    than single-vector embeddings, enabling more expressive semantic matching.
    """

    _instances = {}  # Dictionary to store instances by model name
    _default_model_name = "LiquidAI/LFM2-ColBERT-350M"

    def __new__(cls, model_name=None):
        if not LIQUID_COLBERT_AVAILABLE:
            raise ImportError(
                "PyLate is required to use LiquidAIColBERTEmbeddings. "
                "Please install it with: pip install pylate"
            )

        model_name = model_name or cls._default_model_name
        print(f"Using Liquid AI ColBERT model: {model_name}")

        # Check if we already have an instance for this model
        if model_name not in cls._instances:
            instance = super(LiquidAIColBERTEmbeddings, cls).__new__(cls)
            instance.model_name = model_name
            instance._model = models.ColBERT(model_name_or_path=model_name)
            # Set pad token for tokenizer compatibility
            instance._model.tokenizer.pad_token = instance._model.tokenizer.eos_token
            cls._instances[model_name] = instance
        
        return cls._instances[model_name]

    def get_model(self):
        """Get the ColBERT model."""
        return self._model

    def encode(
        self, 
        texts: Union[str, List[str]], 
        is_query: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts to create ColBERT embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            is_query: Whether encoding queries (True) or documents (False)
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress during encoding
            **kwargs: Additional arguments
        
        Returns:
            Numpy array of multi-vector embeddings
            
        Note: Returns multi-vector embeddings (shape: [batch_size, num_tokens, dim])
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        # Encode using PyLate's ColBERT model
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            is_query=is_query,
            show_progress_bar=show_progress_bar,
        )
        
        return embeddings

    def rerank(
        self,
        queries: List[str],
        documents: List[List[str]],
        documents_ids: List[List[int]],
        batch_size: int = 32,
    ) -> List[List[Dict]]:
        """
        Rerank documents for given queries using ColBERT's late interaction.
        
        Args:
            queries: List of query strings
            documents: List of document lists (one list per query)
            documents_ids: List of document ID lists (one list per query)
            batch_size: Batch size for encoding
            
        Returns:
            List of reranked document results with scores
        """
        # Encode queries and documents
        queries_embeddings = self.encode(
            queries,
            is_query=True,
            batch_size=batch_size,
        )
        
        documents_embeddings = self.encode(
            documents,
            is_query=False,
            batch_size=batch_size,
        )
        
        # Rerank using PyLate's rank function
        reranked_documents = rank.rerank(
            documents_ids=documents_ids,
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
        )
        
        return reranked_documents

    @classmethod
    def get_model_by_name(cls, model_name=None):
        """Get model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.get_model()

    @classmethod
    def encode_by_name(
        cls, 
        texts, 
        model_name=None,
        is_query: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode texts using model by name (for backward compatibility)"""
        instance = cls(model_name)
        return instance.encode(texts, is_query=is_query, **kwargs)


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


def colbert_retrieve_top_k_chunks(
    queries: List[str],
    chunks: List[str] = None,
    k: int = 10,
    model_name: str = None,
    index_folder: str = "pylate-index",
    index_name: str = "colbert_index",
    batch_size: int = 32,
    recreate_index: bool = False,
) -> List[str]:
    """
    Retrieves top k chunks using Liquid AI's ColBERT late-interaction retriever.
    
    This function uses PyLate's PLAID index for efficient similarity search with
    ColBERT's multi-vector embeddings. The index is created once and reused for
    subsequent queries unless recreate_index is True.
    
    Args:
        queries: List of query strings
        chunks: List of text chunks to search through
        k: Number of top chunks to retrieve
        model_name: Optional ColBERT model name (defaults to LFM2-ColBERT-350M)
        index_folder: Folder to store the PLAID index
        index_name: Name for the index
        batch_size: Batch size for encoding
        recreate_index: Whether to recreate the index from scratch
        
    Returns:
        List of top k relevant chunks
        
    Example:
        >>> queries = ["What is machine learning?"]
        >>> chunks = ["ML is a subset of AI...", "Deep learning uses neural nets..."]
        >>> results = colbert_retrieve_top_k_chunks(queries, chunks, k=5)
    """
    if not LIQUID_COLBERT_AVAILABLE:
        raise ImportError(
            "PyLate is required for ColBERT retrieval. "
            "Please install it with: pip install pylate"
        )
    
    # Initialize the ColBERT model
    embedding_model = LiquidAIColBERTEmbeddings(model_name)
    
    # Initialize or load the PLAID index
    index = indexes.PLAID(
        index_folder=index_folder,
        index_name=index_name,
        override=recreate_index,
    )
    
    # If recreating index or index is empty, encode and add documents
    if recreate_index or not hasattr(index, '_documents_ids') or len(getattr(index, '_documents_ids', [])) == 0:
        print(f"Creating ColBERT index with {len(chunks)} documents...")
        
        # Create document IDs
        documents_ids = [str(i) for i in range(len(chunks))]
        
        # Encode documents (not queries)
        documents_embeddings = embedding_model.encode(
            chunks,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        
        # Add documents to index
        index.add_documents(
            documents_ids=documents_ids,
            documents_embeddings=documents_embeddings,
        )
        print("ColBERT index created successfully.")
    
    # Initialize the ColBERT retriever
    retriever = retrieve.ColBERT(index=index)
    
    # Encode queries
    queries_embeddings = embedding_model.encode(
        queries,
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    
    # Retrieve top-k documents
    scores_dict = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=k,
    )
    
    # Aggregate scores across all queries
    aggregated_scores = {}
    for query_scores in scores_dict:
        for doc_id, score in query_scores:
            if doc_id in aggregated_scores:
                aggregated_scores[doc_id] += score
            else:
                aggregated_scores[doc_id] = score
    
    # Sort by aggregated scores and get top k
    sorted_doc_ids = sorted(
        aggregated_scores.keys(),
        key=lambda x: aggregated_scores[x],
        reverse=True
    )[:k]
    
    # Convert document IDs back to chunk indices and return chunks
    relevant_chunks = [chunks[int(doc_id)] for doc_id in sorted_doc_ids]
    
    return relevant_chunks


def colbert_rerank_chunks(
    queries: List[str],
    chunks_list: List[List[str]],
    model_name: str = None,
    batch_size: int = 32,
) -> List[List[Dict]]:
    """
    Rerank chunks for each query using Liquid AI's ColBERT model.
    
    This is useful for reranking results from a first-stage retriever
    (like BM25 or a bi-encoder) with ColBERT's more expressive late interaction.
    
    Args:
        queries: List of query strings
        chunks_list: List of chunk lists (one list per query to rerank)
        model_name: Optional ColBERT model name (defaults to LFM2-ColBERT-350M)
        batch_size: Batch size for encoding
        
    Returns:
        List of reranked results with scores for each query
        
    Example:
        >>> queries = ["What is AI?"]
        >>> chunks_list = [["AI is...", "Machine learning is...", "Deep learning is..."]]
        >>> reranked = colbert_rerank_chunks(queries, chunks_list)
    """
    if not LIQUID_COLBERT_AVAILABLE:
        raise ImportError(
            "PyLate is required for ColBERT reranking. "
            "Please install it with: pip install pylate"
        )
    
    # Initialize the ColBERT model
    embedding_model = LiquidAIColBERTEmbeddings(model_name)
    
    # Create document IDs for each list
    documents_ids = [
        [i for i in range(len(chunks))]
        for chunks in chunks_list
    ]
    
    # Rerank using the model's rerank method
    reranked_results = embedding_model.rerank(
        queries=queries,
        documents=chunks_list,
        documents_ids=documents_ids,
        batch_size=batch_size,
    )
    
    return reranked_results