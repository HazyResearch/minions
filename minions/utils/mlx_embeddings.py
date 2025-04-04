import numpy as np
from typing import List, Union, Optional
import os
from .retrievers import BaseEmbeddingModel

try:
    import mlx.core as mx
    from mlx_embeddings.utils import load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


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
        return embeddings
