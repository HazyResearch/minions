"""
Base abstract client class for all minions clients.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from minions.usage import Usage


class MinionsClient(ABC):
    """
    Abstract base class for all minions clients.
    
    This class defines the common interface that all clients must implement
    and provides shared initialization and utility methods.
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize the client with common parameters.
        
        Args:
            model_name: Name/identifier of the model to use (required)
            temperature: Sampling temperature (optional, client-specific defaults apply)
            max_tokens: Maximum number of tokens to generate (optional, client-specific defaults apply)
            api_key: API key for authentication (optional, if required by client)
            base_url: Custom API endpoint URL (optional, if supported by client)
            verbose: Enable verbose logging/output (optional, client-specific defaults apply)
            **kwargs: Additional client-specific parameters
        """
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Only set attributes if they were explicitly provided
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        if verbose is not None:
            self.verbose = verbose
            # Set logging level based on verbose flag if provided
            if verbose:
                self.logger.setLevel(logging.INFO)
        
        # Store additional kwargs for client-specific initialization
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[List[str], Usage, ...]:
        """
        Primary chat interface that all clients must implement.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters specific to the client
            
        Returns:
            Tuple containing at minimum:
            - List[str]: Generated responses
            - Usage: Token usage information
            
            May also include:
            - List[str]: Finish reasons (optional)
            - List[Any]: Tool calls (optional)
        """
        pass
    
    def embed(
        self, 
        content: Union[str, List[str]], 
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for the given content.
        
        Args:
            content: Text content to embed (single string or list of strings)
            **kwargs: Additional parameters specific to the client
            
        Returns:
            List of embedding vectors
            
        Raises:
            NotImplementedError: If the client doesn't support embeddings
        """
        raise NotImplementedError(f"Embedding not supported by {self.__class__.__name__}")
    
    def complete(
        self, 
        prompts: Union[str, List[str]], 
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Generate text completions for the given prompts.
        
        Args:
            prompts: Prompt text(s) to complete
            **kwargs: Additional parameters specific to the client
            
        Returns:
            Tuple of:
            - List[str]: Generated completions
            - Usage: Token usage information
            
        Raises:
            NotImplementedError: If the client doesn't support text completion
        """
        raise NotImplementedError(f"Text completion not supported by {self.__class__.__name__}")
    
    def __str__(self) -> str:
        """String representation of the client."""
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the client."""
        attrs = [f"model_name='{self.model_name}'"]
        if hasattr(self, 'temperature'):
            attrs.append(f"temperature={self.temperature}")
        if hasattr(self, 'max_tokens'):
            attrs.append(f"max_tokens={self.max_tokens}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"