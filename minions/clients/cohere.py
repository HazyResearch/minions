import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class CohereClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "command-a-03-2025",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        local: bool = False,
        **kwargs
    ):
        """
        Initialize the Cohere client using OpenAI SDK with Cohere's compatibility API.

        Args:
            model_name: The name of the Cohere model to use (default: "command-a-03-2025")
            api_key: Cohere API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Cohere compatibility API (default: Cohere's compatibility endpoint)
            local: If this is communicating with a local client (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.base_url = base_url or "https://api.cohere.ai/compatibility/v1"
        
        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. Set COHERE_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize the OpenAI client with Cohere's compatibility endpoint
        self.client = openai.OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url
        )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Cohere compatibility API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat completions API

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Cohere API call: {e}")
            raise

        # Extract usage information if it exists
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        # Extract response content
        if self.local:
            return [choice.message.content for choice in response.choices], usage, [choice.finish_reason for choice in response.choices]
        else:
            return [choice.message.content for choice in response.choices], usage

    def embed(
        self, 
        content: Union[str, List[str]], 
        model: str = "embed-v4.0",
        encoding_format: str = "float",
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using Cohere's embedding API through the compatibility endpoint.
        
        Args:
            content: Text content to embed (single string or list of strings)
            model: Embedding model to use (default: "embed-v4.0")
            encoding_format: Format of embeddings ("float" or "base64", default: "float")
            **kwargs: Additional parameters for the embeddings API
            
        Returns:
            List of embedding vectors
        """
        try:
            # Ensure content is a list
            if isinstance(content, str):
                content = [content]
            
            response = self.client.embeddings.create(
                input=content,
                model=model,
                encoding_format=encoding_format,
                **kwargs
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            self.logger.error(f"Error during Cohere embedding API call: {e}")
            raise

    def list_models(self):
        """
        List available models from the Cohere API.
        
        Returns:
            Dict containing the models data from the Cohere API response
        """
        try:
            response = self.client.models.list()
            return {
                "object": "list",
                "data": [model.model_dump() for model in response.data]
            }
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise 