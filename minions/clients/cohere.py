import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient

# Try to import the native Cohere SDK
try:
    import cohere
    COHERE_SDK_AVAILABLE = True
except ImportError:
    COHERE_SDK_AVAILABLE = False


class CohereClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "command-a-03-2025",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        local: bool = False,
        use_native_sdk: bool = True,
        **kwargs
    ):
        """
        Initialize the Cohere client.
        
        Supports two modes:
        1. Native Cohere SDK (v2) - default when SDK is available
        2. OpenAI compatibility API - fallback when SDK unavailable or disabled

        Args:
            model_name: The name of the Cohere model to use (default: "command-a-03-2025")
            api_key: Cohere API key (optional, falls back to COHERE_API_KEY or CO_API_KEY env vars)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Cohere compatibility API (default: Cohere's compatibility endpoint)
            local: If this is communicating with a local client (default: False)
            use_native_sdk: If True, use native Cohere SDK when available (default: True)
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
        self.api_key = api_key or os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        self.base_url = base_url or "https://api.cohere.ai/compatibility/v1"
        self.use_native_sdk = use_native_sdk and COHERE_SDK_AVAILABLE
        
        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. Set COHERE_API_KEY or CO_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize clients based on configuration
        self.cohere_client = None
        self.cohere_client_v2 = None
        self.openai_client = None
        
        if self.use_native_sdk:
            # Initialize native Cohere SDK clients
            # ClientV2 for chat (uses messages format)
            self.cohere_client_v2 = cohere.ClientV2(api_key=self.api_key)
            # Client for embed and rerank (v1 API methods)
            self.cohere_client = cohere.Client(api_key=self.api_key)
            self.logger.info("Using native Cohere SDK")
        else:
            # Initialize the OpenAI client with Cohere's compatibility endpoint
            self.openai_client = openai.OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url
            )
            self.logger.info("Using OpenAI compatibility API")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using either native Cohere SDK or OpenAI compatibility API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat API

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        if self.use_native_sdk:
            return self._chat_native(messages, **kwargs)
        else:
            return self._chat_openai_compat(messages, **kwargs)

    def _chat_native(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat using native Cohere SDK v2.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat API
            
        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        try:
            # Build parameters for Cohere API
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            # Add max_tokens if specified
            if self.max_tokens:
                params["max_tokens"] = self.max_tokens
            
            # Merge any additional kwargs
            params.update(kwargs)
            
            response = self.cohere_client_v2.chat(**params)
            
        except Exception as e:
            self.logger.error(f"Error during Cohere native SDK call: {e}")
            raise

        # Extract usage information
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        if hasattr(response, 'usage') and response.usage is not None:
            usage = Usage(
                prompt_tokens=getattr(response.usage, 'input_tokens', 0) or getattr(response.usage, 'prompt_tokens', 0),
                completion_tokens=getattr(response.usage, 'output_tokens', 0) or getattr(response.usage, 'completion_tokens', 0),
            )

        # Extract response content from v2 API format
        # v2 API: response.message.content[0].text
        content = ""
        if hasattr(response, 'message') and response.message is not None:
            if hasattr(response.message, 'content') and response.message.content:
                # content is a list of content blocks
                content_parts = []
                for block in response.message.content:
                    if hasattr(block, 'text'):
                        content_parts.append(block.text)
                content = "".join(content_parts)

        if self.local:
            finish_reason = getattr(response, 'finish_reason', 'stop')
            return [content], usage, [finish_reason]
        else:
            return [content], usage

    def _chat_openai_compat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat using OpenAI compatibility API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat API
            
        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            response = self.openai_client.chat.completions.create(**params)
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
        input_type: str = "search_document",
        encoding_format: str = "float",
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using Cohere's embedding API.
        
        Args:
            content: Text content to embed (single string or list of strings)
            model: Embedding model to use (default: "embed-v4.0")
            input_type: Type of input for embedding ("search_document", "search_query", 
                       "classification", "clustering", default: "search_document")
            encoding_format: Format of embeddings ("float" or "base64", default: "float")
            **kwargs: Additional parameters for the embeddings API
            
        Returns:
            List of embedding vectors
        """
        # Ensure content is a list
        if isinstance(content, str):
            content = [content]
            
        if self.use_native_sdk:
            return self._embed_native(content, model, input_type, **kwargs)
        else:
            return self._embed_openai_compat(content, model, encoding_format, **kwargs)

    def _embed_native(
        self, 
        texts: List[str], 
        model: str,
        input_type: str,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using native Cohere SDK.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            input_type: Type of input for embedding
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model=model,
                input_type=input_type,
                **kwargs
            )
            
            return response.embeddings
            
        except Exception as e:
            self.logger.error(f"Error during Cohere native embed call: {e}")
            raise

    def _embed_openai_compat(
        self, 
        texts: List[str], 
        model: str,
        encoding_format: str,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI compatibility API.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            encoding_format: Format of embeddings
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=model,
                encoding_format=encoding_format,
                **kwargs
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            self.logger.error(f"Error during Cohere embedding API call: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Union[str, Dict[str, str]]],
        model: str = "rerank-v3.5",
        top_n: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to a query using Cohere's rerank API.
        
        Requires native Cohere SDK.
        
        Args:
            query: The query to rank documents against
            documents: List of documents to rerank (strings or dicts with 'text' key)
            model: Rerank model to use (default: "rerank-v3.5")
            top_n: Number of top results to return (default: all documents)
            return_documents: Whether to return document text in results (default: True)
            **kwargs: Additional parameters for the rerank API
            
        Returns:
            List of reranked results with 'index', 'relevance_score', and optionally 'document'
            
        Raises:
            RuntimeError: If native Cohere SDK is not available
        """
        if not self.use_native_sdk:
            raise RuntimeError(
                "Rerank requires the native Cohere SDK. Install with: pip install -U cohere"
            )
            
        try:
            params = {
                "query": query,
                "documents": documents,
                "model": model,
                "return_documents": return_documents,
            }
            
            if top_n is not None:
                params["top_n"] = top_n
                
            params.update(kwargs)
            
            response = self.cohere_client.rerank(**params)
            
            # Convert response to list of dicts
            results = []
            for result in response.results:
                item = {
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                }
                if return_documents and hasattr(result, 'document'):
                    item["document"] = result.document
                results.append(item)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error during Cohere rerank call: {e}")
            raise

    def list_models(self):
        """
        List available models from the Cohere API.
        
        Returns:
            Dict containing the models data from the Cohere API response
        """
        if self.use_native_sdk:
            try:
                response = self.cohere_client.models.list()
                return {
                    "object": "list",
                    "data": [model.model_dump() if hasattr(model, 'model_dump') else vars(model) for model in response.models]
                }
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                raise
        else:
            try:
                response = self.openai_client.models.list()
                return {
                    "object": "list",
                    "data": [model.model_dump() for model in response.data]
                }
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                raise
