import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from openai import OpenAI
from minions.clients.openai import OpenAIClient

from minions.usage import Usage


class OpenRouterClient(OpenAIClient):
    """Client for OpenRouter API, which provides access to various LLMs through a unified API.

    OpenRouter uses the OpenAI API format, so we can inherit from OpenAIClient.
    OpenRouter provides access to hundreds of AI models through a single endpoint with automatic
    fallbacks and cost optimization.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        verbosity: str = "medium",
        use_responses_api: bool = False,
        reasoning_effort: str = "low",
        **kwargs
    ):
        """Initialize the OpenRouter client.

        Args:
            model_name: The model to use (e.g., "anthropic/claude-3-5-sonnet", "openai/gpt-4o", "openai/o4-mini")
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the OpenRouter API. If not provided, will look for OPENROUTER_BASE_URL env var or use default.
            site_url: Optional site URL for rankings on openrouter.ai (used in HTTP-Referer header)
            site_name: Optional site name for rankings on openrouter.ai (used in X-Title header)
            verbosity: Controls the verbosity and length of the model response. Options: "low", "medium", "high". Default: "medium"
            use_responses_api: Whether to use responses API for reasoning models (default: False)
            reasoning_effort: Reasoning effort level for reasoning models (default: "low")
            **kwargs: Additional parameters passed to base class
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set."
                )

        # Get base URL from parameter, environment variable, or use default
        if base_url is None:
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Store OpenRouter-specific headers for rankings
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL")
        self.site_name = site_name or os.environ.get("OPENROUTER_SITE_NAME")
        
        # Validate and store verbosity parameter
        valid_verbosity_levels = ["low", "medium", "high"]
        if verbosity not in valid_verbosity_levels:
            raise ValueError(f"Invalid verbosity level '{verbosity}'. Must be one of: {valid_verbosity_levels}")
        self.verbosity = verbosity

        # Store responses API settings
        # Automatically use responses API for o4 models
        if "o4" in model_name or "o1" in model_name or "o3" in model_name:
            self.use_responses_api = True
        else:
            self.use_responses_api = use_responses_api
        self.reasoning_effort = reasoning_effort

        # Call parent constructor (but don't pass use_responses_api/reasoning_effort to avoid double handling)
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            use_responses_api=False,  # We handle this in OpenRouterClient
            **kwargs
        )

        # Reinitialize client with default headers if we have OpenRouter-specific headers
        # This ensures headers are sent with all requests (chat, embeddings, etc.)
        extra_headers = self._get_extra_headers()
        if extra_headers:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=extra_headers
            )

        self.logger.info(f"Initialized OpenRouter client with model: {model_name}, verbosity: {verbosity}, use_responses_api: {self.use_responses_api}")

    def _get_extra_headers(self) -> Dict[str, str]:
        """Get OpenRouter-specific headers for API requests.
        
        Returns:
            Dictionary of extra headers for OpenRouter API requests
        """
        headers = {}
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
            
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        return headers

    def responses(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenRouter Responses API.
        
        The Responses API is designed for reasoning models and provides extended thinking capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the responses API
        
        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # Handle response_format conversion if present
        if "response_format" in kwargs:
            kwargs["text"] = {"format": kwargs["response_format"]}
            del kwargs["response_format"]

        try:
            # Convert messages to responses API format if needed
            input_messages = messages
            
            # Replace "system" role with "developer" for responses API
            for message in input_messages:
                if message["role"] == "system":
                    message["role"] = "developer"

            params = {
                "model": self.model_name,
                "input": input_messages,
                "max_output_tokens": self.max_tokens,
                "stream": False,
                **kwargs,
            }
            
            # Add reasoning effort for reasoning models
            if "o1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name:
                params["reasoning"] = {"effort": self.reasoning_effort}

            response = self.client.responses.create(**params)
            
        except Exception as e:
            self.logger.error(f"Error during OpenRouter Responses API call: {e}")
            raise

        # Extract output text from response
        outputs = []
        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if output_item.type == "message" and hasattr(output_item, 'content'):
                    for content_part in output_item.content:
                        if content_part.type == "output_text":
                            outputs.append(content_part.text)

        # Extract usage information if it exists
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

        return outputs, usage

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenRouter API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     For file inputs, content should be a list of dictionaries with:
                     - text messages: {"type": "text", "text": "your message"}
                     - file messages: {
                         "type": "file",
                         "file": {
                             "filename": "document.pdf",
                             "file_data": "https://example.com/path/to/document.pdf"
                         }
                     }
            **kwargs: Additional arguments to pass to openai.chat.completions.create
                     For PDF processing, you can specify:
                     plugins=[{
                         "id": "file-parser",
                         "pdf": {
                             "engine": "pdf-text" # or "mistral-ocr" or "native"
                         }
                     }]

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        # Use responses API for reasoning models or when explicitly requested
        if self.use_responses_api:
            return self.responses(messages, **kwargs)
        
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                "verbosity": self.verbosity,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
            
        except Exception as e:
            self.logger.error(f"Error during OpenRouter API call: {e}")
            raise

        # Extract usage information
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        # Return response content and any file annotations if present
        responses = []
        for choice in response.choices:
            content = choice.message.content
            # If there are file annotations, include them in the response
            if hasattr(choice.message, 'annotations'):
                responses.append({
                    'content': content,
                    'annotations': choice.message.annotations
                })
            else:
                responses.append(content)

        return responses, usage

    def embed(
        self,
        content: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using the OpenRouter API.

        Args:
            content: Text content to embed (single string or list of strings).
            model: Embedding model to use (e.g., "openai/text-embedding-3-small", "openai/text-embedding-ada-002").
                   If not provided, uses the instance model_name.
            encoding_format: Format of embeddings ("float" or "base64", default: "float").
            **kwargs: Additional parameters for the embeddings API.

        Returns:
            List of embedding vectors.

        Note:
            OpenRouter supports various embedding models. Check https://openrouter.ai/models?output_modalities=embeddings
            for available embedding models.
        """
        try:
            # Ensure content is a list
            if isinstance(content, str):
                content = [content]

            # Use provided model or fall back to instance model
            embedding_model = model or self.model_name

            params = {
                "input": content,
                "model": embedding_model,
                "encoding_format": encoding_format,
                **kwargs,
            }

            response = self.client.embeddings.create(**params)

            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            self.logger.error(f"Error generating embeddings via OpenRouter: {e}")
            raise

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available models from OpenRouter.
        
        Returns:
            List[str]: List of model names available through OpenRouter
        """
        try:
            import requests
            
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
            
        except Exception as e:
            logging.error(f"Failed to get OpenRouter model list: {e}")
            # Return some common models as fallback
            return [
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/o4-mini",
                "openai/o1-preview",
                "openai/o1-mini",
                "openai/o3-mini",
                "openrouter/horizon-beta",
                "anthropic/claude-3-5-sonnet",
                "anthropic/claude-3-5-haiku",
                "meta-llama/llama-3.1-405b-instruct",
                "google/gemini-2.0-flash",
                "mistralai/mistral-large",
            ]
