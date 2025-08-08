import logging
from typing import Any, Dict, List, Optional, Tuple
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
        **kwargs
    ):
        """Initialize the OpenRouter client.

        Args:
            model_name: The model to use (e.g., "anthropic/claude-3-5-sonnet", "openai/gpt-4o")
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the OpenRouter API. If not provided, will look for OPENROUTER_BASE_URL env var or use default.
            site_url: Optional site URL for rankings on openrouter.ai (used in HTTP-Referer header)
            site_name: Optional site name for rankings on openrouter.ai (used in X-Title header)
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

        # Call parent constructor
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.info(f"Initialized OpenRouter client with model: {model_name}")

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
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Add OpenRouter-specific headers if they exist
            extra_headers = self._get_extra_headers()
            if extra_headers:
                params["extra_headers"] = extra_headers

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
                "openrouter/horizon-beta",
                "anthropic/claude-3-5-sonnet",
                "anthropic/claude-3-5-haiku",
                "meta-llama/llama-3.1-405b-instruct",
                "google/gemini-2.0-flash",
                "mistralai/mistral-large",
            ]
