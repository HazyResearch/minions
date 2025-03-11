import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from openai import OpenAI
from minions.clients.openai import OpenAIClient

from minions.usage import Usage


class OpenRouterClient(OpenAIClient):
    """Client for OpenRouter API, which provides access to various LLMs through a unified API.

    OpenRouter uses the OpenAI API format, so we can inherit from OpenAIClient.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenRouter client.

        Args:
            model_name: The model to use (e.g., "anthropic/claude-3-5-sonnet")
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the OpenRouter API. If not provided, will look for OPENROUTER_BASE_URL env var or use default.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set."
                )

        # Get base URL from parameter, environment variable, or use default
        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Initialize the OpenAI client with the OpenRouter base URL
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger("OpenRouterClient")
        self.logger.setLevel(logging.INFO)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI  client, but route to perplexity

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # add a system prompt to the top of the messages

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                **kwargs,
            }

            params["temperature"] = self.temperature

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during OpenRouter API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage
