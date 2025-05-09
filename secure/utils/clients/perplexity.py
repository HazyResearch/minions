import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage


class PerplexityAIClient:
    def __init__(
        self,
        model_name: str = "sonar-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Perplexity client.

        Args:
            model_name: The name of the model to use (default: "sonar")
            api_key: Perplexity API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Perplexity API (optional, falls back to PERPLEXITY_BASE_URL environment variable or default URL)
        """
        self.model_name = model_name
        openai.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.api_key = openai.api_key
        self.logger = logging.getLogger("PerplexityAIClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get base URL from parameter, environment variable, or use default
        base_url = base_url or os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
        
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=base_url
        )

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
        messages.insert(
            0,
            {
                "role": "system",
                "content": "You are language model that has access to the internet if you need it.",
            },
        )

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
            self.logger.error(f"Error during Sonar API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage
