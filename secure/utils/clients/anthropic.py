import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import anthropic

from minions.usage import Usage


class AnthropicClient:
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        """
        Initialize the Anthropic client.

        Args:
            model_name: The name of the model to use (default: "claude-3-sonnet-20240229")
            api_key: Anthropic API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 2048)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.logger = logging.getLogger("AnthropicClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Anthropic API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.messages.create

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

            response = self.client.messages.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Anthropic API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens
        )

        return [response.content[0].text], usage 