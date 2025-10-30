import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from together import Together

from minions.usage import Usage
from minions.clients.base import MinionsClient
from minions.clients.response import ChatResponse


class TogetherClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        local: bool = False,
        **kwargs
    ):
        """
        Initialize the Together client.

        Args:
            model_name: The name of the model to use (default: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
            api_key: Together API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.client = Together(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Together API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.chat.completions.create

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
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
            self.logger.error(f"Error during Together API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
        # Extract done reasons (finish_reason in OpenAI-compatible APIs)
        done_reasons = [choice.finish_reason for choice in response.choices]
        return ChatResponse(
            responses=[choice.message.content for choice in response.choices],
            usage=usage,
            done_reasons=done_reasons if self.local else None
        )
