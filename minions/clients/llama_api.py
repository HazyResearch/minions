from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
from minions.clients.base import MinionsClient
from minions.clients.response import ChatResponse
import logging
import os
import openai


class LlamaApiClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.llama.com/compat/v1/",
        **kwargs
    ):
        """
        Initialize the LlamaAPI client.

        Args:
            model_name: The name of the model to use (default: "Llama-4-Maverick-17B-128E-Instruct-FP8")
            api_key: LlamaAPI API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the LlamaAPI (default: "https://api.llama.com/compat/v1/")
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        openai.api_key = api_key or os.getenv("LLAMA_API_KEY")
        # self.base_url = base_url # Handled by base

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the LLamaAPI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to llama.chat.completions.create

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

            # Only add temperature if NOT using the reasoning models
            if "reasoner" not in self.model_name:
                params["temperature"] = self.temperature

            client = openai.OpenAI(api_key=openai.api_key, base_url=self.base_url)
            response = client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during LLamaAPI API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # The content is now nested under message
        return ChatResponse(
            responses=[choice.message.content for choice in response.choices],
            usage=usage
        )