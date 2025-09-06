from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
from minions.clients.base import MinionsClient
import logging
import os
import openai


class QwenClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "qwen3-max",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        local: bool = False,
        **kwargs
    ):
        """
        Initialize the Qwen client.

        Args:
            model_name: The name of the model to use (default: "qwen-plus")
            api_key: Qwen API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Qwen API (default: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
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
        openai.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        # self.base_url = base_url # Handled by base

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Qwen API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to qwen.chat.completions.create

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
            if "deep_thinking" in kwargs:
                params["deep_thinking"] = kwargs["deep_thinking"]
                del kwargs["deep_thinking"]
                extra_body = {
                    "deep_thinking": True,
                }
                params.update(extra_body)

            client = openai.OpenAI(api_key=openai.api_key, base_url=self.base_url)
            response = client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Qwen API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # Extract finish reasons
        finish_reasons = [choice.finish_reason for choice in response.choices]

        # The content is now nested under message
        if self.local:
            return [choice.message.content for choice in response.choices], usage, finish_reasons
        else:
            return [choice.message.content for choice in response.choices], usage 