from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
from minions.clients.base import MinionsClient
import logging
import os
import openai


class SambanovaClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.sambanova.ai/v1",
        local: bool = False,
        reasoning_effort=None, #low, medium, high
        **kwargs
    ):
        """
        Initialize the SambaNova client.

        Args:
            model_name: The name of the model to use (default: "Meta-Llama-3.1-8B-Instruct")
            api_key: SambaNova API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: SambaNova API base URL (default: "https://api.sambanova.ai/v1")
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
        self.api_key = api_key or os.getenv("SAMBANOVA_API_KEY")
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the SambaNova API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Create OpenAI client with SambaNova base URL
            client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort is not None else {}),
                **kwargs,
            }

            # Note: SambaNova doesn't support some OpenAI parameters like:
            # logprobs, top_logprobs, n, presence_penalty, frequency_penalty, logit_bias, seed
            # These will be ignored if passed

            response = client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during SambaNova API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # Extract finish reasons
        finish_reasons = [choice.finish_reason for choice in response.choices]
        
        # Extract content from response
        if self.local:
            return [choice.message.content for choice in response.choices], usage, finish_reasons
        else:
            return [choice.message.content for choice in response.choices], usage
