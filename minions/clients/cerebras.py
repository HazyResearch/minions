from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
import logging
import os

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    raise ImportError(
        "cerebras-cloud-sdk is required for CerebrasClient. "
        "Install it with: pip install cerebras-cloud-sdk"
    )


class CerebrasClient:
    def __init__(
        self,
        model_name: str = "llama3.1-8b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        reasoning_effort="low", #low, medium, high
        local: bool = False,
    ):
        '''
        Initialize the Cerebras client.

        Args:
            model_name: The name of the model to use (default: "llama3.1-8b")
            api_key: Cerebras API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the API (optional, uses default if not provided)
        '''
        self.model_name = model_name
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self.logger = logging.getLogger("CerebrasClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local = local
        self.reasoning_effort = reasoning_effort

        
        # Initialize the Cerebras client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = Cerebras(**client_kwargs)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        '''
        Handle chat completions using the Cerebras API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to cerebras.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        '''
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Cerebras API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # Extract finish reasons
        finish_reasons = ["stop"] * len(response.choices)
        
        # Extract response content
        if self.local:
            return [choice.message.content for choice in response.choices], usage, finish_reasons
        else:
            return [choice.message.content for choice in response.choices], usage 