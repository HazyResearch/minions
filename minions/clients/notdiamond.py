import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class NotDiamondAIClient(MinionsClient):
    @staticmethod
    def get_supported_providers() -> str:
        """
        Returns a formatted string listing the supported AI providers.
        
        Returns:
            str: A string listing major AI providers supported by NotDiamond
        """
        providers = [
            "OpenAI",
            "Anthropic",
            "Google",
            "Mistral AI",
            "Together AI",
            "Perplexity AI"
        ]
        return "Supported AI Providers:\n" + "\n".join(f"- {provider}" for provider in providers)

    def __init__(
        self,
        model_name: str = "notdiamond",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        models: Optional[List[str]] = None,
        tradeoff: Optional[str] = "cost",
        preference_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the NotDiamond AI client.

        Args:
            model_name: The name of the model to use (default: "notdiamond")
            api_key: NotDiamond API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the NotDiamond API (optional, falls back to NOTDIAMOND_BASE_URL environment variable or default URL)
            models: List of LLM options to route between (e.g. ["gpt-4", "claude-3-5-sonnet"])
            tradeoff: Routing strategy ("cost", "latency", etc.)
            preference_id: Optional preference ID for routing
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
        openai.api_key = api_key or os.getenv("NOTDIAMOND_API_KEY")
        self.api_key = openai.api_key
        
        # Get base URL from parameter, environment variable, or use default
        base_url = base_url or os.getenv("NOTDIAMOND_BASE_URL", "https://proxy.notdiamond.ai/v1/proxy")
        
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=base_url
        )

        # NotDiamond specific parameters
        self.models = models or ["gpt-4", "claude-3-5-sonnet"]
        self.tradeoff = tradeoff
        self.preference_id = preference_id

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI client, but route to NotDiamond

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Prepare the extra body parameters for NotDiamond
            extra_body = {
                "models": kwargs.pop("models", self.models),
                "tradeoff": kwargs.pop("tradeoff", self.tradeoff),
            }
            
            # Only add preference_id if it's set
            if self.preference_id or kwargs.get("preference_id"):
                extra_body["preference_id"] = kwargs.pop("preference_id", self.preference_id)

            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "extra_body": extra_body,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during NotDiamond API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return [choice.message.content for choice in response.choices], usage 