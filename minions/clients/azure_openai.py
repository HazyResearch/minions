import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AzureOpenAI

from minions.usage import Usage


class AzureOpenAIClient:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize the Azure OpenAI client.

        Args:
            model_name: The name of the model deployment to use (default: "gpt-4o")
            api_key: Azure OpenAI API key (optional, falls back to environment variable if not provided)
            api_version: Azure OpenAI API version (optional, falls back to environment variable if not provided)
            azure_endpoint: Azure OpenAI endpoint URL (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required. Set it via the api_key parameter or AZURE_OPENAI_API_KEY environment variable.")
        
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set it via the azure_endpoint parameter or AZURE_OPENAI_ENDPOINT environment variable.")
        
        # Initialize the Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )
        
        self.logger = logging.getLogger("AzureOpenAIClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Azure OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                **kwargs,
            }

            # Only add temperature if NOT using the reasoning models (e.g., o3-mini model)
            if "o1" not in self.model_name and "o3" not in self.model_name:
                params["temperature"] = self.temperature

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Azure OpenAI API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage 