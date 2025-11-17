import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI

from minions.usage import Usage
from minions.clients.base import MinionsClient


class AzureOpenAIClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        local: bool = False,
        **kwargs
    ):
        """
        Initialize the Azure OpenAI client using the v1 API.

        Args:
            model_name: The name of the model deployment to use (default: "gpt-4o")
            api_key: Azure OpenAI API key (optional, falls back to AZURE_OPENAI_API_KEY environment variable)
            azure_endpoint: Azure OpenAI endpoint URL (optional, falls back to AZURE_OPENAI_ENDPOINT environment variable)
                          Format: https://YOUR-RESOURCE-NAME.openai.azure.com
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            **kwargs: Additional parameters passed to base class
            
        Reference:
            https://learn.microsoft.com/en-us/azure/ai-foundry/openai/supported-languages
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
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required. Set it via the api_key parameter or AZURE_OPENAI_API_KEY environment variable.")
        
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set it via the azure_endpoint parameter or AZURE_OPENAI_ENDPOINT environment variable.")
        
        # Initialize the OpenAI client with Azure v1 API endpoint
        # Format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
        base_url = self.azure_endpoint
        if not base_url.endswith('/'):
            base_url += '/'
        if not base_url.endswith('openai/v1/'):
            # Add /openai/v1/ suffix if not present
            if base_url.endswith('openai/'):
                base_url += 'v1/'
            elif '/openai/v1/' not in base_url:
                base_url += 'openai/v1/'
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.logger.info(f"Initialized Azure OpenAI client with v1 API endpoint: {base_url}")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Azure OpenAI API.
        
        Supports both standard models and reasoning models (o1, o3, o4).
        For reasoning models, temperature is not supported and special parameters
        like reasoning_effort can be used.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create
                     Common kwargs:
                     - max_tokens: Override default max_tokens
                     - temperature: Override default temperature (ignored for reasoning models)
                     - reasoning_effort: For reasoning models (low, medium, high)
                     - stream: Enable streaming responses

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                **kwargs,
            }

            # Check if this is a reasoning model (o1, o3, o4 series)
            is_reasoning_model = any(
                model_prefix in self.model_name.lower() 
                for model_prefix in ["o1", "o3", "o4"]
            )

            if is_reasoning_model:
                # Reasoning models don't support temperature
                # Use max_completion_tokens instead of max_tokens
                if "max_tokens" not in kwargs:
                    params["max_completion_tokens"] = self.max_tokens
                    if "max_tokens" in params:
                        del params["max_tokens"]
                
                # Remove temperature if present (not supported for reasoning models)
                params.pop("temperature", None)
                
                self.logger.info(f"Using reasoning model: {self.model_name}")
            else:
                # Standard models support temperature and max_tokens
                if "temperature" not in kwargs:
                    params["temperature"] = self.temperature
                if "max_tokens" not in kwargs:
                    params["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**params)
            
        except openai.APIConnectionError as e:
            self.logger.error(f"The server could not be reached: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit exceeded (429): {e}")
            raise
        except openai.APIStatusError as e:
            self.logger.error(f"API error (status {e.status_code}): {e}")
            raise
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