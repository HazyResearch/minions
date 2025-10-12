import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class ExaClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "exa",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Exa client.

        Args:
            model_name: The name of the model to use (default: "exa")
            api_key: Exa API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Exa API (optional, falls back to EXA_BASE_URL environment variable or default URL)
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
        openai.api_key = api_key or os.getenv("EXA_API_KEY")
        self.api_key = openai.api_key
        
        # Get base URL from parameter, environment variable, or use default
        base_url = base_url or os.getenv("EXA_BASE_URL", "https://api.exa.ai")
        
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=base_url
        )

    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI client, but route to Exa.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            output_schema: Optional JSON schema to structure the output
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Add output_schema to extra_body if provided
            if output_schema:
                if "extra_body" not in params:
                    params["extra_body"] = {}
                params["extra_body"]["output_schema"] = output_schema

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Exa API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        
        # Extract cost information if available
        try:
            if hasattr(response.usage, 'cost') and response.usage.cost:
                cost = response.usage.cost
                usage.input_tokens_cost = getattr(cost, 'input_tokens_cost', None)
                usage.output_tokens_cost = getattr(cost, 'output_tokens_cost', None)
                usage.request_cost = getattr(cost, 'request_cost', None)
                usage.total_cost = getattr(cost, 'total_cost', None)
                
                # Log cost information if available
                if usage.total_cost is not None:
                    self.logger.info(f"Exa API cost: ${usage.total_cost:.6f} (input: ${usage.input_tokens_cost:.6f}, output: ${usage.output_tokens_cost:.6f}, request: ${usage.request_cost:.6f})")
        except AttributeError:
            # Cost information not available in this response
            pass

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage

    @staticmethod
    def get_available_models():
        """
        Get a list of available models from Exa.
        
        Returns:
            List[str]: List of model names available through Exa API
        """
        return [
            "exa",  # Default Exa model
        ]

