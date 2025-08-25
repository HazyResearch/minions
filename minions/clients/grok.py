from typing import Any, Dict, List, Optional, Tuple, Union
from minions.usage import Usage
from minions.clients.base import MinionsClient
import logging
import os
import openai


class GrokClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "grok-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.x.ai/v1",
        local: bool = False,
        enable_reasoning_output: bool = False,
        **kwargs
    ):
        """
        Initialize the Grok client.

        Args:
            model_name: The name of the model to use (default: "grok-4")
            api_key: Grok API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Grok API (default: "https://api.x.ai/v1")
            reasoning_effort: Reasoning effort level for reasoning models ("low", "medium", "high", default: None)
            enable_reasoning_output: Whether to include reasoning traces in output (default: False)
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
        openai.api_key = api_key or os.getenv("XAI_API_KEY")
        self.enable_reasoning_output = enable_reasoning_output
        # self.base_url = base_url # Handled by base

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available Grok models from the X.AI API.

        Returns:
            List[str]: List of model names available through X.AI
        """
        try:
            import requests
            
            # Try to use API key from environment or provided key
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                logging.warning("No XAI_API_KEY found in environment variables")
                # Return default models if no API key
                return []

            # Make API call to list models
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.x.ai/v1/models", headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            model_names = [model["id"] for model in models_data.get("data", [])]
            
            # If we got models from API, return them
            if model_names:
                return sorted(model_names, reverse=True)  # Sort with newest first
            # Fallback to default models if API returned empty
            return []
        except Exception as e:
            logging.error(f"Failed to get Grok model list: {e}")
            # Return fallback models on error
            return []

    def _is_reasoning_model(self, model_name: str) -> bool:
        """
        Check if the given model supports reasoning.
        
        Args:
            model_name: The model name to check
            
        Returns:
            bool: True if the model supports reasoning
        """
        reasoning_models = ["grok-3-mini", "grok-3-mini-fast", "grok-4"]
        return any(reasoning_model in model_name.lower() for reasoning_model in reasoning_models)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Union[Tuple[List[str], Usage], Tuple[List[str], Usage, List[str]], Tuple[List[str], Usage, List[str], List[Optional[str]]]]:
        """
        Handle chat completions using the Grok API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to grok.chat.completions.create

        Returns:
            Tuple containing response strings, token usage, and optionally finish reasons and reasoning content
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                **kwargs,
            }

            # Handle reasoning parameters for reasoning models
            is_reasoning_model = self._is_reasoning_model(self.model_name)
            
            if is_reasoning_model:
                # Reasoning models don't use temperature the same way
                if "temperature" not in kwargs:
                    # Only add temperature if not explicitly provided and model doesn't use reasoning
                    pass  # Don't set temperature for reasoning models unless explicitly requested
            else:
                # Regular models use temperature normally
                params["temperature"] = self.temperature

            client = openai.OpenAI(api_key=openai.api_key, base_url=self.base_url)
            response = client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Grok API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # Extract finish reasons
        finish_reasons = [choice.finish_reason for choice in response.choices]
        
        # Extract response content
        response_content = [choice.message.content for choice in response.choices]
        
        # Extract reasoning content if available and enabled
        reasoning_content = []
        if self.enable_reasoning_output and is_reasoning_model:
            for choice in response.choices:
                reasoning = getattr(choice.message, 'reasoning_content', None)
                reasoning_content.append(reasoning)

        # Return appropriate tuple based on what's requested
        if self.local:
            if self.enable_reasoning_output and reasoning_content and any(r is not None for r in reasoning_content):
                return f"{reasoning_content} \n {response_content}", usage, finish_reasons
            else:
                return response_content, usage, finish_reasons
        else:
            if self.enable_reasoning_output and reasoning_content and any(r is not None for r in reasoning_content):
                return f"{reasoning_content} \n {response_content}", usage
            else:
                return response_content, usage

   
