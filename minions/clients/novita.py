import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from minions.clients.openai import OpenAIClient

from minions.usage import Usage


class NovitaClient(OpenAIClient):
    """Client for Novita AI API, which provides access to various LLMs through OpenAI-compatible API.

    Novita AI uses the OpenAI API format, so we can inherit from OpenAIClient.
    Novita AI provides access to various AI models with competitive pricing and performance.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Novita client.

        Args:
            model_name: The model to use (e.g., "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-70b-instruct")
            api_key: Novita AI API key. If not provided, will look for NOVITA_API_KEY env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the Novita API. If not provided, will look for NOVITA_BASE_URL env var or use default.
            **kwargs: Additional parameters passed to base class
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("NOVITA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Novita API key not provided and NOVITA_API_KEY environment variable not set. "
                    "Get your API key from: https://novita.ai/settings/key-management"
                )

        # Get base URL from parameter, environment variable, or use default
        if base_url is None:
            base_url = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/v3/openai")

        # Call parent constructor
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.info(f"Initialized Novita client with model: {model_name}")

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available models from Novita AI.
        
        Returns:
            List[str]: List of model names available through Novita AI
        """
        try:
            import requests
            
            api_key = os.environ.get("NOVITA_API_KEY")
            if not api_key:
                raise ValueError("NOVITA_API_KEY environment variable not set")
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.novita.ai/v3/openai/models", headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
            
        except Exception as e:
            logging.error(f"Failed to get Novita model list: {e}")
            # Return some common models as fallback based on documentation
            return [
                "moonshotai/kimi-k2-0905",
                "deepseek/deepseek-v3.1",
                "meta-llama/llama-3.1-8b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-405b-instruct",
                "mistralai/mistral-7b-instruct",
                "mistralai/mixtral-8x7b-instruct",
                "microsoft/wizardlm-2-8x22b",
                "google/gemma-2-9b-it",
                "qwen/qwen2.5-72b-instruct",
            ] 