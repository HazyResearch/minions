import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from openai import OpenAI
from minions.clients.openai import OpenAIClient

from minions.usage import Usage


class TencentClient(OpenAIClient):
    """Client for Tencent Hunyuan API.

    Tencent Hunyuan uses the OpenAI API format, so we can inherit from OpenAIClient.
    Provides access to Tencent's Hunyuan models through their cloud API.
    """

    def __init__(
        self,
        model_name: str = "hunyuan-turbos-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        enable_enhancement: bool = True,
        **kwargs
    ):
        """Initialize the Tencent Hunyuan client.

        Args:
            model_name: The model to use (default: "hunyuan-turbos-latest")
            api_key: Hunyuan API key. If not provided, will look for HUNYUAN_API_KEY env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the Hunyuan API. If not provided, will look for HUNYUAN_BASE_URL env var or use default.
            enable_enhancement: Whether to enable Hunyuan's enhancement features.
            **kwargs: Additional parameters passed to base class
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("HUNYUAN_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Hunyuan API key not provided and HUNYUAN_API_KEY environment variable not set."
                )

        # Get base URL from parameter, environment variable, or use default
        if base_url is None:
            base_url = os.environ.get("HUNYUAN_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1")

        # Store Tencent-specific parameters
        self.enable_enhancement = enable_enhancement

        # Call parent constructor
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.info(f"Initialized Tencent Hunyuan client with model: {model_name}")

    def _get_extra_body(self) -> Dict[str, Any]:
        """Get Tencent-specific extra_body parameters for API requests.
        
        Returns:
            Dictionary of extra_body parameters for Hunyuan API requests
        """
        extra_body = {}
        
        if self.enable_enhancement:
            extra_body["enable_enhancement"] = True
            
        return extra_body

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Tencent Hunyuan API.

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
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Add Tencent-specific extra_body parameters
            extra_body = self._get_extra_body()
            if extra_body:
                params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**params)
            
        except Exception as e:
            self.logger.error(f"Error during Tencent Hunyuan API call: {e}")
            raise

        # Extract usage information
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        # Return response content
        return [choice.message.content for choice in response.choices], usage

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available models from Tencent Hunyuan.
        
        Returns:
            List[str]: List of model names available through Hunyuan
        """
        # Return known Hunyuan models
        return [
            "hunyuan-turbos-latest",
            "hunyuan-t1-latest",
            "hunyuan-lite",
            "hunyuan-a13b",
        ] 