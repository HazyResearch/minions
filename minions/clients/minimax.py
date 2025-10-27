import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os

from minions.clients.openai import OpenAIClient
from minions.usage import Usage


class MiniMaxClient(OpenAIClient):
    """Client for MiniMax API (OpenAI-compatible API).

    MiniMax offers state-of-the-art language models with OpenAI API compatibility.
    The MiniMax-M2 model excels at text generation, reasoning, and function calling.

    Key Features:
    - OpenAI-compatible API for easy integration
    - High-quality text generation and reasoning
    - Support for streaming responses
    - Function calling (tool use) capabilities
    - Competitive pricing and performance

    Reference: https://platform.minimax.io/docs/guides/text-generation
    """

    def __init__(
        self,
        model_name: str = "MiniMax-M2",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the MiniMax client.

        Args:
            model_name: The model to use (default: "MiniMax-M2").
                Only "MiniMax-M2" is currently supported.
            api_key: MiniMax API key. If not provided, uses MINIMAX_API_KEY env var.
            temperature: Temperature parameter for generation (range: 0.0 < temp <= 1.0).
                Default is 1.0 (recommended value). Values outside this range will return an error.
            max_tokens: Maximum number of tokens to generate (default: 1024).
            base_url: Base URL for the MiniMax API. If not provided, uses
                MINIMAX_BASE_URL env var or defaults to "https://api.minimax.io/v1".
            **kwargs: Additional parameters passed to base class.

        Note:
            - Temperature must be in range (0.0, 1.0]
            - Some OpenAI parameters (presence_penalty, frequency_penalty, logit_bias) are ignored
            - Image and audio type inputs are not currently supported
            - The 'n' parameter only supports value 1
        """

        # Resolve API key
        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY")
            if api_key is None:
                raise ValueError(
                    "MiniMax API key not provided and MINIMAX_API_KEY environment variable not set."
                )

        # Resolve base URL
        if base_url is None:
            base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")

        # Validate temperature range (0.0 < temperature <= 1.0)
        if temperature <= 0.0 or temperature > 1.0:
            raise ValueError(
                f"MiniMax temperature must be in range (0.0, 1.0], got {temperature}. "
                "Recommended value is 1.0."
            )

        # Initialize via OpenAI-compatible client
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized MiniMax client with model: {model_name}")
        self.logger.info(f"Using API endpoint: {base_url}")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """Handle chat completions via the MiniMax API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional arguments such as:
                - tools: List of tool definitions for function calling
                - tool_choice: Control which tool to call
                - stream: Whether to stream the response
                - response_format: Format for structured outputs

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage.

        Note:
            - Parameters like presence_penalty, frequency_penalty, logit_bias are ignored
            - Image and audio inputs are not currently supported
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # Build params consistent with OpenAI API
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Merge additional kwargs (e.g., tools, tool_choice, response_format, stream)
        params.update(kwargs)


        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during MiniMax API call: {e}")
            raise

        # Extract usage
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return [choice.message.content for choice in response.choices], usage


    def list_models(self) -> Dict[str, Any]:
        """List available models from the MiniMax API.

        Returns:
            Dict containing the models data with structure:
                {
                    "object": "list",
                    "data": [...]
                }
        """
        try:
            response = self.client.models.list()
            return {
                "object": "list",
                "data": [model.model_dump() for model in response.data]
            }
        except Exception as e:
            self.logger.error(f"Error listing models via MiniMax API: {e}")
            raise

