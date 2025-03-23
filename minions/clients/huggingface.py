import logging
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator
import os
from huggingface_hub import InferenceClient, AsyncInferenceClient

from minions.usage import Usage
from minions.clients.utils import ServerMixin


class HuggingFaceClient:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        api_token: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace client.

        Args:
            model_name: The name of the model to use (default: "meta-llama/Llama-3.2-3B-Instruct")
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            api_token: HuggingFace API token (optional, falls back to HF_TOKEN environment variable)
        """
        self.model_name = model_name
        self.logger = logging.getLogger("HuggingFaceClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API token from parameter or environment variable
        self.api_token = api_token or os.getenv("HF_TOKEN")

        self.client = InferenceClient(model=self.model_name, token=self.api_token)
        self.async_client = AsyncInferenceClient(
            model=self.model_name, token=self.api_token
        )

    def chat(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, str]:
        """
        Handle chat completions using the HuggingFace Inference API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat_completion method

        Returns:
            Tuple of (List[str], Usage, str) containing response strings, token usage, and model info
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Set default parameters if not provided in kwargs
            if "temperature" not in kwargs:
                kwargs["temperature"] = self.temperature

            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat_completion(
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            self.logger.error(f"Error during HuggingFace API call: {e}")
            raise

        # HuggingFace doesn't provide token usage information in the same way as OpenAI
        # We'll create a placeholder Usage object
        usage = Usage(
            prompt_tokens=0,  # Not provided by HuggingFace API
            completion_tokens=0,  # Not provided by HuggingFace API
        )

        # Extract the content from the response
        content = response.choices[0].message.content

        return [content], usage, self.model_name

    async def achat(
        self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs
    ) -> Tuple[List[str], Usage, str] | AsyncIterator[str]:
        """
        Asynchronously handle chat completions using the HuggingFace Inference API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            stream: Whether to stream the response (default: False)
            **kwargs: Additional arguments to pass to the chat_completion method

        Returns:
            If stream=False: Tuple of (List[str], Usage, str) containing response strings, token usage, and model info
            If stream=True: AsyncIterator yielding response chunks as they become available
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # Set default parameters if not provided in kwargs
        if "temperature" not in kwargs:
            kwargs["temperature"] = self.temperature

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens

        if stream:
            try:
                stream_response = await self.async_client.chat_completion(
                    messages=messages,
                    stream=True,
                    **kwargs,
                )

                async def response_generator():
                    async for chunk in stream_response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return response_generator()
            except Exception as e:
                self.logger.error(
                    f"Error during async streaming HuggingFace API call: {e}"
                )
                raise
        else:
            try:
                response = await self.async_client.chat_completion(
                    messages=messages,
                    **kwargs,
                )

                # HuggingFace doesn't provide token usage information in the same way as OpenAI
                usage = Usage(
                    prompt_tokens=0,  # Not provided by HuggingFace API
                    completion_tokens=0,  # Not provided by HuggingFace API
                )

                # Extract the content from the response
                content = response.choices[0].message.content

                return [content], usage, self.model_name
            except Exception as e:
                self.logger.error(f"Error during async HuggingFace API call: {e}")
                raise

    # TODO: extend to other huggingface client types:  https://huggingface.co/docs/huggingface_hub/en/guides/inference
