import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from minions.usage import Usage


class MLXOmniClient:
    """
    Client for interacting with MLX Omni Server using TestClient method.
    This allows direct interaction with the application without starting a server.

    Read more details here: https://github.com/madroidmaq/mlx-omni-server
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        use_test_client: bool = True,
    ):
        """
        Initialize the MLX Omni client.

        Args:
            model_name: The name of the MLX model to use
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            use_test_client: Whether to use TestClient (True) or HTTP client (False)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_test_client = use_test_client

        self.logger = logging.getLogger("MLXOmniClient")
        self.logger.setLevel(logging.INFO)

        # Initialize the client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on configuration."""
        try:
            import openai

            if self.use_test_client:
                # Import TestClient and app for direct interaction
                try:
                    from fastapi.testclient import TestClient
                    from mlx_omni_server.main import app

                    self.logger.info(
                        "Using TestClient for direct interaction with MLX Omni Server"
                    )
                    self.client = openai.OpenAI(http_client=TestClient(app))
                except ImportError as e:
                    self.logger.error(
                        f"Failed to import TestClient or MLX Omni Server: {e}"
                    )
                    self.logger.warning("Falling back to HTTP client")
                    self._initialize_http_client()
            else:
                self._initialize_http_client()

        except ImportError as e:
            self.logger.error(f"Failed to import OpenAI: {e}")
            raise ImportError(
                "OpenAI package is required for MLXOmniClient. Install with 'pip install openai'"
            )

    def _initialize_http_client(self):
        """Initialize HTTP client for MLX Omni Server."""
        import openai

        base_url = os.getenv("MLX_OMNI_BASE_URL", "http://localhost:10240/v1")
        self.logger.info(f"Using HTTP client for MLX Omni Server at {base_url}")

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="not-needed",  # API key is not required for local server
        )

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available MLX models from the server.

        Returns:
            List[str]: List of model names
        """
        try:
            import openai

            # Create a temporary client to get models
            base_url = os.getenv("MLX_OMNI_BASE_URL", "http://localhost:10240/v1")
            client = openai.OpenAI(base_url=base_url, api_key="not-needed")

            # Get models
            models = client.models.list()
            return [model.id for model in models.data]

        except Exception as e:
            logging.error(f"Failed to get MLX Omni model list: {e}")
            return []

    def chat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the MLX Omni Server.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the API

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]

        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Prepare parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }

            # Call the API
            response = self.client.chat.completions.create(**params)

            # Extract the content from the response
            texts = [choice.message.content for choice in response.choices]

            # Extract usage information
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

            # Extract finish reasons
            done_reasons = [choice.finish_reason for choice in response.choices]

            return texts, usage, done_reasons

        except Exception as e:
            self.logger.error(f"Error during MLX Omni API call: {e}")
            raise

    def generate_image(
        self, prompt: str, n: int = 1, size: str = "512x512", **kwargs
    ) -> List[str]:
        """
        Generate images using the MLX Omni Server.

        Args:
            prompt: The prompt to generate images from
            n: Number of images to generate
            size: Size of the images to generate
            **kwargs: Additional arguments to pass to the API

        Returns:
            List[str]: List of image URLs or base64-encoded images
        """
        try:
            # Prepare parameters
            params = {
                "model": kwargs.pop("model", "argmaxinc/mlx-FLUX.1-schnell"),
                "prompt": prompt,
                "n": n,
                "size": size,
                **kwargs,
            }

            # Call the API
            response = self.client.images.generate(**params)

            # Extract the URLs or base64 data
            if hasattr(response.data[0], "url"):
                return [image.url for image in response.data]
            else:
                return [image.b64_json for image in response.data]

        except Exception as e:
            self.logger.error(f"Error during MLX Omni image generation: {e}")
            raise

    def text_to_speech(
        self, text: str, model: str = "lucasnewman/f5-tts-mlx", **kwargs
    ) -> bytes:
        """
        Convert text to speech using the MLX Omni Server.

        Args:
            text: The text to convert to speech
            model: The TTS model to use
            **kwargs: Additional arguments to pass to the API

        Returns:
            bytes: Audio data
        """
        try:
            # Call the API
            response = self.client.audio.speech.create(
                model=model, input=text, **kwargs
            )

            # Return the audio data
            return response.content

        except Exception as e:
            self.logger.error(f"Error during MLX Omni text-to-speech: {e}")
            raise

    def speech_to_text(
        self, audio_file, model: str = "mlx-community/whisper-large-v3-turbo", **kwargs
    ) -> str:
        """
        Convert speech to text using the MLX Omni Server.

        Args:
            audio_file: File-like object containing audio data
            model: The STT model to use
            **kwargs: Additional arguments to pass to the API

        Returns:
            str: Transcribed text
        """
        try:
            # Call the API
            transcript = self.client.audio.transcriptions.create(
                model=model, file=audio_file, **kwargs
            )

            # Return the transcribed text
            return transcript.text

        except Exception as e:
            self.logger.error(f"Error during MLX Omni speech-to-text: {e}")
            raise
