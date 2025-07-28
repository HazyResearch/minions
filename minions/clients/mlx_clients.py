import asyncio
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO, Literal

from minions.usage import Usage
from minions.clients.base import MinionsClient


class MLXParallmClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        temperature: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
        local: bool = True,
        **kwargs
    ):
        """
        Initialize the MLX PARALLM client.

        Args:
            model_name: The name or identifier of the MLX PARALLM model.
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 100)
            verbose: Whether to print verbose output (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.info(f"Loading MLX PARALLM model: {model_name}")

        # wrap mlx_parallm import in a try catch
        try:
            from mlx_parallm.utils import load, generate
            self.load = load
            self.generate = generate
        except ImportError:
            raise ImportError(
                "mlx_parallm is not installed. Please install it using 'pip install git+https://github.com/akhileshvb/mlx_parallm.git'."
            )

        self.model, self.tokenizer = self.load(model_name)
        self.logger.info(f"Model {model_name} loaded successfully")

    def chat(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, str]:
        """
        Generate a response for a chat conversation using the MLX PARALLM model.

        Args:
            messages: List of message dictionaries, each with 'role' and 'content' keys.
            **kwargs: Additional keyword arguments to pass to the generate function.

        Returns:
            A tuple containing:
              - List of response strings.
              - Usage object with prompt and completion token counts.
        """
        assert len(messages) > 0, "Messages cannot be empty."

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, temp=self.temperature
        )

        params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "verbose": self.verbose,
            "temp": self.temperature,
            **kwargs,
        }

        response = self.generate(**params)

        # messages are structured as list of lists in run.py
        # extract prompt tokens across all messages
        prompt_tokens = sum(len(prompt_i) for prompt_i in prompt)

        try:
            encoded = self.tokenizer.encode(response)
            completion_tokens = len(encoded)
        except Exception as e:
            self.logger.error(f"Error during token encoding: {e}")
            completion_tokens = len(response)

        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

        if self.local:
            return [response], usage, "END_OF_TEXT"
        else:
            return [response], usage


class MLXLMClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        verbose: bool = False,
        use_async: bool = False,
        enable_thinking: bool = False,
        local: bool = True,
        **kwargs
    ):
        """
        Initialize the MLX LM client.

        Args:
            model_name: The name or path of the model to use (default: "mlx-community/Llama-3.2-3B-Instruct")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 1000)
            verbose: Whether to print tokens and timing information (default: False)
            use_async: Whether to use async mode (default: False)
            enable_thinking: Whether to enable thinking mode (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.use_async = use_async
        self.enable_thinking = enable_thinking

        # Import and load the model
        try:
            from mlx_lm import generate, load
            self.generate = generate
            self.load = load
        except ImportError:
            raise ImportError(
                "mlx_lm is not installed. Please install it using 'pip install mlx_lm'."
            )

        # Load the model and tokenizer
        self.logger.info(f"Loading MLX LM model: {model_name}")
        self.model, self.tokenizer = self.load(path_or_hf_repo=model_name)
        self.logger.info(f"Model {model_name} loaded successfully")

    def _prepare_params(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare parameters for generation.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Dictionary of parameters for generation
        """
        # Apply the chat template to the messages
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            temp=self.temperature,
            enable_thinking=self.enable_thinking,
        )

        # Generate response params
        params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "verbose": self.verbose,
            **kwargs,
        }

        return params, prompt

    def chat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the MLX LM client.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)

    def schat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]

        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params, prompt = self._prepare_params(messages, **kwargs)
            response = self.generate(**params)

            # Since MLX LM doesn't provide token usage information directly,
            # we'll estimate it based on the input and output lengths
            prompt_tokens = len(prompt)
            completion_tokens = len(self.tokenizer.encode(response))

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            if self.local:
                return [response], usage, ["stop"]
            else:
                return [response], usage

        except Exception as e:
            self.logger.error(f"Error during MLX LM generation: {e}")
            raise

    def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )

        # Check if we're already in an event loop
        try:
            print("Checking if we're already in an event loop")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                print("We're in a running event loop")
                # We're in a running event loop (e.g., in Streamlit)
                # Create a new loop in a separate thread to avoid conflicts
                import threading
                import concurrent.futures

                # Use a thread to run our async code
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_in_new_loop, messages, **kwargs)
                    return future.result()
            else:
                # We have a loop but it's not running
                return loop.run_until_complete(self._achat_internal(messages, **kwargs))
        except RuntimeError:
            # No event loop exists, create one (the normal case)
            try:
                return asyncio.run(self._achat_internal(messages, **kwargs))
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Create a new event loop and set it as the current one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._achat_internal(messages, **kwargs)
                        )
                    finally:
                        loop.close()
                raise

    def _run_in_new_loop(self, messages, **kwargs):
        """Run the async chat in a new event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._achat_internal(messages, **kwargs))
        finally:
            loop.close()

    async def _achat_internal(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle async chat with multiple messages in parallel.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]

        assert len(messages) > 0, "Messages cannot be empty."

        async def process_one(msg):
            # We need to run the generation in a thread pool since MLX LM's generate
            # function is synchronous
            params, prompt = self._prepare_params([msg], **kwargs)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.generate(**params))
            print(response)

            prompt_tokens = len(prompt)
            completion_tokens = len(self.tokenizer.encode(response))

            return response, prompt_tokens, completion_tokens

        # Run tasks in parallel
        tasks = [process_one(m) for m in messages]
        results = await asyncio.gather(*tasks)

        # Gather results
        texts = []
        usage_total = Usage()
        done_reasons = []

        for response, prompt_tokens, completion_tokens in results:
            texts.append(response)
            usage_total += Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            done_reasons.append("stop")

        if self.local:
            return texts, usage_total, done_reasons
        else:
            return texts, usage_total


class MLXOmniClient(MinionsClient):
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
        local: bool = True,
        **kwargs
    ):
        """
        Initialize the MLX Omni client.

        Args:
            model_name: The name of the MLX model to use
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            use_test_client: Whether to use TestClient (True) or HTTP client (False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.use_test_client = use_test_client

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

            if self.local:
                return texts, usage, done_reasons
            else:
                return texts, usage

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


class MLXAudioClient(MinionsClient):
    """
    Client for interacting with the mlx-audio library for text-to-speech generation.

    This client provides an interface to the mlx-audio library, which is a text-to-speech
    implementation using the MLX framework.

    GitHub: https://github.com/Blaizzy/mlx-audio
    """

    def __init__(
        self,
        model_name: str = "prince-canuma/Kokoro-82M",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: Optional[str] = "a",
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the MLX Audio client.

        Args:
            model_name: The name of the model to use (default: "prince-canuma/Kokoro-82M")
            voice: The voice to use (default: "af_heart")
            speed: Speech speed multiplier (default: 1.0)
            lang_code: Language code (default: "a" for Kokoro's af_heart voice)
            verbose: Whether to print verbose output (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            verbose=verbose,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Check if mlx-audio is installed
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            import mlx_audio

            self.logger.info("MLX Audio package found")
        except ImportError as e:
            self.logger.error(f"Failed to import mlx-audio: {e}")
            raise ImportError(
                "mlx-audio package is required. Install with 'pip install mlx-audio'"
            )

    def load_audio(self, audio_path: str, sample_rate: int = 24000):
        """
        Load an audio file from a file path.
        """
        try:
            import mlx.core as mx
            from mlx_audio.tts.generate import load_audio

            return load_audio(audio_path, sample_rate)
        except Exception as e:
            self.logger.error(f"Error loading audio file: {e}")
            raise

    def text_to_speech(
        self,
        text: str,
        output_file: Optional[Union[str, BinaryIO]] = None,
        return_type: Literal["bytes", "file"] = "bytes",
        sample_rate: int = 24000,
        audio_format: str = "wav",
        join_audio: bool = True,
    ) -> Union[bytes, str, None]:
        """
        Convert text to speech using MLX Audio.

        Args:
            text: The text to convert to speech
            output_file: Optional file path or file-like object to save the audio to
            return_type: Type of return value:
                - "bytes": Return the audio as bytes (default)
                - "file": Save to output_file and return the path
                - "play": Play the audio and return None
            sample_rate: Sample rate of the output audio (default: 24000)
            audio_format: Format of the output audio (default: "wav")
            join_audio: Whether to join audio chunks (default: True)

        Returns:
            Union[bytes, str, None]: Audio data as bytes, file path, or None if played
        """
        try:
            from mlx_audio.tts.generate import generate_audio

            # Create a temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                file_prefix = os.path.join(temp_dir, "audio_output")

                # Generate the audio
                self.logger.info(f"Generating speech for text: {text[:50]}...")
                _ = generate_audio(
                    text=text,
                    model_path=self.model_name,
                    voice=self.voice,
                    speed=self.speed,
                    lang_code=self.lang_code,
                    file_prefix=file_prefix,
                    audio_format=audio_format,
                    sample_rate=sample_rate,
                    join_audio=join_audio,
                    verbose=self.verbose,
                )

                generated_file = f"{file_prefix}.{audio_format}"

                if return_type == "file" or output_file:
                    # If output_file is a string, copy the file there
                    if isinstance(output_file, str):
                        import shutil

                        shutil.copy2(generated_file, output_file)
                        return output_file

                    # If output_file is a file object, write to it
                    elif output_file is not None:
                        with open(generated_file, "rb") as f:
                            output_file.write(f.read())
                        return (
                            output_file.name if hasattr(output_file, "name") else None
                        )

                    # Otherwise return the temporary file path
                    else:
                        # Create a more permanent file outside the temp directory
                        permanent_file = os.path.join(
                            tempfile.gettempdir(),
                            f"mlx_audio_{os.path.basename(generated_file)}",
                        )
                        import shutil

                        shutil.copy2(generated_file, permanent_file)
                        return permanent_file

                # Default: return audio as bytes
                with open(generated_file, "rb") as f:
                    return f.read()

        except Exception as e:
            self.logger.error(f"Error during MLX Audio text-to-speech: {e}")
            raise 