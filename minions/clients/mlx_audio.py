import logging
import os
import tempfile
from typing import Optional, Union, BinaryIO, Literal


class MLXAudioClient:
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
    ):
        """
        Initialize the MLX Audio client.

        Args:
            model_name: The name of the model to use (default: "prince-canuma/Kokoro-82M")
            voice: The voice to use (default: "af_heart")
            speed: Speech speed multiplier (default: 1.0)
            lang_code: Language code (default: "a" for Kokoro's af_heart voice)
            verbose: Whether to print verbose output (default: False)
        """
        self.model_name = model_name
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.verbose = verbose

        self.logger = logging.getLogger("MLXAudioClient")
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
