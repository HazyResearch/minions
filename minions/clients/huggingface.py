import logging
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator, Union
import os
import base64
import tempfile
from pathlib import Path
import requests
import io
import soundfile as sf
import numpy as np
from huggingface_hub import InferenceClient, AsyncInferenceClient
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient
from minions.clients.utils import ServerMixin


class HuggingFaceClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        api_token: Optional[str] = None,
        use_router: bool = False,
        **kwargs
    ):
        """
        Initialize the HuggingFace client.

        Args:
            model_name: The name of the model to use (default: "meta-llama/Llama-3.2-3B-Instruct")
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            api_token: HuggingFace API token (optional, falls back to HF_TOKEN environment variable)
            use_router: Whether to use HuggingFace Router instead of Inference API (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)

        # Get API token from parameter or environment variable
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.use_router = use_router

        if self.use_router:
            # Initialize OpenAI client for HuggingFace Router
            self.router_client = openai.OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.api_token,
            )
            self.logger.info(f"Using HuggingFace Router for model: {model_name}")
        else:
            # Initialize regular HuggingFace clients
            self.client = InferenceClient(model=self.model_name, token=self.api_token)
            self.async_client = AsyncInferenceClient(
                model=self.model_name, token=self.api_token
            )

            if model_name.startswith("Qwen/Qwen2.5-Omni"):
                try:
                    # Import required libraries for multimodal processing
                    from transformers import Qwen2_5OmniModel

                    self.logger.info(f"Loading Qwen2.5-Omni model: {model_name}")

                    # Load model and processor
                    self.client = Qwen2_5OmniModel.from_pretrained(
                        self.model_name,
                        torch_dtype="auto",
                        device_map="auto",
                        attn_implementation="flash_attention_2",
                    )

                    self.logger.info(
                        f"Successfully loaded Qwen2.5-Omni model: {model_name}"
                    )
                except ImportError:
                    self.logger.warning(
                        "Required packages for Qwen2.5-Omni not installed. "
                        "Please install with: pip install transformers qwen-omni-utils[decord]"
                    )
                except Exception as e:
                    self.logger.error(f"Error loading Qwen2.5-Omni model: {e}")

    def chat(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, str]:
        """
        Handle chat completions using the HuggingFace Inference API or Router.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat_completion method

        Returns:
            Tuple of (List[str], Usage, str) containing response strings, token usage, and model info
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # if model is Qwen2.5-Omni, use multimodal_chat
        if self.model_name.startswith("Qwen/Qwen2.5-Omni"):
            return self.multimodal_chat(messages, **kwargs)

        if self.use_router:
            # Use HuggingFace Router (OpenAI-compatible API)
            try:
                # Set default parameters if not provided in kwargs
                if "temperature" not in kwargs:
                    kwargs["temperature"] = self.temperature

                if "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = self.max_tokens

                response = self.router_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )

                # Extract usage information if available
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                )

                # Extract the content from the response
                content = response.choices[0].message.content

                return [content], usage, self.model_name

            except Exception as e:
                self.logger.error(f"Error during HuggingFace Router API call: {e}")
                raise
        else:
            # Use regular HuggingFace Inference API
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
        Asynchronously handle chat completions using the HuggingFace Inference API or Router.

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

        if self.use_router:
            # Use HuggingFace Router (OpenAI-compatible API)
            if stream:
                try:
                    stream_response = await self.router_client.chat.completions.create(
                        model=self.model_name,
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
                        f"Error during async streaming HuggingFace Router API call: {e}"
                    )
                    raise
            else:
                try:
                    response = await self.router_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **kwargs,
                    )

                    # Extract usage information if available
                    usage = Usage(
                        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                        completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    )

                    # Extract the content from the response
                    content = response.choices[0].message.content

                    return [content], usage, self.model_name
                except Exception as e:
                    self.logger.error(f"Error during async HuggingFace Router API call: {e}")
                    raise
        else:
            # Use regular HuggingFace Inference API
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

    def _process_media_file(self, file_path_or_url: str) -> Tuple[str, bytes]:
        """
        Process a media file (image, audio, video) from a file path or URL.

        Args:
            file_path_or_url: Path to local file or URL

        Returns:
            Tuple of (file_type, file_bytes)
        """
        # Determine if input is a URL or local file path
        if file_path_or_url.startswith(("http://", "https://")):
            # Download from URL
            response = requests.get(file_path_or_url, stream=True)
            response.raise_for_status()
            file_bytes = response.content

            # Determine file type from URL extension
            file_extension = Path(file_path_or_url).suffix.lower()
        else:
            # Read local file
            file_path = Path(file_path_or_url)
            file_bytes = file_path.read_bytes()
            file_extension = file_path.suffix.lower()

        # Map file extension to media type
        if file_extension in [".jpg", ".jpeg", ".png", ".webp"]:
            file_type = "image"
        elif file_extension in [".mp3", ".wav", ".ogg", ".flac"]:
            file_type = "audio"
        elif file_extension in [".mp4", ".avi", ".mov", ".webm"]:
            file_type = "video"
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        return file_type, file_bytes

    def _format_multimodal_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a message with multimodal content for the Qwen2.5-Omni model.

        Args:
            message: Message dictionary with 'role' and 'content'

        Returns:
            Formatted message dictionary
        """
        role = message["role"]
        content = message["content"]

        # If content is already in the correct format, return as is
        if isinstance(content, list) and all(
            isinstance(item, dict) for item in content
        ):
            return message

        # If content is a string, convert to the expected format
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}

        # If content is a dict with media information
        if isinstance(content, dict) and "media" in content and "text" in content:
            media_items = []

            # Process each media item
            for media_path in content["media"]:
                media_type, media_bytes = self._process_media_file(media_path)
                media_b64 = base64.b64encode(media_bytes).decode("utf-8")
                media_items.append({"type": media_type, media_type: media_b64})

            # Add text if present
            if content["text"]:
                media_items.append({"type": "text", "text": content["text"]})

            return {"role": role, "content": media_items}

        # If content format is not recognized
        raise ValueError(f"Unsupported message content format: {type(content)}")

    def multimodal_chat(
        self,
        messages: List[Dict[str, Any]],
        return_audio: bool = False,
        voice_type: str = "Chelsie",
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Handle multimodal chat completions using the Qwen2.5-Omni model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                Content can be:
                - A string (text only)
                - A dict with 'media' (list of file paths or URLs) and 'text' keys
                - A list of dicts with 'type' and corresponding media keys
            return_audio: Whether to return audio output (default: False)
            voice_type: Voice type for audio output ("Chelsie" or "Ethan", default: "Chelsie")
            use_audio_in_video: Whether to use audio from video inputs (default: True)
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dictionary with 'text' key and optional 'audio' key (if return_audio=True)
        """
        if not self.model_name.startswith("Qwen/Qwen2.5-Omni"):
            raise ValueError(
                f"multimodal_chat is only supported for Qwen2.5-Omni models, got {self.model_name}"
            )

        assert len(messages) > 0, "Messages cannot be empty."

        # Ensure system message is present with correct content for audio output
        if return_audio:
            has_correct_system = False
            for msg in messages:
                if msg["role"] == "system" and isinstance(msg["content"], str):
                    if (
                        "capable of perceiving auditory and visual inputs, as well as generating text and speech"
                        in msg["content"]
                    ):
                        has_correct_system = True
                        break

            if not has_correct_system:
                # Add the required system message at the beginning
                system_msg = {
                    "role": "system",
                    "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
                messages = [system_msg] + messages

        # Format messages for multimodal input
        formatted_messages = [self._format_multimodal_message(msg) for msg in messages]

        try:
            # Import required libraries for multimodal processing
            try:
                from transformers import Qwen2_5OmniProcessor
                from qwen_omni_utils import process_mm_info
            except ImportError:
                self.logger.error(
                    "Required packages not installed. Please install with: "
                    "pip install transformers qwen-omni-utils[decord]"
                )
                raise

            processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)

            # Prepare inputs
            text = processor.apply_chat_template(
                formatted_messages, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                formatted_messages, use_audio_in_video=use_audio_in_video
            )
            inputs = processor(
                text=text,
                audios=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.client.device).to(self.client.dtype)

            # get len of inputs
            len_inputs = len(inputs)

            usage = Usage(
                prompt_tokens=len_inputs,  # Not provided by HuggingFace API
                completion_tokens=0,  # Not provided by HuggingFace API
            )

            # Generate response
            if return_audio:
                text_ids, audio = self.client.generate(
                    **inputs,
                    use_audio_in_video=use_audio_in_video,
                    spk=voice_type,
                    **kwargs,
                )

                # Decode text
                text_output = processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Process audio
                audio_array = audio.reshape(-1).detach().cpu().numpy()

                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    sf.write(temp_file.name, audio_array, samplerate=24000)
                    audio_path = temp_file.name

                usage.completion_tokens = len(audio_array) + len(text_ids)

                # TODO: add audio to response
                return [text_output], usage, "STOP"
            else:
                text_ids = self.client.generate(
                    **inputs,
                    use_audio_in_video=use_audio_in_video,
                    return_audio=False,
                    **kwargs,
                )

                # Decode text
                text_output = processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                usage.completion_tokens = len(text_ids)

                # TODO: add audio to response
                return [text_output], usage, "STOP"

        except Exception as e:
            self.logger.error(f"Error during multimodal chat: {e}")
            raise

    async def amultimodal_chat(
        self,
        messages: List[Dict[str, Any]],
        return_audio: bool = False,
        voice_type: str = "Chelsie",
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Asynchronously handle multimodal chat completions using the Qwen2.5-Omni model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                Content can be:
                - A string (text only)
                - A dict with 'media' (list of file paths or URLs) and 'text' keys
                - A list of dicts with 'type' and corresponding media keys
            return_audio: Whether to return audio output (default: False)
            voice_type: Voice type for audio output ("Chelsie" or "Ethan", default: "Chelsie")
            use_audio_in_video: Whether to use audio from video inputs (default: True)
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dictionary with 'text' key and optional 'audio' key (if return_audio=True)
        """
        # For now, we'll use the synchronous implementation with asyncio.to_thread in the future
        # This is a placeholder that calls the synchronous version
        import asyncio

        return await asyncio.to_thread(
            self.multimodal_chat,
            messages=messages,
            return_audio=return_audio,
            voice_type=voice_type,
            use_audio_in_video=use_audio_in_video,
            **kwargs,
        )

    # TODO: extend to other huggingface client types:  https://huggingface.co/docs/huggingface_hub/en/guides/inference
