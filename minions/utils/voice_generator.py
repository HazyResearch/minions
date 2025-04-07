import os
import tempfile
import base64
import time
from pathlib import Path
from typing import List, Optional
import streamlit as st
import numpy as np

try:
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    # raise error error message
    raise ImportError(
        "mlx_lm is not installed. Please install it with: pip install mlx-lm"
    )

try:
    from csm_mlx import CSM, csm_1b, generate
    from huggingface_hub import hf_hub_download
    import torch
    import torchaudio
except ImportError:
    # raise error error message
    raise ImportError(
        "csm_mlx is not installed. Please install it with: pip install git+https://github.com/senstella/csm-mlx"
    )


class VoiceGenerator:
    """Utility class for generating voice audio from text using CSM-MLX."""

    def __init__(self):
        self.csm_available = False
        try:
            # Try to import CSM-MLX
            from csm_mlx import CSM

            self.csm_available = True
            self.csm_mlx = VoiceGeneratorMLX()
            print("CSM-MLX voice generation initialized successfully")
        except ImportError:
            print("CSM-MLX not available. Voice generation will be disabled.")

    def generate_audio(self, text, voice=0):
        """
        Generate audio from text using CSM-MLX.

        Args:
            text (str): The text to convert to speech
            voice (str): Voice style to use (default, happy, sad, etc.)

        Returns:
            str: Base64 encoded audio data or None if generation failed
        """
        if not self.csm_available:
            return None

        try:

            file_path = self.csm_mlx.generate_audio(
                text=text, speaker=voice, max_audio_length_ms=30000
            )

            # Read the audio file and encode as base64
            with open(file_path, "rb") as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            # Clean up the temporary file
            os.unlink(file_path)
            return audio_base64

        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None

    @staticmethod
    def get_audio_html(audio_base64):
        """
        Create HTML for audio playback from base64 encoded audio.

        Args:
            audio_base64 (str): Base64 encoded audio data

        Returns:
            str: HTML for audio playback
        """
        if not audio_base64:
            return ""

        audio_html = f"""
        <audio controls autoplay="false">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html


class VoiceGeneratorMLX:
    """Voice generator using CSM-1B model."""

    def __init__(self):
        self.generator = None
        self.is_initialized = False
        self.temp_dir = tempfile.mkdtemp()
        self.csm = None

    def initialize(self):
        """Initialize the voice generator model."""
        if not self.is_initialized:
            try:
                from huggingface_hub import hf_hub_download
                from csm_mlx import CSM, csm_1b, generate

                csm = CSM(csm_1b())  # csm_1b() is a configuration for the CSM model.
                weight = hf_hub_download(
                    repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors"
                )
                csm.load_weights(weight)
                self.csm = csm
            except ImportError as e:
                st.error(f"Failed to import required modules for voice generation: {e}")
                st.info(
                    "Please install the required packages with: pip install csm-mlx"
                )
                return False
            except Exception as e:
                st.error(f"Failed to initialize voice generator: {e}")
                return False
        return True

    def generate_audio(
        self,
        text: str,
        speaker: int = 0,
        context: Optional[List] = None,
        max_audio_length_ms: int = 30000,
    ) -> str:
        """Generate audio from text and return the path to the audio file."""
        if not self.is_initialized:
            if not self.initialize():
                return None

        if context is None:
            context = []

        try:
            import torchaudio

            # Generate audio
            audio = generate(
                self.csm,
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=max_audio_length_ms,
                sampler=make_sampler(temp=0.8, min_p=0.05),
            )

            # Save audio to a temporary file
            file_path = os.path.join(self.temp_dir, f"audio_{hash(text)}.wav")
            torchaudio.save(
                file_path, torch.Tensor(np.asarray(audio)).unsqueeze(0).cpu(), 24_000
            )

            return file_path
        except Exception as e:
            st.error(f"Error generating audio: {e}")
            return None
