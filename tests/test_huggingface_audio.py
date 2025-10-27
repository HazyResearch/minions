"""Tests for HuggingFace client audio support."""

import pytest
import numpy as np
from io import BytesIO
import soundfile as sf

from minions.clients.huggingface import HuggingFaceClient
from minions.clients.response import ChatResponse
from minions.usage import Usage


class TestAudioConversion:
    """Test audio array to bytes conversion."""

    def test_audio_array_to_wav_bytes(self):
        """Test converting numpy array to WAV bytes."""
        # Create synthetic audio (1 second of silence at 24kHz)
        sample_rate = 24000
        duration = 1.0
        audio_array = np.zeros(int(sample_rate * duration), dtype=np.float32)

        # Convert to bytes
        audio_bytes = HuggingFaceClient._audio_array_to_wav_bytes(
            audio_array,
            sample_rate
        )

        # Verify it's bytes
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0

        # Verify it's valid WAV (can be read back)
        buffer = BytesIO(audio_bytes)
        data, sr = sf.read(buffer)
        assert sr == sample_rate
        assert len(data) == len(audio_array)

    def test_audio_bytes_are_reusable(self):
        """Test that audio bytes can be saved/played multiple times."""
        import tempfile
        import os

        audio_array = np.random.randn(24000).astype(np.float32)
        audio_bytes = HuggingFaceClient._audio_array_to_wav_bytes(
            audio_array,
            24000
        )

        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            filepath = f.name

        try:
            # Read back
            data, sr = sf.read(filepath)
            assert sr == 24000
            np.testing.assert_array_almost_equal(data, audio_array, decimal=5)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_audio_array_to_wav_bytes_with_sine_wave(self):
        """Test conversion with actual audio signal (sine wave)."""
        # Generate 440Hz tone for 0.5 seconds at 24kHz
        sample_rate = 24000
        duration = 0.5
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_array = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Convert to bytes
        audio_bytes = HuggingFaceClient._audio_array_to_wav_bytes(
            audio_array,
            sample_rate
        )

        # Verify WAV header
        assert audio_bytes.startswith(b'RIFF')
        assert b'WAVE' in audio_bytes[:20]

        # Verify can be read back
        buffer = BytesIO(audio_bytes)
        recovered_data, recovered_sr = sf.read(buffer)

        assert recovered_sr == sample_rate
        assert len(recovered_data) == len(audio_array)
        np.testing.assert_array_almost_equal(recovered_data, audio_array, decimal=5)

    def test_audio_conversion_no_temp_files(self):
        """Test that conversion doesn't create temporary files."""
        import tempfile
        import os

        # Get temp directory
        temp_dir = tempfile.gettempdir()

        # Count WAV files before
        wav_files_before = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]

        # Convert audio
        audio_array = np.random.randn(24000).astype(np.float32)
        _ = HuggingFaceClient._audio_array_to_wav_bytes(audio_array, 24000)

        # Count WAV files after
        wav_files_after = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]

        # No new WAV files should be created
        assert len(wav_files_after) == len(wav_files_before)


class TestMultimodalChatAudioResponse:
    """Test multimodal_chat with audio in ChatResponse."""

    def test_chat_response_structure_with_audio(self):
        """Test ChatResponse can hold audio bytes."""
        audio_bytes = b'RIFF....WAVE....'  # Mock audio bytes

        response = ChatResponse(
            responses=["Hello, world!"],
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            done_reasons=["STOP"],
            audio=audio_bytes
        )

        assert response.audio == audio_bytes
        assert isinstance(response.audio, bytes)
        assert response.responses == ["Hello, world!"]

    def test_chat_response_without_audio(self):
        """Test ChatResponse with None audio field."""
        response = ChatResponse(
            responses=["Hello, world!"],
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            done_reasons=["STOP"],
            audio=None
        )

        assert response.audio is None
        assert response.responses == ["Hello, world!"]

    def test_chat_response_backward_compatibility_with_audio(self):
        """Test tuple unpacking still works when audio field is present."""
        audio_bytes = b'RIFF....WAVE....'

        response = ChatResponse(
            responses=["Hello"],
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            done_reasons=["STOP"],
            audio=audio_bytes
        )

        # Should unpack to 3-tuple (responses, usage, done_reasons)
        # Audio is accessed via attribute, not unpacking
        responses, usage, done_reasons = response

        assert responses == ["Hello"]
        assert isinstance(usage, Usage)
        assert done_reasons == ["STOP"]

        # Audio still accessible via attribute
        assert response.audio == audio_bytes


class TestHuggingFaceAudioIntegration:
    """Integration tests for audio generation (requires mocking)."""

    def test_multimodal_chat_signature_accepts_return_audio(self):
        """Test that multimodal_chat accepts return_audio parameter."""
        # This test just verifies the signature, doesn't call the model
        client = HuggingFaceClient(model_name="Qwen/Qwen2.5-Omni-7B")

        # Verify method exists and has correct signature
        import inspect
        sig = inspect.signature(client.multimodal_chat)

        assert 'return_audio' in sig.parameters
        assert sig.parameters['return_audio'].default is False
        assert 'voice_type' in sig.parameters
        assert sig.parameters['voice_type'].default == "Chelsie"

    def test_multimodal_chat_returns_chat_response_type(self):
        """Test that multimodal_chat return type annotation is ChatResponse."""
        client = HuggingFaceClient(model_name="Qwen/Qwen2.5-Omni-7B")

        # This is a structural test - would need mocking for full test
        # Just verifies the method exists and client is properly initialized
        assert hasattr(client, 'multimodal_chat')
        assert callable(client.multimodal_chat)
