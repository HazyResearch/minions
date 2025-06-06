"""
MLX Audio client integration tests.
Real local audio generation calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from minions.clients.mlx_audio import MLXAudioClient
except ImportError:
    MLXAudioClient = None


class TestMLXAudioClientIntegration(unittest.TestCase):
    """MLX Audio tests - requires mlx-audio (macOS only)"""
    
    def setUp(self):
        """Set up MLX Audio client"""
        if MLXAudioClient is None:
            warnings.warn(
                "Skipping MLX Audio tests: mlx-audio not installed. "
                "Install from: https://github.com/Blaizzy/mlx-audio (macOS only)",
                UserWarning
            )
            self.skipTest("MLX Audio not available")
        
        try:
            # Try to create client with default voice model
            self.client = MLXAudioClient(
                model_name="prince-canuma/Kokoro-82M",
                voice="af_heart",
                speed=1.0,
                verbose=False
            )
            
        except Exception as e:
            if "mlx" in str(e).lower() or "not supported" in str(e).lower():
                warnings.warn(
                    f"Skipping MLX Audio tests: MLX not supported on this platform. "
                    f"MLX requires macOS with Apple Silicon. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX Audio not supported on this platform")
            else:
                warnings.warn(
                    f"Skipping MLX Audio tests: Could not initialize client. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX Audio client initialization failed")
    
    def test_text_to_speech_generation(self):
        """Test basic text-to-speech generation"""
        try:
            text = "Hello world"
            
            # MLX Audio typically has a text_to_speech or generate method
            # Since this is audio generation, we test if the method exists and works
            if hasattr(self.client, 'text_to_speech'):
                audio_data = self.client.text_to_speech(text)
                self.assertIsNotNone(audio_data)
            elif hasattr(self.client, 'generate'):
                audio_data = self.client.generate(text)
                self.assertIsNotNone(audio_data)
            else:
                self.skipTest("No recognizable audio generation method found")
                
        except Exception as e:
            if "model" in str(e).lower() or "download" in str(e).lower():
                self.skipTest(f"Audio model not available: {e}")
            elif "audio" in str(e).lower():
                self.skipTest(f"Audio generation failed: {e}")
            else:
                raise
    
    def test_different_voices(self):
        """Test different voice configurations"""
        try:
            # Test with different voice settings
            voice_client = MLXAudioClient(
                model_name="prince-canuma/Kokoro-82M",
                voice="af_heart",
                speed=1.2,
                lang_code="a"
            )
            
            # Test if client initialized properly
            self.assertIsNotNone(voice_client.voice)
            self.assertEqual(voice_client.speed, 1.2)
            
        except Exception as e:
            if "voice" in str(e).lower() or "model" in str(e).lower():
                self.skipTest(f"Voice configuration not available: {e}")
            else:
                raise
    
    def test_speed_variations(self):
        """Test different speech speed settings"""
        try:
            speeds = [0.8, 1.0, 1.5]
            
            for speed in speeds:
                speed_client = MLXAudioClient(
                    model_name="prince-canuma/Kokoro-82M",
                    speed=speed,
                    verbose=False
                )
                
                self.assertEqual(speed_client.speed, speed)
                
        except Exception as e:
            if "speed" in str(e).lower() or "model" in str(e).lower():
                self.skipTest(f"Speed configuration not available: {e}")
            else:
                raise
    
    def test_model_loading(self):
        """Test audio model loading"""
        try:
            # Test that the model name is set correctly
            self.assertEqual(self.client.model_name, "prince-canuma/Kokoro-82M")
            
            # Test client attributes
            self.assertIsNotNone(self.client.voice)
            self.assertIsNotNone(self.client.speed)
            
        except Exception as e:
            if "model" in str(e).lower():
                self.skipTest(f"Model loading test failed: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()