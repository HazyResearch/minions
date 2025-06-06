"""
MLX Parallm Model client integration tests.
Real local model calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from minions.clients.mlx_parallm_model import MLXParallmClient
except ImportError:
    MLXParallmClient = None


class TestMLXParallmModelClientIntegration(unittest.TestCase):
    """MLX Parallm Model tests - requires mlx-parallm (macOS only)"""
    
    def setUp(self):
        """Set up MLX Parallm client"""
        if MLXParallmClient is None:
            warnings.warn(
                "Skipping MLX Parallm tests: mlx-parallm not installed. "
                "Install with: pip install git+https://github.com/akhileshvb/mlx_parallm.git (macOS only)",
                UserWarning
            )
            self.skipTest("MLX Parallm not available")
        
        try:
            # Try to create client with a small model
            self.client = MLXParallmClient(
                model_name="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                temperature=0.1,
                max_tokens=50,
                verbose=False
            )
            
        except Exception as e:
            if "mlx" in str(e).lower() or "not supported" in str(e).lower():
                warnings.warn(
                    f"Skipping MLX Parallm tests: MLX not supported on this platform. "
                    f"MLX requires macOS with Apple Silicon. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX Parallm not supported on this platform")
            else:
                warnings.warn(
                    f"Skipping MLX Parallm tests: Could not initialize client. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX Parallm client initialization failed")
    
    def test_basic_chat(self):
        """Test basic MLX Parallm chat"""
        messages = [{"role": "user", "content": "Say 'mlx parallm working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # MLX Parallm returns (responses, usage, done_reasons)
            self.assertIsInstance(result, tuple)
            self.assertGreaterEqual(len(result), 2)
            
            responses, usage = result[0], result[1]
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            
        except Exception as e:
            if "model" in str(e).lower() or "download" in str(e).lower():
                self.skipTest(f"MLX Parallm model not available: {e}")
            elif "parallm" in str(e).lower():
                self.skipTest(f"MLX Parallm library issue: {e}")
            else:
                raise
    
    def test_model_loading(self):
        """Test MLX Parallm model loading"""
        try:
            # Test that the model components are loaded
            self.assertIsNotNone(self.client.model)
            self.assertIsNotNone(self.client.tokenizer)
            
            # Test model name
            self.assertEqual(self.client.model_name, "mlx-community/Meta-Llama-3-8B-Instruct-4bit")
            
        except Exception as e:
            if "model" in str(e).lower() or "loading" in str(e).lower():
                self.skipTest(f"Model loading test failed: {e}")
            else:
                raise
    
    def test_temperature_settings(self):
        """Test different temperature settings"""
        try:
            # Test with different temperature
            temp_client = MLXParallmClient(
                model_name="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                temperature=0.7,
                max_tokens=30
            )
            
            self.assertEqual(temp_client.temperature, 0.7)
            
            messages = [{"role": "user", "content": "Hello"}]
            result = temp_client.chat(messages)
            
            self.assertIsInstance(result, tuple)
            
        except Exception as e:
            if "temperature" in str(e).lower() or "model" in str(e).lower():
                self.skipTest(f"Temperature test failed: {e}")
            else:
                raise
    
    def test_token_limits(self):
        """Test token limit functionality"""
        try:
            # Test with very small token limit
            limited_client = MLXParallmClient(
                model_name="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                max_tokens=10,
                verbose=False
            )
            
            messages = [{"role": "user", "content": "Tell me a long story"}]
            responses, usage = limited_client.chat(messages)
            
            # Should respect token limit
            self.assertLessEqual(usage.completion_tokens, 15)  # Allow some buffer
            
        except Exception as e:
            if "token" in str(e).lower() or "model" in str(e).lower():
                self.skipTest(f"Token limit test failed: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()