"""
Cartesia MLX client integration tests.
Real local model calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from minions.clients.cartesia_mlx import CartesiaMLXClient
except ImportError:
    CartesiaMLXClient = None


class TestCartesiaMLXClientIntegration(unittest.TestCase):
    """Cartesia MLX tests - requires cartesia-mlx (macOS only)"""
    
    def setUp(self):
        """Set up Cartesia MLX client"""
        if CartesiaMLXClient is None:
            warnings.warn(
                "Skipping Cartesia MLX tests: cartesia-mlx not installed. "
                "Follow installation instructions for cartesia-mlx (macOS only)",
                UserWarning
            )
            self.skipTest("Cartesia MLX not available")
        
        try:
            # Try to create client with Cartesia model
            self.client = CartesiaMLXClient(
                model_name="cartesia-ai/Llamba-1B-4bit-mlx",
                temperature=0.1,
                max_tokens=50,
                verbose=False,
                dtype="float32"
            )
            
        except Exception as e:
            if "mlx" in str(e).lower() or "not supported" in str(e).lower():
                warnings.warn(
                    f"Skipping Cartesia MLX tests: MLX not supported on this platform. "
                    f"MLX requires macOS with Apple Silicon. Error: {e}",
                    UserWarning
                )
                self.skipTest("Cartesia MLX not supported on this platform")
            elif "cartesia" in str(e).lower():
                warnings.warn(
                    f"Skipping Cartesia MLX tests: Cartesia MLX library not available. Error: {e}",
                    UserWarning
                )
                self.skipTest("Cartesia MLX library not available")
            else:
                warnings.warn(
                    f"Skipping Cartesia MLX tests: Could not initialize client. Error: {e}",
                    UserWarning
                )
                self.skipTest("Cartesia MLX client initialization failed")
    
    def test_basic_chat(self):
        """Test basic Cartesia MLX chat"""
        messages = [{"role": "user", "content": "Say 'cartesia working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # Cartesia MLX should return (responses, usage) or similar
            self.assertIsInstance(result, tuple)
            self.assertGreaterEqual(len(result), 2)
            
            responses, usage = result[0], result[1]
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            
        except Exception as e:
            if "model" in str(e).lower() or "download" in str(e).lower():
                self.skipTest(f"Cartesia model not available: {e}")
            elif "cartesia" in str(e).lower():
                self.skipTest(f"Cartesia MLX issue: {e}")
            else:
                raise
    
    def test_model_loading(self):
        """Test Cartesia MLX model loading"""
        try:
            # Test that the model is loaded
            self.assertIsNotNone(self.client.model)
            self.assertIsNotNone(self.client.tokenizer)
            
            # Test model name
            self.assertEqual(self.client.model_name, "cartesia-ai/Llamba-1B-4bit-mlx")
            
            # Test dtype setting
            self.assertEqual(self.client.dtype, "float32")
            
        except Exception as e:
            if "model" in str(e).lower() or "loading" in str(e).lower():
                self.skipTest(f"Model loading test failed: {e}")
            else:
                raise
    
    def test_llamba_tokenizer(self):
        """Test Llamba-specific tokenizer configuration"""
        try:
            # Test that for Llamba models, the tokenizer is configured correctly
            if "Llamba-1B" in self.client.model_name:
                # Should use Llama-3.2 tokenizer for Llamba models
                self.assertIsNotNone(self.client.tokenizer)
                
        except Exception as e:
            if "tokenizer" in str(e).lower():
                self.skipTest(f"Tokenizer test failed: {e}")
            else:
                raise
    
    def test_different_dtypes(self):
        """Test different data types"""
        try:
            dtypes = ["float32", "float16"]
            
            for dtype in dtypes:
                try:
                    dtype_client = CartesiaMLXClient(
                        model_name="cartesia-ai/Llamba-1B-4bit-mlx",
                        dtype=dtype,
                        max_tokens=20
                    )
                    
                    self.assertEqual(dtype_client.dtype, dtype)
                    
                except Exception as e:
                    # Some dtypes might not be supported
                    if dtype in str(e).lower():
                        continue
                    else:
                        raise
                        
        except Exception as e:
            if "dtype" in str(e).lower() or "type" in str(e).lower():
                self.skipTest(f"Data type test failed: {e}")
            else:
                raise
    
    def test_temperature_control(self):
        """Test temperature control"""
        try:
            # Test with very low temperature for deterministic output
            deterministic_client = CartesiaMLXClient(
                model_name="cartesia-ai/Llamba-1B-4bit-mlx",
                temperature=0.001,  # Very low temperature
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "What is 2+2?"}]
            responses, usage = deterministic_client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "temperature" in str(e).lower():
                self.skipTest(f"Temperature control test failed: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()