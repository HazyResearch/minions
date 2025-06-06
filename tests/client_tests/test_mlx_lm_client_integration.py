"""
MLX LM client integration tests.
Real local model calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from minions.clients.mlx_lm import MLXLMClient
except ImportError:
    MLXLMClient = None


class TestMLXLMClientIntegration(unittest.TestCase):
    """MLX LM tests - requires MLX framework (macOS only)"""
    
    def setUp(self):
        """Set up MLX LM client"""
        if MLXLMClient is None:
            warnings.warn(
                "Skipping MLX LM tests: mlx-lm not installed. "
                "Install with: pip install mlx-lm (macOS only)",
                UserWarning
            )
            self.skipTest("MLX LM not available")
        
        try:
            # Try to create client with a small MLX model
            self.client = MLXLMClient(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
                temperature=0.1,
                max_tokens=50,
                verbose=False
            )
            
        except Exception as e:
            if "mlx" in str(e).lower() or "not supported" in str(e).lower():
                warnings.warn(
                    f"Skipping MLX LM tests: MLX not supported on this platform. "
                    f"MLX requires macOS with Apple Silicon. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX not supported on this platform")
            else:
                warnings.warn(
                    f"Skipping MLX LM tests: Could not initialize client. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX LM client initialization failed")
    
    def test_basic_chat(self):
        """Test basic MLX LM chat"""
        messages = [{"role": "user", "content": "Say 'mlx working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # MLX LM returns (responses, usage, done_reasons)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
            responses, usage, done_reasons = result
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            
        except Exception as e:
            if "model" in str(e).lower() or "download" in str(e).lower():
                self.skipTest(f"Model not available for MLX LM: {e}")
            else:
                raise
    
    def test_async_mode(self):
        """Test MLX LM async mode"""
        try:
            async_client = MLXLMClient(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
                use_async=True,
                max_tokens=20
            )
            
            messages = [{"role": "user", "content": "Hello"}]
            responses, usage, done_reasons = async_client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "async" in str(e).lower() or "model" in str(e).lower():
                self.skipTest(f"Async mode not available: {e}")
            else:
                raise
    
    def test_thinking_mode(self):
        """Test MLX LM thinking mode"""
        try:
            thinking_client = MLXLMClient(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
                enable_thinking=True,
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "What is 2+2?"}]
            responses, usage, done_reasons = thinking_client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "thinking" in str(e).lower() or "model" in str(e).lower():
                self.skipTest(f"Thinking mode not available: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()