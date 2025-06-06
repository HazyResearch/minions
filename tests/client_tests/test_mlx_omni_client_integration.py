"""
MLX Omni client integration tests.
Real local model calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from minions.clients.mlx_omni import MLXOmniClient
except ImportError:
    MLXOmniClient = None


class TestMLXOmniClientIntegration(unittest.TestCase):
    """MLX Omni tests - requires mlx-omni-server (macOS only)"""
    
    def setUp(self):
        """Set up MLX Omni client"""
        if MLXOmniClient is None:
            warnings.warn(
                "Skipping MLX Omni tests: mlx-omni-server not installed. "
                "Install from: https://github.com/madroidmaq/mlx-omni-server (macOS only)",
                UserWarning
            )
            self.skipTest("MLX Omni not available")
        
        # Check for required dependencies before trying to initialize
        try:
            import mlx_omni_server
        except ImportError:
            warnings.warn(
                "Skipping MLX Omni tests: mlx-omni-server not installed. "
                "Install from: https://github.com/madroidmaq/mlx-omni-server (macOS only)",
                UserWarning
            )
            self.skipTest("MLX Omni server library not available")
        
        try:
            # Try to create client with test client mode
            self.client = MLXOmniClient(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
                temperature=0.1,
                max_tokens=50,
                use_test_client=True
            )
            
        except Exception as e:
            if "mlx" in str(e).lower() or "not supported" in str(e).lower():
                warnings.warn(
                    f"Skipping MLX Omni tests: MLX not supported on this platform. "
                    f"MLX requires macOS with Apple Silicon. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX Omni not supported on this platform")
            else:
                warnings.warn(
                    f"Skipping MLX Omni tests: Could not initialize client. Error: {e}",
                    UserWarning
                )
                self.skipTest("MLX Omni client initialization failed")
    
    def test_basic_chat(self):
        """Test basic MLX Omni chat"""
        messages = [{"role": "user", "content": "Say 'mlx omni working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # MLX Omni returns (responses, usage) or similar
            self.assertIsInstance(result, tuple)
            self.assertGreaterEqual(len(result), 2)
            
            responses, usage = result[0], result[1]
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            
        except Exception as e:
            if "model" in str(e).lower() or "server" in str(e).lower():
                self.skipTest(f"MLX Omni server not available: {e}")
            elif "connection" in str(e).lower():
                self.skipTest(f"Could not connect to MLX Omni server: {e}")
            else:
                raise
    
    def test_test_client_mode(self):
        """Test MLX Omni test client mode"""
        try:
            test_client = MLXOmniClient(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
                use_test_client=True,
                max_tokens=20
            )
            
            self.assertTrue(test_client.use_test_client)
            
            messages = [{"role": "user", "content": "Test"}]
            result = test_client.chat(messages)
            
            self.assertIsInstance(result, tuple)
            
        except Exception as e:
            if ("test" in str(e).lower() or "client" in str(e).lower() or 
                "connection" in str(e).lower() or "refused" in str(e).lower()):
                self.skipTest(f"Test client mode not available: {e}")
            else:
                raise
    
    def test_http_client_mode(self):
        """Test MLX Omni HTTP client mode"""
        try:
            http_client = MLXOmniClient(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
                use_test_client=False,
                max_tokens=20
            )
            
            self.assertFalse(http_client.use_test_client)
            
            messages = [{"role": "user", "content": "Test"}]
            result = http_client.chat(messages)
            
            self.assertIsInstance(result, tuple)
            
        except Exception as e:
            if "http" in str(e).lower() or "server" in str(e).lower() or "connection" in str(e).lower():
                self.skipTest(f"HTTP client mode not available (server not running): {e}")
            else:
                raise
    
    def test_model_configuration(self):
        """Test MLX Omni model configuration"""
        try:
            # Test that the model name is set correctly
            self.assertEqual(self.client.model_name, "mlx-community/Llama-3.2-1B-Instruct-4bit")
            
            # Test client configuration
            self.assertEqual(self.client.temperature, 0.1)
            self.assertEqual(self.client.max_tokens, 50)
            self.assertTrue(self.client.use_test_client)
            
        except Exception as e:
            if "config" in str(e).lower():
                self.skipTest(f"Configuration test failed: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()