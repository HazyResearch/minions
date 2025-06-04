"""
LlamaCpp client integration tests.
Real local model calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from minions.clients.llamacpp import LlamaCppClient


class TestLlamaCppClientIntegration(unittest.TestCase):
    """LlamaCpp tests - requires local model files or Hugging Face download"""
    
    def setUp(self):
        """Set up LlamaCpp client"""
        print("Setting up LlamaCpp client...")
        try:
            # Try local model path first (from test_llama_cpp.py pattern)
            local_model_path = os.path.join(os.getcwd(), "models", "qwen2-0_5b-instruct-q8_0.gguf")
            print(f"Looking for local model at: {local_model_path}")
            
            if os.path.exists(local_model_path):
                print("Found local model, using local path...")
                self.client = LlamaCppClient(
                    model_path=local_model_path,
                    chat_format="chatml",
                    temperature=0.1,
                    max_tokens=50,
                    n_gpu_layers=0,  # Use CPU only for compatibility
                    verbose=False
                )
            else:
                print("Local model not found, will download from HF...")
                # Fallback to HF download if local model doesn't exist
                self.client = LlamaCppClient(
                    model_repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
                    model_file_pattern="*q8_0.gguf",
                    chat_format="chatml",
                    temperature=0.1,
                    max_tokens=50,
                    n_gpu_layers=0,  # Use CPU only for compatibility
                    verbose=False
                )
            print("LlamaCpp client created successfully!")
            
        except Exception as e:
            print(f"Failed to create LlamaCpp client: {e}")
            warnings.warn(
                f"Skipping LlamaCpp tests: Could not initialize client. "
                f"Make sure llama-cpp-python is installed: pip install llama-cpp-python. "
                f"Error: {e}",
                UserWarning
            )
            self.skipTest("LlamaCpp not available")
    
    def test_basic_chat(self):
        """Test basic LlamaCpp chat"""
        print("Starting basic chat test...")
        messages = [{"role": "user", "content": "Hello"}]
        
        try:
            print("Calling client.chat...")
            result = self.client.chat(messages)
            print(result)
            print(f"Chat result type: {type(result)}")
            print(f"Chat result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            # LlamaCpp returns (responses, usage, done_reasons)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
            responses, usage, done_reasons = result
            print(f"Responses: {responses}")
            print(f"Usage: {usage}")
            print(f"Done reasons: {done_reasons}")
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            # Just verify we got a reasonable response (non-empty string with some content)
            self.assertGreater(len(responses[0].strip()), 0)
            print("Basic chat test passed!")
            
        except Exception as e:
            print(f"Chat test failed with error: {e}")
            if "not found" in str(e).lower() or "download" in str(e).lower():
                self.skipTest(f"Model not available for LlamaCpp: {e}")
            else:
                raise
    
    def test_model_loading_from_hf(self):
        """Test model loading from local path or Hugging Face Hub"""
        print("Starting model loading test...")
        try:
            # Test creating another client instance using same logic
            local_model_path = os.path.join(os.getcwd(), "models", "qwen2-0.5b-instruct-q8_0.gguf")
            
            if os.path.exists(local_model_path):
                print("Testing with local model...")
                test_client = LlamaCppClient(
                    model_path=local_model_path,
                    chat_format="chatml",
                    n_gpu_layers=0,
                    max_tokens=10
                )
            else:
                print("Testing with HF model...")
                test_client = LlamaCppClient(
                    model_repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
                    model_file_pattern="*q8_0.gguf",
                    chat_format="chatml",
                    n_gpu_layers=0,
                    max_tokens=10
                )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage, done_reasons = test_client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "download" in str(e).lower() or "not found" in str(e).lower():
                self.skipTest(f"Could not download model: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()