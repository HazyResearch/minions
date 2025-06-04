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
        try:
            # Try to create client with a small model from HF
            self.client = LlamaCppClient(
                model_path="",
                model_repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
                model_file_pattern="*q8_0.gguf",
                temperature=0.1,
                max_tokens=50,
                n_gpu_layers=0,  # Use CPU only for compatibility
                verbose=False
            )
            
        except Exception as e:
            warnings.warn(
                f"Skipping LlamaCpp tests: Could not initialize client. "
                f"Make sure llama-cpp-python is installed: pip install llama-cpp-python. "
                f"Error: {e}",
                UserWarning
            )
            self.skipTest("LlamaCpp not available")
    
    def test_basic_chat(self):
        """Test basic LlamaCpp chat"""
        messages = [{"role": "user", "content": "Say 'llamacpp working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # LlamaCpp returns (responses, usage, done_reasons)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
            responses, usage, done_reasons = result
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            self.assertIn("llamacpp", responses[0].lower())
            
        except Exception as e:
            if "not found" in str(e).lower() or "download" in str(e).lower():
                self.skipTest(f"Model not available for LlamaCpp: {e}")
            else:
                raise
    
    def test_embedding_generation(self):
        """Test LlamaCpp embedding generation if enabled"""
        try:
            # Create embedding-enabled client
            embedding_client = LlamaCppClient(
                model_path="",
                model_repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF", 
                model_file_pattern="*q8_0.gguf",
                embedding=True,
                n_gpu_layers=0
            )
            
            embeddings = embedding_client.embed("Test embedding text")
            self.assertIsInstance(embeddings, list)
            self.assertGreater(len(embeddings[0]), 0)
            self.assertTrue(all(isinstance(x, float) for x in embeddings[0]))
            
        except Exception as e:
            if "embedding" in str(e).lower() or "not supported" in str(e).lower():
                self.skipTest(f"Embedding not supported: {e}")
            elif "not found" in str(e).lower():
                self.skipTest(f"Model not available: {e}")
            else:
                raise
    
    def test_model_loading_from_hf(self):
        """Test model loading from Hugging Face Hub"""
        try:
            # Test creating another client instance to verify model loading works
            test_client = LlamaCppClient(
                model_path="",
                model_repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
                model_file_pattern="*q8_0.gguf",
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