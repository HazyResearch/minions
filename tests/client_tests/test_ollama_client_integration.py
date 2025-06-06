"""
Ollama client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from minions.clients.ollama import OllamaClient


class TestOllamaClientIntegration(unittest.TestCase):
    """Ollama tests - no API key needed but requires local Ollama"""
    
    def setUp(self):
        """Set up Ollama client"""
        try:
            # Test if Ollama is available
            self.client = OllamaClient(
                model_name="llama3.2:1b",  # Small model
                temperature=0.1,
                max_tokens=50
            )
            
            # Try to get available models to test connection
            available_models = self.client.get_available_models()
            
        except Exception as e:
            warnings.warn(
                f"Skipping Ollama tests: Ollama not available locally. "
                f"Install and start Ollama to run these tests. Error: {e}",
                UserWarning
            )
            self.skipTest("Ollama not available")
    
    def test_basic_chat(self):
        """Test basic Ollama chat"""
        messages = [{"role": "user", "content": "Say 'ollama working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # Ollama returns (responses, usage, done_reasons) or (responses, usage, done_reasons, tools)
            self.assertIsInstance(result, tuple)
            self.assertGreaterEqual(len(result), 3)
            
            responses, usage, done_reasons = result[0], result[1], result[2]
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIn("ollama working", responses[0].lower())
            
        except Exception as e:
            if "not found" in str(e).lower():
                self.skipTest(f"Model not available in Ollama: {e}")
            else:
                raise
    
    def test_embedding_generation(self):
        """Test Ollama embedding generation if supported"""
        try:
            embeddings = self.client.embed("Test embedding text")
            self.assertIsInstance(embeddings, list)
            self.assertGreater(len(embeddings[0]), 0)
            
        except NotImplementedError:
            self.skipTest("Embedding not supported by this Ollama model")
        except Exception as e:
            if "not found" in str(e).lower():
                self.skipTest(f"Embedding model not available: {e}")
            else:
                raise
    
    def test_model_availability(self):
        """Test model availability checking"""
        try:
            available_models = self.client.get_available_models()
            self.assertIsInstance(available_models, list)
            
        except Exception as e:
            self.skipTest(f"Could not check available models: {e}")


if __name__ == '__main__':
    unittest.main()