"""
Transformers client integration tests.
Real local model calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os
import logging

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Also suppress transformers logging during tests
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from minions.clients.transformers import TransformersClient
except ImportError:
    TransformersClient = None


class TestTransformersClientIntegration(unittest.TestCase):
    """Transformers tests - requires transformers and torch"""
    
    def setUp(self):
        """Set up Transformers client"""
        if TransformersClient is None:
            warnings.warn(
                "Skipping Transformers tests: transformers not installed. "
                "Install with: pip install transformers torch",
                UserWarning
            )
            self.skipTest("Transformers not available")
        
        try:
            # Try to create client with a small model that has chat template support
            self.client = TransformersClient(
                model_name="dleemiller/Penny-1.7B", # This is the only model I could find that worked with this setup
                temperature=0.1,
                max_tokens=50,
                do_sample=True
            )
            
        except Exception as e:
            warnings.warn(
                f"Skipping Transformers tests: Could not initialize client. "
                f"Make sure transformers and torch are installed: pip install transformers torch. "
                f"Error: {e}",
                UserWarning
            )
            self.skipTest("Transformers client initialization failed")
    
    def test_basic_chat(self):
        """Test basic Transformers chat"""
        messages = [{"role": "user", "content": "Hello"}]
        
        try:
            result = self.client.chat(messages)
            
            # Transformers returns (responses, usage)
            self.assertIsInstance(result, tuple)
            self.assertGreaterEqual(len(result), 2)
            
            responses, usage = result[0], result[1]
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            
        except Exception as e:
            if ("model" in str(e).lower() or "download" in str(e).lower() or
                "chat_template" in str(e).lower() or "template" in str(e).lower()):
                self.skipTest(f"Model not available or lacks chat template: {e}")
            elif "cuda" in str(e).lower() or "device" in str(e).lower():
                self.skipTest(f"GPU/device issue: {e}")
            else:
                raise
    
    def test_embedding_support(self):
        """Test Transformers embedding functionality"""
        try:
            # Create embedding client with a sentence transformer model
            embedding_client = TransformersClient(
                model_name="dleemiller/Penny-1.7B",
                embedding_model="dleemiller/Penny-1.7B"
            )
            
            embeddings = embedding_client.embed("Test embedding text")
            
            self.assertIsInstance(embeddings, list)
            self.assertGreater(len(embeddings[0]), 0)
            self.assertTrue(all(isinstance(x, float) for x in embeddings[0]))
            
        except Exception as e:
            if "embedding" in str(e).lower() or "sentence-transformers" in str(e).lower():
                self.skipTest(f"Embedding model not available: {e}")
            elif "download" in str(e).lower():
                self.skipTest(f"Could not download embedding model: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()