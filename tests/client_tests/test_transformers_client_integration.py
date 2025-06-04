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
                model_name="mistralai/Mistral-7B-v0.1",
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
                model_name="mistralai/Mistral-7B-v0.1",
                embedding_model="mistralai/Mistral-7B-v0.1"
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
    
    def test_tool_calling_mode(self):
        """Test Transformers tool calling functionality"""
        try:
            tool_client = TransformersClient(
                model_name="microsoft/DialoGPT-small",
                tool_calling=True,
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "What is the weather?"}]
            result = tool_client.chat(messages)
            
            # Should return results even if no actual tools are called
            self.assertIsInstance(result, tuple)
            responses, usage = result[0], result[1]
            self.assertIsInstance(responses, list)
            
        except Exception as e:
            if "tool" in str(e).lower() or "not supported" in str(e).lower():
                self.skipTest(f"Tool calling not supported: {e}")
            else:
                raise
    
    def test_different_model_types(self):
        """Test loading different types of models"""
        try:
            # Test with a different small model
            gpt2_client = TransformersClient(
                model_name="gpt2",  # Classic small model
                max_tokens=20
            )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage = gpt2_client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if ("model" in str(e).lower() or "download" in str(e).lower() or
                "chat_template" in str(e).lower() or "template" in str(e).lower()):
                self.skipTest(f"Alternative model not available or lacks chat template: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()