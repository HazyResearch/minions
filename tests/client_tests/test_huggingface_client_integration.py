"""
Hugging Face client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from minions.clients.huggingface import HuggingFaceClient
except ImportError as e:
    HuggingFaceClient = None
    import_error = str(e)

from test_base_client_integration import BaseClientIntegrationTest


class TestHuggingFaceClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = HuggingFaceClient
    SERVICE_NAME = "huggingface"
    DEFAULT_MODEL = "microsoft/DialoGPT-medium"
    
    def setUp(self):
        """Set up HuggingFace client"""
        if HuggingFaceClient is None:
            warnings.warn(
                f"Skipping Hugging Face tests: Required dependencies not installed. "
                f"Install with: pip install soundfile huggingface-hub numpy. Error: {import_error}",
                UserWarning
            )
            self.skipTest("HuggingFace dependencies not available")
        
        # Call parent setup
        super().setUp()
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_different_models(self):
        """Test with different Hugging Face models"""
        # Test with a different model
        client = HuggingFaceClient(
            model_name="facebook/blenderbot-400M-distill",
            max_tokens=30
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        try:
            responses, usage = client.chat(messages)
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
        except Exception as e:
            if "model" in str(e).lower() or "not available" in str(e).lower():
                self.skipTest(f"Alternative model not available: {e}")
            else:
                raise
    
    def test_inference_api_features(self):
        """Test Hugging Face Inference API specific features"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'HF_INFERENCE_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "HF_INFERENCE_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'HF_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "HF_SYSTEM_OK")


if __name__ == '__main__':
    unittest.main()