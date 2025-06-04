"""
Llama API client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.llama_api import LlamaApiClient
from test_base_client_integration import BaseClientIntegrationTest


class TestLlamaAPIClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = LlamaApiClient
    SERVICE_NAME = "llama_api"
    DEFAULT_MODEL = "llama3.1-8b"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_llama_specific_features(self):
        """Test Llama API specific features"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'LLAMA_API_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "LLAMA_API_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'LLAMA_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "LLAMA_SYSTEM_OK")
    
    def test_different_llama_models(self):
        """Test with different Llama models"""
        # Test with a different model if available
        try:
            client = LlamaApiClient(
                model_name="llama3.1-70b",
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage = client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "model" in str(e).lower() or "not available" in str(e).lower():
                self.skipTest(f"Alternative Llama model not available: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()