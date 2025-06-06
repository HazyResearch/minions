"""
Tokasaurus client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.tokasaurus import TokasaurusClient
from test_base_client_integration import BaseClientIntegrationTest


class TestTokasaurusClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = TokasaurusClient
    SERVICE_NAME = "tokasaurus"
    DEFAULT_MODEL = "tokasaurus-default"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_tokasaurus_features(self):
        """Test Tokasaurus specific features"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'TOKASAURUS_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "TOKASAURUS_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'TOKASAURUS_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "TOKASAURUS_SYSTEM_OK")
    
    def test_token_optimization(self):
        """Test Tokasaurus token optimization features"""
        # Test with a longer prompt to see token handling
        messages = [
            {"role": "user", "content": "This is a longer message to test token optimization. Please respond with 'TOKEN_OPT_OK'."}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "TOKEN_OPT_OK")
        
        # Verify usage tracking
        self.assertGreater(usage.total_tokens, 0)
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.completion_tokens, 0)
    
    def test_model_configuration(self):
        """Test Tokasaurus model configuration"""
        # Test client configuration
        self.assertEqual(self.client.model_name, "tokasaurus-default")
        self.assertEqual(self.client.temperature, 0.1)
        self.assertEqual(self.client.max_tokens, 50)


if __name__ == '__main__':
    unittest.main()