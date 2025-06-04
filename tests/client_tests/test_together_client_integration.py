"""
Together client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.together import TogetherClient
from test_base_client_integration import BaseClientIntegrationTest


class TestTogetherClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = TogetherClient
    SERVICE_NAME = "together"
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result[0], result[1]
        self.assert_response_content(responses, "test successful")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'TOGETHER_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        result = self.client.chat(messages)
        responses, usage = result[0], result[1]
        self.assert_response_content(responses, "TOGETHER_SYSTEM_OK")


if __name__ == '__main__':
    unittest.main()