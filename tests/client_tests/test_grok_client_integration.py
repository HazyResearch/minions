"""
Grok client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.grok import GrokClient
from test_base_client_integration import BaseClientIntegrationTest


class TestGrokClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = GrokClient
    SERVICE_NAME = "grok"
    DEFAULT_MODEL = "grok-beta"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_grok_humor(self):
        """Test Grok's humorous responses"""
        messages = [
            {"role": "user", "content": "Tell me a short joke about AI"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assertIsInstance(responses, list)
        self.assertGreater(len(responses), 0)
        self.assertIsInstance(responses[0], str)
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'GROK_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "GROK_SYSTEM_OK")
    
    def test_real_time_information(self):
        """Test Grok's real-time information capabilities"""
        messages = [
            {"role": "user", "content": "What is today's date? Just say 'DATE_TEST_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "DATE_TEST_OK")


if __name__ == '__main__':
    unittest.main()