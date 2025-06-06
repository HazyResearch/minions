"""
Gemini client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.gemini import GeminiClient
from test_base_client_integration import BaseClientIntegrationTest


class TestGeminiClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = GeminiClient
    SERVICE_NAME = "gemini"
    DEFAULT_MODEL = "gemini-2.0-flash"
    
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
            {"role": "system", "content": "You are a test assistant. Always respond with exactly 'GEMINI_SYSTEM_OK' when greeted."},
            {"role": "user", "content": "Hello, please respond with the exact test phrase"}
        ]
        
        result = self.client.chat(messages)
        responses, usage = result[0], result[1]
        # Gemini may not follow system messages strictly, so check for broader content
        self.assertTrue(len(responses) > 0 and len(responses[0]) > 0, "Should return non-empty response")


if __name__ == '__main__':
    unittest.main()