"""
Groq client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.groq import GroqClient
from test_base_client_integration import BaseClientIntegrationTest


class TestGroqClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = GroqClient
    SERVICE_NAME = "groq"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        # Groq returns only (responses, usage) tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        responses, usage = result
        self.assertIsInstance(responses, list)
        self.assertGreater(len(responses), 0)
        self.assertIsInstance(responses[0], str)
        self.assert_response_content(responses, "test successful")
    
    def test_fast_inference(self):
        """Test Groq's fast inference capability"""
        import time
        
        messages = [{"role": "user", "content": "Say 'speed test' quickly"}]
        
        start_time = time.time()
        responses, usage = self.client.chat(messages)
        end_time = time.time()
        
        # Groq should be very fast - under 5 seconds for simple requests
        self.assertLess(end_time - start_time, 5.0)
        self.assert_response_content(responses, "speed test")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'GROQ_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "GROQ_SYSTEM_OK")


if __name__ == '__main__':
    unittest.main()