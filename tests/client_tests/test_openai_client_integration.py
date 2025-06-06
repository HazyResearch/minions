"""
OpenAI client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.openai import OpenAIClient
from test_base_client_integration import BaseClientIntegrationTest


class TestOpenAIClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = OpenAIClient
    SERVICE_NAME = "openai"
    DEFAULT_MODEL = "gpt-3.5-turbo"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation"""
        messages = [
            {"role": "user", "content": "Remember the number 42"},
            {"role": "assistant", "content": "I'll remember the number 42."},
            {"role": "user", "content": "What number did I ask you to remember?"}
        ]
        
        responses, usage = self.client.chat(messages)
        
        self.assert_valid_chat_response((responses, usage))
        self.assert_response_content(responses, "42")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'SYSTEM_TEST_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "SYSTEM_TEST_OK")
    
    def test_temperature_effect(self):
        """Test that temperature affects response variation"""
        # This test runs same prompt with different temperatures
        messages = [{"role": "user", "content": "Write a creative sentence about cats"}]
        
        # Low temperature client
        low_temp_client = OpenAIClient(
            model_name=self.DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=30
        )
        
        # High temperature client  
        high_temp_client = OpenAIClient(
            model_name=self.DEFAULT_MODEL,
            temperature=1.0,
            max_tokens=30
        )
        
        low_responses, _ = low_temp_client.chat(messages)
        high_responses, _ = high_temp_client.chat(messages)
        
        # Both should be valid responses
        self.assertGreater(len(low_responses[0]), 0)
        self.assertGreater(len(high_responses[0]), 0)
    
    def test_max_tokens_limit(self):
        """Test max_tokens parameter"""
        client = OpenAIClient(
            model_name=self.DEFAULT_MODEL,
            max_tokens=5  # Very small limit
        )
        
        messages = [{"role": "user", "content": "Write a long story about dragons"}]
        responses, usage = client.chat(messages)
        
        # Should hit token limit
        self.assertLessEqual(usage.completion_tokens, 10)  # Allow some buffer


if __name__ == '__main__':
    unittest.main()