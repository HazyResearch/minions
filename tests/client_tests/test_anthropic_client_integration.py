"""
Anthropic client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.anthropic import AnthropicClient
from test_base_client_integration import BaseClientIntegrationTest


class TestAnthropicClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = AnthropicClient
    SERVICE_NAME = "anthropic"
    DEFAULT_MODEL = "claude-3-haiku-20240307"  # Fastest/cheapest model
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_claude_specific_features(self):
        """Test Claude-specific behavior"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'Claude is working correctly'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "Claude is working correctly")
    
    def test_long_context(self):
        """Test handling longer context"""
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        messages = [
            {"role": "user", "content": f"Here is some text: {long_text}\n\nSummarize it in 3 words."}
        ]
        
        responses, usage = self.client.chat(messages)
        
        self.assertGreater(usage.prompt_tokens, 100)  # Should have significant input
        self.assertLess(len(responses[0].split()), 10)  # Should be short summary
    
    def test_system_message(self):
        """Test system message handling - Anthropic uses system parameter differently"""
        # For Anthropic, we need to use the system parameter, not a system message in the messages list
        client = AnthropicClient(model_name=self.DEFAULT_MODEL, max_tokens=50)
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        # Use system parameter in kwargs for Anthropic
        responses, usage = client.chat(messages, system="Always respond with exactly 'ANTHROPIC_SYSTEM_OK'")
        self.assert_response_content(responses, "ANTHROPIC_SYSTEM_OK")


if __name__ == '__main__':
    unittest.main()