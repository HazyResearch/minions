"""
Azure OpenAI client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.azure_openai import AzureOpenAIClient
from test_base_client_integration import BaseClientIntegrationTest


class TestAzureOpenAIClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = AzureOpenAIClient
    SERVICE_NAME = "azure"
    DEFAULT_MODEL = "gpt-4o"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_azure_specific_configuration(self):
        """Test Azure-specific configuration"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'AZURE_CONFIG_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "AZURE_CONFIG_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'AZURE_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "AZURE_SYSTEM_OK")
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation"""
        messages = [
            {"role": "user", "content": "Remember the word 'azure'"},
            {"role": "assistant", "content": "I'll remember the word 'azure'."},
            {"role": "user", "content": "What word did I ask you to remember?"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "azure")


if __name__ == '__main__':
    unittest.main()