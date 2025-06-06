"""
Mistral client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.mistral import MistralClient
from test_base_client_integration import BaseClientIntegrationTest


class TestMistralClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = MistralClient
    SERVICE_NAME = "mistral"
    DEFAULT_MODEL = "mistral-small-latest"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage, finish_reasons = result
        self.assert_response_content(responses, "test successful")
        
        # Mistral-specific assertions
        self.assertIsInstance(finish_reasons, list)
        self.assertIn(finish_reasons[0], ['stop', 'length'])
    
    def test_embedding_support(self):
        """Test Mistral embedding functionality"""
        try:
            embeddings = self.client.embed("Test embedding text")
            self.assertIsInstance(embeddings, list)
            self.assertGreater(len(embeddings[0]), 0)
            
        except NotImplementedError:
            self.skipTest("Embedding not supported by Mistral client")
        except Exception as e:
            if "not supported" in str(e).lower():
                self.skipTest(f"Embedding not supported: {e}")
            else:
                raise
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'MISTRAL_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage, finish_reasons = self.client.chat(messages)
        self.assert_response_content(responses, "MISTRAL_SYSTEM_OK")


if __name__ == '__main__':
    unittest.main()