"""
Cohere client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.cohere import CohereClient
from test_base_client_integration import BaseClientIntegrationTest


class TestCohereClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = CohereClient
    SERVICE_NAME = "cohere"
    DEFAULT_MODEL = "command-a-03-2025"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")

    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'COHERE_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "COHERE_SYSTEM_OK")

    def test_embeddings(self):
        """Test embeddings endpoint via compatibility API"""
        embeddings = self.client.embed(["hello world", "goodbye world"]) 
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], list)
        self.assertGreater(len(embeddings[0]), 0)

    def test_list_models(self):
        """Test listing models endpoint"""
        models = self.client.list_models()
        self.assertIsInstance(models, dict)
        self.assertEqual(models.get("object"), "list")
        self.assertIn("data", models)


if __name__ == '__main__':
    unittest.main()


