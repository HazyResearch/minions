"""Lemonade client integration tests."""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.lemonade import LemonadeClient
from test_base_client_integration import BaseClientIntegrationTest


class TestLemonadeClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = LemonadeClient
    SERVICE_NAME = "lemonade"
    DEFAULT_MODEL = "Llama-3.2-3B-Instruct-Hybrid"

    @classmethod
    def setUpClass(cls):
        """Override to skip API key checking since Lemonade is a local service."""
        # Skip the parent's setUpClass which checks for API keys
        # Lemonade runs locally and doesn't require API keys
        pass

    def setUp(self):
        """Set up Lemonade client if server URL provided."""
        base_url = "http://localhost:8000/api/v1"
        try:
            self.client = self.CLIENT_CLASS(
                model_name=self.DEFAULT_MODEL,
                base_url=base_url,
                temperature=0.1,
                max_tokens=50,
            )
        except Exception as e:
            warnings.warn(
                f"Skipping Lemonade tests: Lemonade not available locally. "
                f"Install and start Lemonade to run these tests. Error: {e}",
                UserWarning
            )
            self.skipTest("Lemonade not available")

    def test_basic_chat(self):
        """Test basic chat functionality."""
        messages = self.get_test_messages()
        result = self.client.chat(messages)

        self.assert_valid_chat_response(result)
        responses, _, _ = result
        self.assert_response_content(responses, "test successful")

if __name__ == "__main__":
    unittest.main()