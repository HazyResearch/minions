"""Lemonade client integration tests."""

import unittest
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
    DEFAULT_MODEL = "Qwen2.5-0.5B-Instruct-CPU"

    def setUp(self):
        """Set up Lemonade client if server URL provided."""
        base_url = os.getenv("LEMONADE_BASE_URL")
        if not base_url:
            self.skipTest("LEMONADE_BASE_URL not set; skipping Lemonade tests")
        self.client = self.CLIENT_CLASS(
            model_name=self.DEFAULT_MODEL,
            base_url=base_url,
            temperature=0.1,
            max_tokens=50,
        )

    def test_basic_chat(self):
        """Test basic chat functionality."""
        messages = self.get_test_messages()
        result = self.client.chat(messages)

        self.assert_valid_chat_response(result)
        responses, _ = result
        self.assert_response_content(responses, "test successful")


if __name__ == "__main__":
    unittest.main()
