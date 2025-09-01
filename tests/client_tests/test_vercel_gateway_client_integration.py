"""
Vercel AI Gateway client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.vercel_gateway import VercelGatewayClient
from test_base_client_integration import BaseClientIntegrationTest


class TestVercelGatewayClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = VercelGatewayClient
    SERVICE_NAME = "vercel"
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def test_basic_chat(self):
        """Test basic chat via gateway"""
        messages = [{"role": "user", "content": "Say hello"}]
        responses, usage = self.client.chat(messages)
        self.assertGreater(len(responses[0].strip()), 0)
        self.assertIsInstance(responses[0], str)

    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always be polite."},
            {"role": "user", "content": "Hello"}
        ]
        responses, usage = self.client.chat(messages)
        self.assertGreater(len(responses[0].strip()), 0)

    def test_provider_options_override(self):
        """Ensure providerOptions can be passed and do not error"""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        provider_options = {
            # This structure mirrors Vercel's providerOptions field; values are examples only
            "openai": {"cache": False}
        }
        responses, usage = self.client.chat(messages, provider_options=provider_options)
        self.assertGreater(len(responses[0].strip()), 0)


if __name__ == '__main__':
    unittest.main()


