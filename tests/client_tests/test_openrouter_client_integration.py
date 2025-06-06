"""
OpenRouter client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.openrouter import OpenRouterClient
from test_base_client_integration import BaseClientIntegrationTest


class TestOpenRouterClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = OpenRouterClient
    SERVICE_NAME = "openrouter"
    DEFAULT_MODEL = "openai/gpt-4o"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        # Use a simpler, more reliable prompt
        messages = [{"role": "user", "content": "Say hello"}]
        result = self.client.chat(messages)
        self.assert_valid_chat_response(result)
        responses, usage = result
        
        # Check that we got a non-empty response
        self.assertGreater(len(responses[0].strip()), 0, "Response should not be empty")
        
        # For free models, just verify we get some text response
        self.assertIsInstance(responses[0], str)
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always be polite."},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        
        # Check that we got a response
        self.assertGreater(len(responses[0].strip()), 0, "Response should not be empty")
        
        # Should be a greeting response
        response_lower = responses[0].lower()
        self.assertTrue(
            any(word in response_lower for word in ["hello", "hi", "greet", "help"]),
            f"Expected greeting response, got: {responses[0]}"
        )
    
    def test_different_providers(self):
        """Test different model providers through OpenRouter"""
        # Test with a different free model
        try:
            client = OpenRouterClient(
                model_name="openai/gpt-4o",
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage = client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "model" in str(e).lower() or "not available" in str(e).lower():
                self.skipTest(f"Alternative OpenRouter model not available: {e}")
            else:
                raise
    
    def test_openrouter_routing(self):
        """Test OpenRouter's model routing capabilities"""
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        responses, usage = self.client.chat(messages)
        
        # Check that we got a response
        self.assertGreater(len(responses[0].strip()), 0, "Response should not be empty")
        
        # Should mention Paris
        response_lower = responses[0].lower()
        self.assertIn("paris", response_lower, f"Expected 'Paris' in response, got: {responses[0]}")


if __name__ == '__main__':
    unittest.main()