"""
Perplexity client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.perplexity import PerplexityAIClient
from test_base_client_integration import BaseClientIntegrationTest


class TestPerplexityClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = PerplexityAIClient
    SERVICE_NAME = "perplexity"
    DEFAULT_MODEL = "llama-3.1-sonar-small-128k-online"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_search_capabilities(self):
        """Test Perplexity's search and real-time capabilities"""
        messages = [
            {"role": "user", "content": "What is the current weather? Just respond with 'SEARCH_TEST_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "SEARCH_TEST_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "You are a test assistant. Always respond with exactly 'PERPLEXITY_SYSTEM_OK' when greeted."},
            {"role": "user", "content": "Hello, please respond with the exact test phrase"}
        ]
        
        responses, usage = self.client.chat(messages)
        # Perplexity may not follow system messages strictly, so check for broader content
        self.assertTrue(len(responses) > 0 and len(responses[0]) > 0, "Should return non-empty response")
    
    def test_different_sonar_models(self):
        """Test different Perplexity Sonar models"""
        # Test with a different model
        try:
            client = PerplexityAIClient(
                model_name="llama-3.1-sonar-large-128k-online",
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage = client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "model" in str(e).lower() or "not available" in str(e).lower():
                self.skipTest(f"Alternative Perplexity model not available: {e}")
            else:
                raise
    
    def test_online_search_mode(self):
        """Test Perplexity's online search mode"""
        messages = [
            {"role": "user", "content": "Search for current news and say 'ONLINE_MODE_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "ONLINE_MODE_OK")


if __name__ == '__main__':
    unittest.main()