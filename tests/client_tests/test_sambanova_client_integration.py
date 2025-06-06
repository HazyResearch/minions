"""
SambaNova client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.sambanova import SambanovaClient
from test_base_client_integration import BaseClientIntegrationTest


class TestSambaNovaClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = SambanovaClient
    SERVICE_NAME = "sambanova"
    DEFAULT_MODEL = "Meta-Llama-3.1-8B-Instruct"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_sambanova_optimization(self):
        """Test SambaNova's optimization features"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'SAMBANOVA_OPTIMIZED_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "SAMBANOVA_OPTIMIZED_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'SAMBANOVA_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "SAMBANOVA_SYSTEM_OK")
    
    def test_different_model_sizes(self):
        """Test different SambaNova model sizes"""
        # Test with a different model size
        try:
            client = SambanovaClient(
                model_name="Meta-Llama-3.1-70B-Instruct",
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage = client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "model" in str(e).lower() or "not available" in str(e).lower():
                self.skipTest(f"Alternative SambaNova model not available: {e}")
            else:
                raise
    
    def test_fast_inference(self):
        """Test SambaNova's fast inference capabilities"""
        import time
        
        messages = [{"role": "user", "content": "Say 'SPEED_TEST_OK' quickly"}]
        
        start_time = time.time()
        responses, usage = self.client.chat(messages)
        end_time = time.time()
        
        # SambaNova should be fast
        self.assertLess(end_time - start_time, 10.0)  # Allow reasonable time
        self.assert_response_content(responses, "SPEED_TEST_OK")


if __name__ == '__main__':
    unittest.main()