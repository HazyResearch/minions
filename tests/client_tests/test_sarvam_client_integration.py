"""
Sarvam client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.sarvam import SarvamClient
from test_base_client_integration import BaseClientIntegrationTest


class TestSarvamClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = SarvamClient
    SERVICE_NAME = "sarvam"
    DEFAULT_MODEL = "sarvamai/sarvam-2b-v0.5"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assert_response_content(responses, "test successful")
    
    def test_sarvam_features(self):
        """Test Sarvam AI specific features"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'SARVAM_AI_OK'"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "SARVAM_AI_OK")
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'SARVAM_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        responses, usage = self.client.chat(messages)
        self.assert_response_content(responses, "SARVAM_SYSTEM_OK")
    
    def test_multilingual_capabilities(self):
        """Test Sarvam's multilingual capabilities"""
        # Test with English
        messages_en = [
            {"role": "user", "content": "Say 'ENGLISH_OK' in English"}
        ]
        
        responses, usage = self.client.chat(messages_en)
        self.assert_response_content(responses, "ENGLISH_OK")
    
    def test_different_sarvam_models(self):
        """Test different Sarvam models"""
        # Test with alternative model if available
        try:
            client = SarvamClient(
                model_name="sarvamai/sarvam-1",
                max_tokens=30
            )
            
            messages = [{"role": "user", "content": "Test"}]
            responses, usage = client.chat(messages)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            
        except Exception as e:
            if "model" in str(e).lower() or "not available" in str(e).lower():
                self.skipTest(f"Alternative Sarvam model not available: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()