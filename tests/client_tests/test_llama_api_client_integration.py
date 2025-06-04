"""
Llama API client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.llama_api import LlamaApiClient
from test_base_client_integration import BaseClientIntegrationTest


class TestLlamaAPIClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = LlamaApiClient
    SERVICE_NAME = "llama_api"
    DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        try:
            result = self.client.chat(messages)
            
            self.assert_valid_chat_response(result)
            responses, usage = result
            self.assert_response_content(responses, "test successful")
        except Exception as e:
            if "Check job status" in str(e) or "inference completed jobs" in str(e):
                self.skipTest(f"LlamaAPI service unavailable - models not ready: {e}")
            else:
                raise
    
    def test_llama_specific_features(self):
        """Test Llama API specific features"""
        messages = [
            {"role": "user", "content": "Respond with exactly: 'LLAMA_API_OK'"}
        ]
        
        try:
            responses, usage = self.client.chat(messages)
            self.assert_response_content(responses, "LLAMA_API_OK")
        except Exception as e:
            if "Check job status" in str(e) or "inference completed jobs" in str(e):
                self.skipTest(f"LlamaAPI service unavailable - models not ready: {e}")
            else:
                raise
    
    def test_system_message(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Always respond with exactly 'LLAMA_SYSTEM_OK'"},
            {"role": "user", "content": "Hello"}
        ]
        
        try:
            responses, usage = self.client.chat(messages)
            self.assert_response_content(responses, "LLAMA_SYSTEM_OK")
        except Exception as e:
            if "Check job status" in str(e) or "inference completed jobs" in str(e):
                self.skipTest(f"LlamaAPI service unavailable - models not ready: {e}")
            else:
                raise



if __name__ == '__main__':
    unittest.main()