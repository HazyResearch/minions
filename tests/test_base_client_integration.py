"""
Base integration test class for minions clients.
Real API calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the tests directory to the path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_checker import APIKeyChecker
from minions.usage import Usage


class BaseClientIntegrationTest(unittest.TestCase):
    """Base class for real API integration tests"""
    
    # Subclasses should override these
    CLIENT_CLASS = None
    SERVICE_NAME = None
    DEFAULT_MODEL = None
    
    @classmethod
    def setUpClass(cls):
        """Check API key availability before running tests"""
        if not cls.SERVICE_NAME:
            cls.skipTest(cls(), "SERVICE_NAME not defined in test class")
        
        if not APIKeyChecker.warn_if_missing(cls.SERVICE_NAME):
            cls.skipTest(cls(), f"API key for {cls.SERVICE_NAME} not available")
    
    def setUp(self):
        """Set up client for each test"""
        api_key = APIKeyChecker.check_key(self.SERVICE_NAME)
        if not api_key:
            self.skipTest(f"No API key for {self.SERVICE_NAME}")
        
        self.client = self.CLIENT_CLASS(
            model_name=self.DEFAULT_MODEL,
            api_key=api_key,
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=50     # Small responses for fast tests
        )
    
    def get_test_messages(self) -> List[Dict[str, Any]]:
        """Standard test messages for consistency"""
        return [
            {"role": "user", "content": "Say exactly 'test successful' and nothing else"}
        ]
    
    def assert_valid_chat_response(self, result):
        """Assert that chat response has correct format"""
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 2)
        
        responses, usage = result[0], result[1]
        self.assertIsInstance(responses, list)
        self.assertGreater(len(responses), 0)
        self.assertIsInstance(responses[0], str)
        self.assertIsInstance(usage, Usage)
        self.assertGreater(usage.total_tokens, 0)
        
        # Additional validation for clients that return more values
        if len(result) >= 3:
            # Third element is typically finish_reasons or done_reasons
            finish_reasons = result[2]
            self.assertIsInstance(finish_reasons, list)
            
        if len(result) >= 4:
            # Fourth element is typically tools
            tools = result[3]
            self.assertIsInstance(tools, list)
    
    def assert_response_content(self, responses: List[str], expected_content: str):
        """Assert response contains expected content"""
        self.assertTrue(
            any(expected_content.lower() in response.lower() for response in responses),
            f"Expected '{expected_content}' not found in responses: {responses}"
        )