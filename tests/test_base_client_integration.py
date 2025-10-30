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
from minions.clients.response import ChatResponse


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
        """
        Validate that chat response is properly formatted.

        All clients must now return ChatResponse objects with backward-compatible
        tuple unpacking support.
        """
        # Should be a ChatResponse instance
        self.assertIsInstance(result, ChatResponse,
            "All clients must now return ChatResponse")

        # Validate required fields
        self.assertIsInstance(result.responses, list,
            "ChatResponse.responses must be a list")
        self.assertGreater(len(result.responses), 0,
            "ChatResponse.responses must not be empty")
        self.assertIsInstance(result.responses[0], str,
            "ChatResponse.responses must contain strings")
        self.assertIsInstance(result.usage, Usage,
            "ChatResponse.usage must be a Usage object")
        self.assertGreater(result.usage.total_tokens, 0,
            "Usage must have total_tokens > 0")

        # Validate optional fields (if present)
        if result.done_reasons is not None:
            self.assertIsInstance(result.done_reasons, list,
                "ChatResponse.done_reasons must be a list if present")

        if result.tool_calls is not None:
            self.assertIsInstance(result.tool_calls, list,
                "ChatResponse.tool_calls must be a list if present")

        if result.audio is not None:
            self.assertIsInstance(result.audio, bytes,
                "ChatResponse.audio must be bytes if present")

        if result.metadata is not None:
            self.assertIsInstance(result.metadata, dict,
                "ChatResponse.metadata must be dict if present")

        # Test backward compatibility: 2-tuple unpacking must work
        responses, usage = result
        self.assertEqual(responses, result.responses,
            "Tuple unpacking [0] must match .responses")
        self.assertEqual(usage, result.usage,
            "Tuple unpacking [1] must match .usage")

        # If done_reasons exists, test 3-tuple unpacking
        if result.done_reasons is not None:
            responses2, usage2, done_reasons = result
            self.assertEqual(responses2, result.responses,
                "3-tuple unpacking [0] must match .responses")
            self.assertEqual(usage2, result.usage,
                "3-tuple unpacking [1] must match .usage")
            self.assertEqual(done_reasons, result.done_reasons,
                "3-tuple unpacking [2] must match .done_reasons")
    
    def assert_response_content(self, responses: List[str], expected_content: str):
        """Assert response contains expected content"""
        self.assertTrue(
            any(expected_content.lower() in response.lower() for response in responses),
            f"Expected '{expected_content}' not found in responses: {responses}"
        )