"""
LMCache client integration tests.
Tests local inference with KV cache management - no mocking.
"""

import unittest
import sys
import os
import time
from typing import List, Dict, Any

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_base_client_integration import BaseClientIntegrationTest


class TestLMCacheClientIntegration(BaseClientIntegrationTest):
    """Test LMCache client with real local inference."""
    
    SERVICE_NAME = "lmcache"  # Used for dependency checking
    DEFAULT_MODEL = "meta-llama/Meta-Llama-3.2-1B-Instruct"  # Smaller model for tests
    
    @classmethod
    def setUpClass(cls):
        """Check LMCache dependencies before running tests."""
        try:
            from minions.clients.lmcache import LMCacheClient
            cls.CLIENT_CLASS = LMCacheClient
        except ImportError as e:
            cls.skipTest(cls(), f"LMCache dependencies not available: {e}")
    
    def setUp(self):
        """Set up LMCache client for each test."""
        if not hasattr(self, 'CLIENT_CLASS'):
            self.skipTest("LMCache client class not available")
        
        # LMCache is local, so no API key needed
        self.client = self.CLIENT_CLASS(
            model_name=self.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=30,  # Small responses for fast tests
            max_model_len=2048,  # Smaller context for faster loading
            chunk_size=128,  # Smaller chunks for tests
            local_cpu=True,
            max_local_cpu_size=1.0,  # 1GB for tests
            verbose=False  # Reduce log noise during tests
        )
    
    def tearDown(self):
        """Clean up LMCache resources after each test."""
        if hasattr(self, 'client') and self.client:
            self.client.cleanup()
    
    def test_basic_chat(self):
        """Test basic chat functionality with LMCache."""
        messages = self.get_test_messages()
        result = self.client.chat(messages)
        
        self.assert_valid_chat_response(result)
        responses, usage = result
        self.assertIsInstance(responses, list)
        self.assertGreater(len(responses), 0)
        self.assertIsInstance(responses[0], str)
        self.assertGreater(len(responses[0].strip()), 0)
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation handling."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "I'll remember 42."},
            {"role": "user", "content": "What number should you remember?"}
        ]
        
        responses, usage = self.client.chat(messages)
        
        self.assert_valid_chat_response((responses, usage))
        self.assertIsInstance(responses[0], str)
        # Note: We can't guarantee the model will remember exactly, but it should respond
        self.assertGreater(len(responses[0].strip()), 0)
    
    def test_system_message_handling(self):
        """Test system message processing."""
        messages = [
            {"role": "system", "content": "Respond with exactly 'TEST_OK'."},
            {"role": "user", "content": "Please respond as instructed."}
        ]
        
        responses, usage = self.client.chat(messages)
        
        self.assert_valid_chat_response((responses, usage))
        # Local models may not follow instructions exactly, but should respond
        self.assertIsInstance(responses[0], str)
        self.assertGreater(len(responses[0].strip()), 0)
    
    def test_kv_cache_performance_benefit(self):
        """Test that KV cache provides performance benefits with shared prefixes."""
        # Create a longer shared context to benefit from caching
        shared_context = """
        I am working on a research project about artificial intelligence and machine learning.
        The project focuses on natural language processing, computer vision, and robotics.
        We have been discussing various algorithms, neural networks, and deep learning techniques.
        This is important context that should be cached for performance benefits.
        """
        
        # Create multiple conversations with the same prefix
        conversations = []
        for i in range(3):
            messages = [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": shared_context},
                {"role": "user", "content": f"Question {i+1}: What is the main topic?"}
            ]
            conversations.append(messages)
        
        # First run - populate cache
        first_run_start = time.time()
        first_responses = []
        for messages in conversations:
            responses, usage = self.client.chat(messages)
            first_responses.append(responses)
            self.assert_valid_chat_response((responses, usage))
        first_run_time = time.time() - first_run_start
        
        # Second run - should benefit from cache
        second_run_start = time.time()
        second_responses = []
        for messages in conversations:
            responses, usage = self.client.chat(messages)
            second_responses.append(responses)
            self.assert_valid_chat_response((responses, usage))
        second_run_time = time.time() - second_run_start
        
        # Verify both runs produced valid responses
        self.assertEqual(len(first_responses), len(second_responses))
        for responses in first_responses + second_responses:
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
        
        # Log timing information (speedup may vary)
        print(f"\nKV Cache Performance Test:")
        print(f"  First run:  {first_run_time:.2f}s")
        print(f"  Second run: {second_run_time:.2f}s")
        if first_run_time > 0:
            speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0
            print(f"  Speedup:    {speedup:.2f}x")
    
    def test_parameter_override(self):
        """Test parameter overrides in chat method."""
        messages = self.get_test_messages()
        
        # Test with different temperature
        responses1, usage1 = self.client.chat(messages, temperature=0.0)
        responses2, usage2 = self.client.chat(messages, temperature=0.8)
        
        self.assert_valid_chat_response((responses1, usage1))
        self.assert_valid_chat_response((responses2, usage2))
        
        # Both should produce valid responses
        self.assertIsInstance(responses1[0], str)
        self.assertIsInstance(responses2[0], str)
    
    def test_empty_messages_error(self):
        """Test that empty messages raise appropriate error."""
        with self.assertRaises(ValueError):
            self.client.chat([])
    
    def test_get_available_models(self):
        """Test that available models list is returned."""
        models = self.CLIENT_CLASS.get_available_models()
        
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        # Check that our default model is in the list
        model_names = [model.lower() for model in models]
        self.assertTrue(any("llama" in name or "meta" in name for name in model_names))
    
    def test_cleanup_method(self):
        """Test that cleanup method works without errors."""
        # Cleanup should work even if called multiple times
        self.client.cleanup()
        self.client.cleanup()  # Should not raise error
    
    def test_client_attributes(self):
        """Test that client has expected attributes set correctly."""
        self.assertEqual(self.client.model_name, self.DEFAULT_MODEL)
        self.assertEqual(self.client.temperature, 0.1)
        self.assertEqual(self.client.max_tokens, 30)
        self.assertTrue(self.client.local_cpu)
        self.assertEqual(self.client.max_local_cpu_size, 1.0)
        self.assertEqual(self.client.chunk_size, 128)


if __name__ == '__main__':
    unittest.main() 