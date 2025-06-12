"""
Distributed Inference client integration tests.
Real API calls only - zero mocking.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tests directory to the path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.distributed_inference import DistributedInferenceClient
from test_base_client_integration import BaseClientIntegrationTest


class TestDistributedInferenceClientIntegration(BaseClientIntegrationTest):
    CLIENT_CLASS = DistributedInferenceClient
    SERVICE_NAME = "distributed_inference"
    DEFAULT_MODEL = None  # Model selection is automatic
    
    @classmethod
    def setUpClass(cls):
        """Override base class setup since distributed inference doesn't require API keys"""
        # Don't check for API keys since they're optional for distributed inference
        pass
    
    def setUp(self):
        """Set up client for each test"""
        # For distributed inference, API key is optional
        api_key = os.getenv("MINIONS_API_KEY")
        base_url = os.getenv("MINIONS_COORDINATOR_URL", "http://localhost:8080")
        
        self.client = self.CLIENT_CLASS(
            model_name=None,  # Let coordinator select
            api_key=api_key,
            temperature=0.1,
            max_tokens=50,
            base_url=base_url
        )
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        messages = self.get_test_messages()
        
        try:
            result = self.client.chat(messages)
            
            # Distributed inference returns (responses, usage, done_reasons) tuple
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
            responses, usage, done_reasons = result
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            self.assertIsInstance(done_reasons, list)
            self.assertGreater(len(done_reasons), 0)
            self.assert_response_content(responses, "test successful")
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
                self.skipTest("Network coordinator not running at configured URL")
            raise
    
    def test_network_status(self):
        """Test getting network status"""
        try:
            status = self.client.get_network_status()
            
            self.assertIsInstance(status, dict)
            self.assertIn("status", status)
            self.assertIn("total_nodes", status)
            self.assertIn("healthy_nodes", status)
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
                self.skipTest("Network coordinator not running at configured URL")
            raise
    
    def test_list_nodes(self):
        """Test listing nodes"""
        try:
            nodes_info = self.client.list_nodes()
            
            self.assertIsInstance(nodes_info, dict)
            self.assertIn("total_nodes", nodes_info)
            self.assertIn("healthy_nodes", nodes_info)
            self.assertIn("nodes", nodes_info)
            
            # If there are nodes, check their structure
            if nodes_info["total_nodes"] > 0:
                nodes = nodes_info["nodes"]
                self.assertIsInstance(nodes, dict)
                for node_url, node_info in nodes.items():
                    self.assertIn("status", node_info)
                    self.assertIn("model_available", node_info)
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
                self.skipTest("Network coordinator not running at configured URL")
            elif "401" in str(e):
                self.skipTest("Authentication required but no API key provided")
            raise
    
    def test_model_preference(self):
        """Test specifying a model preference"""
        # Create client with specific model preference
        client = self.CLIENT_CLASS(
            model_name="llama3.2:1b",  # Specific model
            api_key=os.getenv("MINIONS_API_KEY"),
            temperature=0.1,
            max_tokens=50,
            base_url=os.getenv("MINIONS_COORDINATOR_URL", "http://localhost:8080")
        )
        
        messages = [{"role": "user", "content": "Say 'model test'"}]
        
        try:
            responses, usage, done_reasons = client.chat(messages)
            self.assert_response_content(responses, "model test")
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
                self.skipTest("Network coordinator not running at configured URL")
            elif "404" in str(e):
                self.skipTest("Requested model not available in the network")
            raise
    
    def test_complete_method(self):
        """Test the complete method"""
        try:
            prompts = "Complete this: The sky is"
            responses, usage = self.client.complete(prompts)
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            self.assertGreater(len(responses[0]), 0)
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
                self.skipTest("Network coordinator not running at configured URL")
            raise


if __name__ == '__main__':
    unittest.main() 