#!/usr/bin/env python3
"""
Unit tests for A2A-Minions client factory.
Tests client creation, connection pooling, and protocol instantiation.
"""

import unittest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock the minions modules before importing client_factory
sys.modules['minions'] = MagicMock()
sys.modules['minions.minion'] = MagicMock()
sys.modules['minions.minions'] = MagicMock()
sys.modules['minions.clients'] = MagicMock()
sys.modules['minions.clients.ollama'] = MagicMock()
sys.modules['minions.clients.openai'] = MagicMock()
sys.modules['minions.clients.anthropic'] = MagicMock()
sys.modules['minions.clients.together'] = MagicMock()
sys.modules['minions.clients.deepseek'] = MagicMock()
sys.modules['minions.clients.groq'] = MagicMock()
sys.modules['minions.clients.gemini'] = MagicMock()
sys.modules['minions.clients.mlx'] = MagicMock()
sys.modules['minions.clients.cartesia_mlx'] = MagicMock()

from a2a_minions.client_factory import ClientFactory
from a2a_minions.config import MinionsConfig, ProviderType, ProtocolType


class TestClientFactory(unittest.TestCase):
    """Test ClientFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock client classes
        self.mock_ollama_client = MagicMock()
        self.mock_openai_client = MagicMock()
        self.mock_anthropic_client = MagicMock()
        
        # Create mock protocol classes
        self.mock_minion = MagicMock()
        self.mock_minions = MagicMock()
        
        # Patch the imports
        with patch.object(ClientFactory, '_import_minions_modules'):
            self.factory = ClientFactory()
        
        # Manually set up the mocked imports
        self.factory.Minion = self.mock_minion
        self.factory.Minions = self.mock_minions
        self.factory.clients = {
            ProviderType.OLLAMA: self.mock_ollama_client,
            ProviderType.OPENAI: self.mock_openai_client,
            ProviderType.ANTHROPIC: self.mock_anthropic_client
        }
    
    def test_get_client_key(self):
        """Test client key generation."""
        # Simple config
        config1 = {"model_name": "llama3.2", "temperature": 0.5}
        key1 = self.factory._get_client_key(ProviderType.OLLAMA, config1)
        
        # Same config should generate same key
        config2 = {"model_name": "llama3.2", "temperature": 0.5}
        key2 = self.factory._get_client_key(ProviderType.OLLAMA, config2)
        self.assertEqual(key1, key2)
        
        # Different config should generate different key
        config3 = {"model_name": "llama3.2", "temperature": 0.7}
        key3 = self.factory._get_client_key(ProviderType.OLLAMA, config3)
        self.assertNotEqual(key1, key3)
        
        # Different provider should generate different key
        key4 = self.factory._get_client_key(ProviderType.OPENAI, config1)
        self.assertNotEqual(key1, key4)
    
    def test_get_client_key_excludes_non_serializable(self):
        """Test that client key generation excludes non-serializable fields."""
        # Config with non-serializable field
        config = {
            "model_name": "test",
            "structured_output_schema": MagicMock()  # Non-serializable
        }
        
        # Should not raise
        key = self.factory._get_client_key(ProviderType.OLLAMA, config)
        self.assertIsInstance(key, str)
    
    def test_create_client_new(self):
        """Test creating a new client."""
        config = {"model_name": "llama3.2", "temperature": 0.5}
        
        # Mock client instance
        mock_instance = MagicMock()
        self.mock_ollama_client.return_value = mock_instance
        
        # Create client
        client = self.factory.create_client(ProviderType.OLLAMA, config)
        
        # Should have created new instance
        self.mock_ollama_client.assert_called_once_with(**config)
        self.assertEqual(client, mock_instance)
        
        # Should be in pool
        key = self.factory._get_client_key(ProviderType.OLLAMA, config)
        self.assertIn(key, self.factory._client_pool)
    
    def test_create_client_from_pool(self):
        """Test retrieving client from pool."""
        config = {"model_name": "gpt-4", "temperature": 0.3}
        
        # Create first client
        mock_instance = MagicMock()
        self.mock_openai_client.return_value = mock_instance
        client1 = self.factory.create_client(ProviderType.OPENAI, config)
        
        # Reset mock
        self.mock_openai_client.reset_mock()
        
        # Create second client with same config
        client2 = self.factory.create_client(ProviderType.OPENAI, config)
        
        # Should not have created new instance
        self.mock_openai_client.assert_not_called()
        
        # Should return same instance
        self.assertIs(client1, client2)
    
    def test_create_client_unsupported_provider(self):
        """Test creating client with unsupported provider."""
        with self.assertRaises(ValueError) as context:
            self.factory.create_client("invalid_provider", {})
        self.assertIn("Unsupported provider", str(context.exception))
    
    def test_create_client_invalid_config(self):
        """Test creating client with invalid configuration."""
        # Mock client that raises TypeError
        self.mock_openai_client.side_effect = TypeError("Missing required parameter")
        
        with self.assertRaises(TypeError) as context:
            self.factory.create_client(ProviderType.OPENAI, {})
        self.assertIn("Invalid configuration", str(context.exception))
    
    def test_create_minions_protocol_minion(self):
        """Test creating Minion protocol instance."""
        config = MinionsConfig(
            protocol=ProtocolType.MINION,
            local_provider=ProviderType.OLLAMA,
            remote_provider=ProviderType.OPENAI,
            max_rounds=3
        )
        
        # Mock client instances
        local_client = MagicMock()
        remote_client = MagicMock()
        self.mock_ollama_client.return_value = local_client
        self.mock_openai_client.return_value = remote_client
        
        # Mock protocol instance
        protocol_instance = MagicMock()
        self.mock_minion.return_value = protocol_instance
        
        # Create protocol
        protocol = self.factory.create_minions_protocol(config)
        
        # Check Minion was created with correct parameters
        self.mock_minion.assert_called_once_with(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=3,
            is_multi_turn=True
        )
        self.assertEqual(protocol, protocol_instance)
    
    def test_create_minions_protocol_minions(self):
        """Test creating Minions protocol instance."""
        config = MinionsConfig(
            protocol=ProtocolType.MINIONS,
            local_provider=ProviderType.ANTHROPIC,
            remote_provider=ProviderType.OPENAI,
            max_rounds=5
        )
        
        # Mock client instances
        local_client = MagicMock()
        remote_client = MagicMock()
        self.mock_anthropic_client.return_value = local_client
        self.mock_openai_client.return_value = remote_client
        
        # Mock protocol instance
        protocol_instance = MagicMock()
        self.mock_minions.return_value = protocol_instance
        
        # Create protocol
        protocol = self.factory.create_minions_protocol(config)
        
        # Check Minions was created with correct parameters
        self.mock_minions.assert_called_once_with(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=5,
            is_multi_turn=True
        )
        self.assertEqual(protocol, protocol_instance)
    
    def test_create_minions_protocol_invalid(self):
        """Test creating protocol with invalid type."""
        config = MinionsConfig()
        config.protocol = "invalid"  # Bypass enum validation
        
        with self.assertRaises(ValueError) as context:
            self.factory.create_minions_protocol(config)
        self.assertIn("Unsupported protocol", str(context.exception))
    
    def test_create_client_pair(self):
        """Test creating local and remote client pair."""
        config = MinionsConfig(
            local_provider=ProviderType.OLLAMA,
            local_model="llama3.2",
            remote_provider=ProviderType.OPENAI,
            remote_model="gpt-4"
        )
        
        # Mock client instances
        local_client = MagicMock()
        remote_client = MagicMock()
        self.mock_ollama_client.return_value = local_client
        self.mock_openai_client.return_value = remote_client
        
        # Create pair
        local, remote = self.factory.create_client_pair(config)
        
        # Check correct clients were created
        self.assertEqual(local, local_client)
        self.assertEqual(remote, remote_client)
        
        # Check correct configs were passed
        self.mock_ollama_client.assert_called_once()
        local_config = self.mock_ollama_client.call_args[1]
        self.assertEqual(local_config["model_name"], "llama3.2")
        
        self.mock_openai_client.assert_called_once()
        remote_config = self.mock_openai_client.call_args[1]
        self.assertEqual(remote_config["model_name"], "gpt-4")
    
    def test_get_local_client_config_ollama(self):
        """Test Ollama-specific local client configuration."""
        config = MinionsConfig(
            local_provider=ProviderType.OLLAMA,
            local_model="llama3.2",
            local_temperature=0.5,
            local_max_tokens=2048,
            num_ctx=8192,
            protocol=ProtocolType.MINIONS
        )
        
        client_config = self.factory._get_local_client_config(config)
        
        # Check base config
        self.assertEqual(client_config["model_name"], "llama3.2")
        self.assertEqual(client_config["temperature"], 0.5)
        self.assertEqual(client_config["max_tokens"], 2048)
        
        # Check Ollama-specific config
        self.assertEqual(client_config["num_ctx"], 8192)
        self.assertTrue(client_config["use_async"])
        
        # Should have structured output schema for MINIONS
        self.assertIn("structured_output_schema", client_config)
    
    def test_get_local_client_config_non_ollama(self):
        """Test non-Ollama local client configuration."""
        config = MinionsConfig(
            local_provider=ProviderType.OPENAI,
            local_model="gpt-3.5",
            local_temperature=0.7,
            local_max_tokens=1024,
            num_ctx=8192  # Should be ignored
        )
        
        client_config = self.factory._get_local_client_config(config)
        
        # Check base config
        self.assertEqual(client_config["model_name"], "gpt-3.5")
        self.assertEqual(client_config["temperature"], 0.7)
        self.assertEqual(client_config["max_tokens"], 1024)
        
        # Should not have Ollama-specific config
        self.assertNotIn("num_ctx", client_config)
        self.assertNotIn("use_async", client_config)
    
    def test_get_remote_client_config(self):
        """Test remote client configuration."""
        config = MinionsConfig(
            remote_provider=ProviderType.ANTHROPIC,
            remote_model="claude-3",
            remote_temperature=0.3,
            remote_max_tokens=4096
        )
        
        client_config = self.factory._get_remote_client_config(config)
        
        # Check config
        self.assertEqual(client_config["model_name"], "claude-3")
        self.assertEqual(client_config["temperature"], 0.3)
        self.assertEqual(client_config["max_tokens"], 4096)
        
        # Should only have basic config
        self.assertEqual(len(client_config), 3)
    
    def test_clear_pool(self):
        """Test clearing the client pool."""
        # Add some clients to pool
        config1 = {"model": "test1"}
        config2 = {"model": "test2"}
        
        self.mock_ollama_client.return_value = MagicMock()
        self.factory.create_client(ProviderType.OLLAMA, config1)
        self.factory.create_client(ProviderType.OPENAI, config2)
        
        # Should have clients in pool
        self.assertEqual(len(self.factory._client_pool), 2)
        
        # Clear pool
        self.factory.clear_pool()
        
        # Pool should be empty
        self.assertEqual(len(self.factory._client_pool), 0)
    
    def test_get_pool_stats(self):
        """Test getting pool statistics."""
        # Empty pool
        stats = self.factory.get_pool_stats()
        self.assertEqual(stats["total_clients"], 0)
        self.assertEqual(stats["providers"], 0)
        
        # Add clients
        self.mock_ollama_client.return_value = MagicMock()
        self.mock_openai_client.return_value = MagicMock()
        
        self.factory.create_client(ProviderType.OLLAMA, {"model": "test1"})
        self.factory.create_client(ProviderType.OLLAMA, {"model": "test2"})
        self.factory.create_client(ProviderType.OPENAI, {"model": "test3"})
        
        # Get stats
        stats = self.factory.get_pool_stats()
        self.assertEqual(stats["total_clients"], 3)
        self.assertEqual(stats["providers"], 2)  # OLLAMA and OPENAI
    
    def test_thread_safety(self):
        """Test thread safety of client pool."""
        import threading
        
        config = {"model_name": "test"}
        created_clients = []
        
        def create_client():
            client = self.factory.create_client(ProviderType.OLLAMA, config)
            created_clients.append(client)
        
        # Mock client creation
        self.mock_ollama_client.return_value = MagicMock()
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_client)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All clients should be the same instance (from pool)
        self.assertEqual(len(set(id(c) for c in created_clients)), 1)
        
        # Only one client should have been created
        self.assertEqual(self.mock_ollama_client.call_count, 1)


if __name__ == "__main__":
    unittest.main()