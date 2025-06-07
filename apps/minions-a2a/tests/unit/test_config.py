#!/usr/bin/env python3
"""
Unit tests for A2A-Minions configuration management.
Tests configuration parsing, validation, and environment variable handling.
"""

import unittest
import os
from unittest.mock import patch, MagicMock

from a2a_minions.config import (
    ProtocolType, ProviderType, MinionsConfig, ConfigManager
)


class TestEnums(unittest.TestCase):
    """Test enum types."""
    
    def test_protocol_types(self):
        """Test protocol type enum values."""
        self.assertEqual(ProtocolType.MINION.value, "minion")
        self.assertEqual(ProtocolType.MINIONS.value, "minions")
    
    def test_provider_types(self):
        """Test provider type enum values."""
        providers = [
            ("ollama", ProviderType.OLLAMA),
            ("openai", ProviderType.OPENAI),
            ("anthropic", ProviderType.ANTHROPIC),
            ("together", ProviderType.TOGETHER),
            ("deepseek", ProviderType.DEEPSEEK),
            ("gemini", ProviderType.GEMINI),
            ("groq", ProviderType.GROQ),
            ("mlx", ProviderType.MLX),
            ("cartesia-mlx", ProviderType.CARTESIA_MLX)
        ]
        
        for value, enum_type in providers:
            self.assertEqual(enum_type.value, value)


class TestMinionsConfig(unittest.TestCase):
    """Test MinionsConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MinionsConfig()
        
        # Check defaults
        self.assertEqual(config.protocol, ProtocolType.MINION)
        self.assertEqual(config.local_provider, ProviderType.OLLAMA)
        self.assertEqual(config.local_model, "llama3.2")
        self.assertEqual(config.local_temperature, 0.0)
        self.assertEqual(config.local_max_tokens, 4096)
        self.assertEqual(config.remote_provider, ProviderType.OPENAI)
        self.assertEqual(config.remote_model, "gpt-4o")
        self.assertEqual(config.remote_temperature, 0.0)
        self.assertEqual(config.remote_max_tokens, 4096)
        self.assertEqual(config.max_rounds, 3)
        self.assertEqual(config.num_ctx, 4096)
        self.assertIsNone(config.max_jobs_per_round)
        self.assertEqual(config.num_tasks_per_round, 3)
        self.assertEqual(config.num_samples_per_task, 1)
        self.assertEqual(config.chunking_strategy, "chunk_by_section")
        self.assertIsNone(config.use_retrieval)
        self.assertFalse(config.privacy_mode)
        self.assertTrue(config.enable_streaming)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MinionsConfig(
            protocol=ProtocolType.MINIONS,
            local_provider=ProviderType.ANTHROPIC,
            local_model="claude-3",
            local_temperature=0.7,
            local_max_tokens=2048,
            remote_provider=ProviderType.GEMINI,
            remote_model="gemini-pro",
            remote_temperature=0.5,
            remote_max_tokens=8192,
            max_rounds=5,
            num_ctx=8192,
            max_jobs_per_round=10,
            num_tasks_per_round=5,
            num_samples_per_task=3,
            chunking_strategy="chunk_by_paragraph",
            use_retrieval="bm25",
            privacy_mode=True,
            enable_streaming=False
        )
        
        self.assertEqual(config.protocol, ProtocolType.MINIONS)
        self.assertEqual(config.local_provider, ProviderType.ANTHROPIC)
        self.assertEqual(config.local_model, "claude-3")
        self.assertEqual(config.local_temperature, 0.7)
        self.assertEqual(config.max_rounds, 5)
        self.assertEqual(config.max_jobs_per_round, 10)
        self.assertEqual(config.use_retrieval, "bm25")
        self.assertTrue(config.privacy_mode)
        self.assertFalse(config.enable_streaming)
    
    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        config = MinionsConfig(local_temperature=1.5, remote_temperature=2.0)
        self.assertEqual(config.local_temperature, 1.5)
        self.assertEqual(config.remote_temperature, 2.0)
        
        # Invalid temperatures
        with self.assertRaises(ValueError):
            MinionsConfig(local_temperature=-0.1)
        
        with self.assertRaises(ValueError):
            MinionsConfig(remote_temperature=2.1)
    
    def test_max_rounds_validation(self):
        """Test max_rounds validation."""
        # Valid values
        config = MinionsConfig(max_rounds=1)
        self.assertEqual(config.max_rounds, 1)
        
        config = MinionsConfig(max_rounds=10)
        self.assertEqual(config.max_rounds, 10)
        
        # Invalid values
        with self.assertRaises(ValueError):
            MinionsConfig(max_rounds=0)
        
        with self.assertRaises(ValueError):
            MinionsConfig(max_rounds=11)
    
    def test_retrieval_validation(self):
        """Test retrieval method validation."""
        valid_methods = ["bm25", "embedding", "multimodal-embedding"]
        
        for method in valid_methods:
            config = MinionsConfig(use_retrieval=method)
            self.assertEqual(config.use_retrieval, method)
        
        # Invalid method
        with self.assertRaises(ValueError):
            MinionsConfig(use_retrieval="invalid-method")


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        default_config = self.config_manager.default_config
        
        self.assertIsInstance(default_config, MinionsConfig)
        self.assertEqual(default_config.local_provider, ProviderType.OLLAMA)
        self.assertEqual(default_config.remote_provider, ProviderType.OPENAI)
    
    @patch.dict(os.environ, {
        "MINIONS_LOCAL_PROVIDER": "anthropic",
        "MINIONS_LOCAL_MODEL": "claude-3",
        "MINIONS_REMOTE_PROVIDER": "gemini",
        "MINIONS_REMOTE_MODEL": "gemini-ultra",
        "MINIONS_MAX_ROUNDS": "7",
        "MINIONS_PRIVACY_MODE": "true"
    })
    def test_environment_variables(self):
        """Test configuration from environment variables."""
        # Create new config manager to pick up env vars
        config_manager = ConfigManager()
        config = config_manager.default_config
        
        self.assertEqual(config.local_provider, ProviderType.ANTHROPIC)
        self.assertEqual(config.local_model, "claude-3")
        self.assertEqual(config.remote_provider, ProviderType.GEMINI)
        self.assertEqual(config.remote_model, "gemini-ultra")
        self.assertEqual(config.max_rounds, 7)
        self.assertTrue(config.privacy_mode)
    
    @patch.dict(os.environ, {"MINIONS_PRIVACY_MODE": "false"})
    def test_privacy_mode_false(self):
        """Test privacy mode false from environment."""
        config_manager = ConfigManager()
        self.assertFalse(config_manager.default_config.privacy_mode)
    
    def test_parse_empty_metadata(self):
        """Test parsing empty A2A metadata."""
        config = self.config_manager.parse_a2a_metadata(None)
        
        # Should return default config
        self.assertEqual(config.local_provider, self.config_manager.default_config.local_provider)
        self.assertEqual(config.max_rounds, self.config_manager.default_config.max_rounds)
    
    def test_parse_partial_metadata(self):
        """Test parsing partial A2A metadata."""
        metadata = {
            "local_model": "llama3.3",
            "max_rounds": 4,
            "privacy_mode": True
        }
        
        config = self.config_manager.parse_a2a_metadata(metadata)
        
        # Overridden values
        self.assertEqual(config.local_model, "llama3.3")
        self.assertEqual(config.max_rounds, 4)
        self.assertTrue(config.privacy_mode)
        
        # Default values
        self.assertEqual(config.local_provider, self.config_manager.default_config.local_provider)
        self.assertEqual(config.remote_model, self.config_manager.default_config.remote_model)
    
    def test_parse_complete_metadata(self):
        """Test parsing complete A2A metadata."""
        metadata = {
            "protocol": "minions",
            "local_provider": "groq",
            "local_model": "mixtral-8x7b",
            "local_temperature": 0.3,
            "local_max_tokens": 2048,
            "remote_provider": "deepseek",
            "remote_model": "deepseek-chat",
            "remote_temperature": 0.8,
            "remote_max_tokens": 4096,
            "max_rounds": 6,
            "num_ctx": 16384,
            "max_jobs_per_round": 20,
            "num_tasks_per_round": 8,
            "num_samples_per_task": 2,
            "chunking_strategy": "chunk_by_tokens",
            "use_retrieval": "embedding",
            "privacy_mode": False,
            "enable_streaming": True
        }
        
        config = self.config_manager.parse_a2a_metadata(metadata)
        
        # Check all values
        self.assertEqual(config.protocol, ProtocolType.MINIONS)
        self.assertEqual(config.local_provider, ProviderType.GROQ)
        self.assertEqual(config.local_model, "mixtral-8x7b")
        self.assertEqual(config.local_temperature, 0.3)
        self.assertEqual(config.local_max_tokens, 2048)
        self.assertEqual(config.remote_provider, ProviderType.DEEPSEEK)
        self.assertEqual(config.remote_model, "deepseek-chat")
        self.assertEqual(config.remote_temperature, 0.8)
        self.assertEqual(config.remote_max_tokens, 4096)
        self.assertEqual(config.max_rounds, 6)
        self.assertEqual(config.num_ctx, 16384)
        self.assertEqual(config.max_jobs_per_round, 20)
        self.assertEqual(config.num_tasks_per_round, 8)
        self.assertEqual(config.num_samples_per_task, 2)
        self.assertEqual(config.chunking_strategy, "chunk_by_tokens")
        self.assertEqual(config.use_retrieval, "embedding")
        self.assertFalse(config.privacy_mode)
        self.assertTrue(config.enable_streaming)
    
    def test_create_client_configs_minimal(self):
        """Test creating minimal client configurations."""
        config = MinionsConfig()
        client_configs = self.config_manager.create_client_configs(config)
        
        # Check structure
        self.assertIn("local", client_configs)
        self.assertIn("remote", client_configs)
        
        # Check local config
        local = client_configs["local"]
        self.assertEqual(local["model_name"], config.local_model)
        self.assertEqual(local["temperature"], config.local_temperature)
        self.assertEqual(local["max_tokens"], config.local_max_tokens)
        
        # Check remote config
        remote = client_configs["remote"]
        self.assertEqual(remote["model_name"], config.remote_model)
        self.assertEqual(remote["temperature"], config.remote_temperature)
        self.assertEqual(remote["max_tokens"], config.remote_max_tokens)
    
    def test_create_client_configs_ollama(self):
        """Test creating Ollama-specific client configuration."""
        config = MinionsConfig(
            local_provider=ProviderType.OLLAMA,
            protocol=ProtocolType.MINIONS,
            num_ctx=8192
        )
        
        client_configs = self.config_manager.create_client_configs(config)
        local = client_configs["local"]
        
        # Check Ollama-specific settings
        self.assertEqual(local["num_ctx"], 8192)
        self.assertTrue(local["use_async"])  # True for MINIONS protocol
    
    def test_create_client_configs_non_ollama(self):
        """Test creating non-Ollama client configuration."""
        config = MinionsConfig(
            local_provider=ProviderType.OPENAI,
            num_ctx=8192  # Should not be included
        )
        
        client_configs = self.config_manager.create_client_configs(config)
        local = client_configs["local"]
        
        # num_ctx should not be in non-Ollama configs
        self.assertNotIn("num_ctx", local)
        self.assertNotIn("use_async", local)
    
    def test_ignored_metadata_keys(self):
        """Test that unknown metadata keys are ignored."""
        metadata = {
            "local_model": "test-model",
            "unknown_key": "ignored",
            "another_unknown": 123
        }
        
        # Should not raise an error
        config = self.config_manager.parse_a2a_metadata(metadata)
        self.assertEqual(config.local_model, "test-model")


if __name__ == "__main__":
    unittest.main()