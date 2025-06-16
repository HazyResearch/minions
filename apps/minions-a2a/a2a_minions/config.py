"""
Configuration management for A2A-Minions integration.
"""

import os
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ProtocolType(str, Enum):
    """Supported Minions protocol types."""
    MINION = "minion"
    MINIONS = "minions"


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    GROQ = "groq"
    LEMONADE = "lemonade"
    MLX = "mlx"
    CARTESIA_MLX = "cartesia-mlx"
    DISTRIBUTED_INFERENCE = "distributed_inference"


class MinionsConfig(BaseModel):
    """Configuration for Minions protocol execution."""
    
    # Protocol selection
    protocol: ProtocolType = Field(default=ProtocolType.MINION)
    
    # Local model configuration
    local_provider: ProviderType = Field(default=ProviderType.OLLAMA)
    local_model: str = Field(default="llama3.2")
    local_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    local_max_tokens: int = Field(default=4096, gt=0)
    
    # Remote model configuration  
    remote_provider: ProviderType = Field(default=ProviderType.OPENAI)
    remote_model: str = Field(default="gpt-4o")
    remote_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    remote_max_tokens: int = Field(default=4096, gt=0)
    
    # Protocol parameters
    max_rounds: int = Field(default=3, ge=1, le=10)
    num_ctx: int = Field(default=4096, gt=0)
    
    # Minions-specific parameters
    max_jobs_per_round: Optional[int] = Field(default=None)
    num_tasks_per_round: int = Field(default=3, ge=1)
    num_samples_per_task: int = Field(default=1, ge=1)
    chunking_strategy: str = Field(default="chunk_by_section")
    use_retrieval: Optional[Literal["bm25", "embedding", "multimodal-embedding"]] = None
    
    # Privacy and security
    privacy_mode: bool = Field(default=False)
    
    # Streaming and callbacks
    enable_streaming: bool = Field(default=True)
    
    class Config:
        use_enum_values = True


class ConfigManager:
    """Manages configuration for A2A-Minions integration.
    
    The A2A protocol allows application-specific metadata. For Minions integration,
    the following metadata fields are supported:
    
    Required:
    - skill_id: "minion_query" or "minions_query"
    
    Optional (with defaults from environment):
    - local_provider: LLM provider for local model (e.g., "ollama", "mlx")
    - local_model: Model name for local provider (e.g., "llama3.2")
    - remote_provider: LLM provider for remote model (e.g., "openai", "anthropic")
    - remote_model: Model name for remote provider (e.g., "gpt-4o", "claude-3")
    - max_rounds: Maximum rounds for minions protocol (1-10)
    - num_tasks_per_round: Number of parallel tasks per round
    - privacy_mode: Enable privacy mode (boolean)
    - And other fields defined in MinionsConfig
    """
    
    def __init__(self):
        self.default_config = self._load_default_config()
    
    def _load_default_config(self) -> MinionsConfig:
        """Load default configuration from environment variables."""
        return MinionsConfig(
            # Read from environment variables if available
            local_provider=os.getenv("MINIONS_LOCAL_PROVIDER", "ollama"),
            local_model=os.getenv("MINIONS_LOCAL_MODEL", "llama3.2"),
            remote_provider=os.getenv("MINIONS_REMOTE_PROVIDER", "openai"),
            remote_model=os.getenv("MINIONS_REMOTE_MODEL", "gpt-4o"),
            max_rounds=int(os.getenv("MINIONS_MAX_ROUNDS", "3")),
            privacy_mode=os.getenv("MINIONS_PRIVACY_MODE", "false").lower() == "true",
        )
    
    def parse_a2a_metadata(self, metadata: Optional[Dict[str, Any]]) -> MinionsConfig:
        """Parse A2A task metadata to create Minions configuration."""
        if not metadata:
            return self.default_config
        
        # Create config from defaults and override with metadata
        config_dict = self.default_config.dict()
        
        # Map A2A metadata keys to Minions config keys
        metadata_mapping = {
            "protocol": "protocol",
            "local_provider": "local_provider", 
            "local_model": "local_model",
            "local_temperature": "local_temperature",
            "local_max_tokens": "local_max_tokens",
            "remote_provider": "remote_provider",
            "remote_model": "remote_model", 
            "remote_temperature": "remote_temperature",
            "remote_max_tokens": "remote_max_tokens",
            "max_rounds": "max_rounds",
            "num_ctx": "num_ctx",
            "max_jobs_per_round": "max_jobs_per_round",
            "num_tasks_per_round": "num_tasks_per_round",
            "num_samples_per_task": "num_samples_per_task",
            "chunking_strategy": "chunking_strategy",
            "use_retrieval": "use_retrieval",
            "privacy_mode": "privacy_mode",
            "enable_streaming": "enable_streaming",
        }
        
        # Update config with metadata values
        for a2a_key, config_key in metadata_mapping.items():
            if a2a_key in metadata:
                config_dict[config_key] = metadata[a2a_key]
        
        return MinionsConfig(**config_dict)
    
    def create_client_configs(self, config: MinionsConfig) -> Dict[str, Dict[str, Any]]:
        """Create client configuration dictionaries for local and remote clients."""
        
        local_config = {
            "model_name": config.local_model,
            "temperature": config.local_temperature,
            "max_tokens": config.local_max_tokens,
        }
        
        remote_config = {
            "model_name": config.remote_model, 
            "temperature": config.remote_temperature,
            "max_tokens": config.remote_max_tokens,
        }
        
        # Add provider-specific configurations
        if config.local_provider == ProviderType.OLLAMA:
            local_config.update({
                "num_ctx": config.num_ctx,
                "use_async": config.protocol == ProtocolType.MINIONS,
            })
        elif config.local_provider == ProviderType.DISTRIBUTED_INFERENCE:
            local_config.update({
                "base_url": "http://localhost:8080",
                "timeout": 30,
            })
            
        return {
            "local": local_config,
            "remote": remote_config,
        } 