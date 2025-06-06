"""
Client factory for instantiating Minions protocol clients.
"""

import sys
import os
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import logging

# Add the parent directory to Python path to import minions
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .config import MinionsConfig, ProviderType, ProtocolType

logger = logging.getLogger(__name__)

class ClientFactory:
    """Factory for creating Minions protocol clients."""
    
    def __init__(self):
        self._import_minions_modules()
    
    def _import_minions_modules(self):
        """Import Minions modules and clients."""
        try:
            # Import Minions core modules
            from minions.minion import Minion
            from minions.minions import Minions
            
            # Import client modules
            from minions.clients.ollama import OllamaClient
            from minions.clients.openai import OpenAIClient
            from minions.clients.anthropic import AnthropicClient
            from minions.clients.together import TogetherClient
            from minions.clients.deepseek import DeepSeekClient
            from minions.clients.groq import GroqClient
            from minions.clients.gemini import GeminiClient
            
            # Store references
            self.Minion = Minion
            self.Minions = Minions
            self.clients = {
                ProviderType.OLLAMA: OllamaClient,
                ProviderType.OPENAI: OpenAIClient,
                ProviderType.ANTHROPIC: AnthropicClient,
                ProviderType.TOGETHER: TogetherClient,
                ProviderType.DEEPSEEK: DeepSeekClient,
                ProviderType.GROQ: GroqClient,
                ProviderType.GEMINI: GeminiClient,
            }
            
            # Try to import optional clients
            try:
                from minions.clients.mlx import MLXLMClient
                self.clients[ProviderType.MLX] = MLXLMClient
            except ImportError:
                pass
            
            try:
                from minions.clients.cartesia_mlx import CartesiaMLXClient
                self.clients[ProviderType.CARTESIA_MLX] = CartesiaMLXClient
            except ImportError:
                pass
                

                
        except ImportError as e:
            logger.error(f"Failed to import Minions modules: {e}")
            logger.error(f"Make sure minions package is installed and in PYTHONPATH")
            raise ImportError(f"Failed to import Minions modules: {e}")
    
    def create_client(self, provider: ProviderType, config: Dict[str, Any]) -> Any:
        """Create a client instance for the specified provider."""
        
        if provider not in self.clients:
            raise ValueError(f"Unsupported provider: {provider}")
        
        client_class = self.clients[provider]
        
        # Create client with configuration
        try:
            return client_class(**config)
        except TypeError as e:
            raise TypeError(f"Invalid configuration for {provider} client: {e}")
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise RuntimeError(f"Failed to create {provider} client: {e}")
    
    def create_minions_protocol(self, config: MinionsConfig) -> Any:
        """Create the appropriate Minions protocol instance."""
        
        # Create local and remote clients
        local_client, remote_client = self.create_client_pair(config)
        
        if config.protocol == ProtocolType.MINION:
            return self.Minion(
                local_client=local_client,
                remote_client=remote_client,
                max_rounds=config.max_rounds,
                is_multi_turn=True,  # Add standard parameters
            )
        
        elif config.protocol == ProtocolType.MINIONS:
            return self.Minions(
                local_client=local_client,
                remote_client=remote_client,
                max_rounds=config.max_rounds,
                is_multi_turn=True,  # Add standard parameters
            )
        
        else:
            raise ValueError(f"Unsupported protocol: {config.protocol}")
    
    def create_client_pair(self, config: MinionsConfig) -> Tuple[Any, Any]:
        """Create local and remote client pair."""
        
        # Create local client
        local_config = self._get_local_client_config(config)
        local_client = self.create_client(config.local_provider, local_config)
        
        # Create remote client
        remote_config = self._get_remote_client_config(config)
        remote_client = self.create_client(config.remote_provider, remote_config)
        
        return local_client, remote_client
    
    def _get_local_client_config(self, config: MinionsConfig) -> Dict[str, Any]:
        """Get local client configuration."""
        
        base_config = {
            "model_name": config.local_model,
            "temperature": config.local_temperature,
            "max_tokens": config.local_max_tokens,
        }
        
        # Add provider-specific configs
        if config.local_provider == ProviderType.OLLAMA:
            base_config.update({
                "num_ctx": config.num_ctx,
                "use_async": config.protocol == ProtocolType.MINIONS,
            })
            
            # Add structured output for Minions protocol
            if config.protocol == ProtocolType.MINIONS:
                try:
                    from pydantic import BaseModel
                    from typing import Optional
                    
                    class StructuredLocalOutput(BaseModel):
                        explanation: str
                        citation: Optional[str]
                        answer: Optional[str]
                    
                    base_config["structured_output_schema"] = StructuredLocalOutput
                except ImportError:
                    pass
        
        return base_config
    
    def _get_remote_client_config(self, config: MinionsConfig) -> Dict[str, Any]:
        """Get remote client configuration."""
        
        return {
            "model_name": config.remote_model,
            "temperature": config.remote_temperature,
            "max_tokens": config.remote_max_tokens,
        }
    



# Global factory instance
client_factory = ClientFactory() 