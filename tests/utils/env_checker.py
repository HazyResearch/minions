"""
Environment variable checker for API keys in integration tests.
"""

import os
import warnings
from typing import Dict, List, Optional


class APIKeyChecker:
    """Utility to check for required API keys and warn if missing"""
    
    REQUIRED_KEYS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY', 
        'mistral': 'MISTRAL_API_KEY',
        'groq': 'GROQ_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'together': 'TOGETHER_API_KEY',
        'perplexity': 'PERPLEXITY_API_KEY',
        'sambanova': 'SAMBANOVA_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'huggingface': 'HUGGINGFACE_API_KEY',
        'grok': 'GROK_API_KEY',
        'sarvam': 'SARVAM_API_KEY',
        'azure': 'AZURE_OPENAI_API_KEY',
        'llama_api': 'LLAMA_API_KEY',
        'cartesia': 'CARTESIA_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'tokasaurus': 'TOKASAURUS_API_KEY',
    }
    
    @classmethod
    def check_key(cls, service: str) -> Optional[str]:
        """Check if API key exists for service"""
        env_var = cls.REQUIRED_KEYS.get(service.lower())
        if not env_var:
            return None
        
        api_key = os.getenv(env_var)
        if not api_key:
            warnings.warn(
                f"Skipping {service} tests: {env_var} not found in environment. "
                f"Set {env_var} to run {service} integration tests.",
                UserWarning
            )
        return api_key
    
    @classmethod
    def warn_if_missing(cls, service: str) -> bool:
        """Warn if key is missing and return True if key exists"""
        return cls.check_key(service) is not None
    
    @classmethod
    def get_available_services(cls) -> List[str]:
        """Get list of services with available API keys"""
        return [
            service for service in cls.REQUIRED_KEYS.keys()
            if os.getenv(cls.REQUIRED_KEYS[service])
        ]
    
    @classmethod
    def print_status(cls):
        """Print status of all API keys"""
        print("API Key Status:")
        print("-" * 40)
        for service, env_var in cls.REQUIRED_KEYS.items():
            status = "✓" if os.getenv(env_var) else "✗"
            print(f"{status} {service:12} ({env_var})")
        print()