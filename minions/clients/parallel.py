import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from minions.clients.openai import OpenAIClient
from minions.usage import Usage


class ParallelClient(OpenAIClient):
    """Client for Parallel AI API, which provides low latency web research capabilities.
    
    Parallel AI uses the OpenAI API format, so we can inherit from OpenAIClient.
    The Chat API is designed for interactive workflows where speed is paramount,
    with a p50 TTFT (time to first token) of 3 seconds.
    
    Available models:
        - "speed": Optimized for lowest latency responses
        - "lite": Light task processor with basic research capabilities
        - "base": Standard task processor with balanced speed/quality
        - "core": Full task processor with comprehensive research and citations
    """

    # Available models for the Parallel Chat API
    AVAILABLE_MODELS = ["speed", "lite", "base", "core"]

    def __init__(
        self,
        model_name: str = "speed",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Parallel client.
        
        Args:
            model_name: The model to use. Options:
                - "speed": Optimized for low latency (default)
                - "lite": Light research with basic citations
                - "base": Balanced research capabilities  
                - "core": Comprehensive research with full citations
            api_key: Parallel AI API key. If not provided, will look for PARALLEL_API_KEY env var.
            temperature: Temperature parameter for generation (note: may be ignored by API).
            max_tokens: Maximum number of tokens to generate (note: may be ignored by API).
            base_url: Base URL for the Parallel API. If not provided, will look for 
                     PARALLEL_BASE_URL env var or use default.
            **kwargs: Additional parameters passed to base class
        """
        # Validate model name
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model_name '{model_name}'. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("PARALLEL_API_KEY")
        if api_key is None:
            raise ValueError(
                "Parallel API key not provided and PARALLEL_API_KEY environment variable not set. "
                "Get your API key from: https://parallel.ai"
            )

        # Get base URL from parameter, environment variable, or use default
        if base_url is None:
            base_url = os.environ.get("PARALLEL_BASE_URL", "https://api.parallel.ai")

        # Call parent constructor
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        self.logger.info(f"Initialized Parallel client with model: {model_name}")

    @staticmethod
    def get_available_models() -> List[str]:
        """Get a list of available models from Parallel AI.
        
        Returns:
            List[str]: List of model names available through Parallel AI
        """
        return ParallelClient.AVAILABLE_MODELS.copy()