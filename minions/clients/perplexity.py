import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class PerplexityAIClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "sonar-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Perplexity client.

        Args:
            model_name: The name of the model to use (default: "sonar-pro")
            api_key: Perplexity API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Perplexity API (optional, falls back to PERPLEXITY_BASE_URL environment variable or default URL)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        # Client-specific configuration
        openai.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.api_key = openai.api_key
        
        # Get base URL from parameter, environment variable, or use default
        base_url = base_url or os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
        
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=base_url
        )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI  client, but route to perplexity

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # add a system prompt to the top of the messages
        messages.insert(
            0,
            {
                "role": "system",
                "content": "You are language model that has access to the internet if you need it.",
            },
        )

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                **kwargs,
            }

            params["temperature"] = self.temperature
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Sonar API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        
        # Extract cost information if available (new in July 2025)
        try:
            if hasattr(response.usage, 'cost') and response.usage.cost:
                cost = response.usage.cost
                usage.input_tokens_cost = getattr(cost, 'input_tokens_cost', None)
                usage.output_tokens_cost = getattr(cost, 'output_tokens_cost', None)
                usage.request_cost = getattr(cost, 'request_cost', None)
                usage.total_cost = getattr(cost, 'total_cost', None)
                
                # Log cost information if available
                if usage.total_cost is not None:
                    self.logger.info(f"Perplexity API cost: ${usage.total_cost:.6f} (input: ${usage.input_tokens_cost:.6f}, output: ${usage.output_tokens_cost:.6f}, request: ${usage.request_cost:.6f})")
        except AttributeError:
            # Cost information not available in this response
            pass
        
        # Extract search context size if available (Perplexity-specific)
        try:
            if hasattr(response.usage, 'search_context_size'):
                usage.search_context_size = response.usage.search_context_size
                self.logger.info(f"Search context size: {usage.search_context_size}")
        except AttributeError:
            # Search context size not available
            pass

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage

    @staticmethod
    def get_available_models():
        """
        Get a list of available models from Perplexity AI.
        
        Returns:
            List[str]: List of model names available through Perplexity API
        """
        return [
            # Search models - lightweight, cost-effective information retrieval
            "sonar",
            "sonar-pro", 
            
            # Reasoning models - complex, multi-step tasks with step-by-step thinking
            "sonar-reasoning",
            "sonar-reasoning-pro",
            
            # Research models - in-depth analysis and comprehensive reports
            "sonar-deep-research",
        ]
