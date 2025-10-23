import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from minions.clients.openai import OpenAIClient

from minions.usage import Usage


class NovitaClient(OpenAIClient):
    """Client for Novita AI API, which provides access to various LLMs through OpenAI-compatible API.

    Novita AI uses the OpenAI API format, so we can inherit from OpenAIClient.
    Novita AI provides access to various AI models with competitive pricing and performance.
    
    Supports reasoning models that return both content and reasoning_content.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Novita client.

        Args:
            model_name: The model to use (e.g., "meta-llama/llama-3.1-8b-instruct", 
                       "deepseek/deepseek-r1" for reasoning models)
            api_key: Novita AI API key. If not provided, will look for NOVITA_API_KEY env var.
            temperature: Temperature parameter for generation (recommended 0.5-0.7 for reasoning models).
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the Novita API. If not provided, will look for NOVITA_BASE_URL env var or use default.
            **kwargs: Additional parameters passed to base class
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("NOVITA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Novita API key not provided and NOVITA_API_KEY environment variable not set. "
                    "Get your API key from: https://novita.ai/settings/key-management"
                )

        # Get base URL from parameter, environment variable, or use default
        if base_url is None:
            base_url = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/v3/openai")

        # Call parent constructor
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.info(f"Initialized Novita client with model: {model_name}")
        
    def is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model.
        
        Returns:
            bool: True if the model supports reasoning_content output
        """
        reasoning_model_prefixes = [
            # DeepSeek R1 Series
            "deepseek/deepseek-r1",
            # Qwen Thinking Series
            "qwen/qwen3-235b-a22b-thinking",
            "qwen/qwen3-235b-a22b-fp8",
            "qwen/qwen3-30b-a3b-fp8",
            "qwen/qwen3-32b-fp8",
            "qwen/qwen3-8b-fp8",
            "qwen/qwen3-4b-fp8",
            # GLM Series
            "thudm/glm-4.1v-9b-thinking",
            "zai-org/glm-4.5",
            # LLaMA Series
            "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        ]
        return any(self.model_name.startswith(prefix) for prefix in reasoning_model_prefixes)
    
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, Optional[List[str]]]:
        """Handle chat completions with support for reasoning models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Tuple of (outputs, usage, reasoning_outputs) where:
                - outputs: List of response strings
                - usage: Token usage information
                - reasoning_outputs: List of reasoning content (None if not a reasoning model)
        """
        assert len(messages) > 0, "Messages cannot be empty."
        
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }
            
            response = self.client.chat.completions.create(**params)
            
            # Extract usage information
            if response.usage is None:
                usage = Usage(prompt_tokens=0, completion_tokens=0)
            else:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
            
            # Extract regular content
            outputs = [choice.message.content for choice in response.choices if choice.message.content]
            
            # Extract reasoning content if available
            reasoning_outputs = None
            if self.is_reasoning_model():
                reasoning_outputs = []
                for choice in response.choices:
                    if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                        reasoning_outputs.append(choice.message.reasoning_content)
                # Only return reasoning outputs if we found any
                reasoning_outputs = reasoning_outputs if reasoning_outputs else None
            
            return outputs, usage, reasoning_outputs
                
        except Exception as e:
            self.logger.error(f"Error during Novita API call: {e}")
            raise

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available models from Novita AI.
        
        Returns:
            List[str]: List of model names available through Novita AI
        """
        try:
            import requests
            
            api_key = os.environ.get("NOVITA_API_KEY")
            if not api_key:
                raise ValueError("NOVITA_API_KEY environment variable not set")
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.novita.ai/v3/openai/models", headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
            
        except Exception as e:
            logging.error(f"Failed to get Novita model list: {e}")
            # Return some common models as fallback based on documentation
            return [
                # Standard models
                "moonshotai/kimi-k2-0905",
                "deepseek/deepseek-v3.1",
                "meta-llama/llama-3.1-8b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-405b-instruct",
                "mistralai/mistral-7b-instruct",
                "mistralai/mixtral-8x7b-instruct",
                "microsoft/wizardlm-2-8x22b",
                "google/gemma-2-9b-it",
                "qwen/qwen2.5-72b-instruct",
                # Reasoning models - DeepSeek Series
                "deepseek/deepseek-r1-0528",
                "deepseek/deepseek-r1-0528-qwen3-8b",
                "deepseek/deepseek-r1-turbo",
                "deepseek/deepseek-r1-distill-qwen-32b",
                "deepseek/deepseek-r1-distill-qwen-14b",
                "deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1-distill-llama-8b",
                # Reasoning models - Qwen Series
                "qwen/qwen3-235b-a22b-fp8",
                "qwen/qwen3-30b-a3b-fp8",
                "qwen/qwen3-32b-fp8",
                "qwen/qwen3-8b-fp8",
                "qwen/qwen3-4b-fp8",
                "qwen/qwen3-235b-a22b-thinking-2507",
                # Reasoning models - GLM Series
                "zai-org/glm-4.5",
                "thudm/glm-4.1v-9b-thinking",
                # Reasoning models - LLaMA Series
                "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            ] 