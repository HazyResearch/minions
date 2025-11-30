import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class NousResearchClient(MinionsClient):
    """
    Client for NousResearch's Inference API.
    
    NousResearch provides OpenAI-compatible API endpoints for their advanced LLMs.
    Supported models include:
    - Hermes-3-Llama-70B: A powerful 70B parameter model
    - DeepHermes-3-8B-Preview: A smaller, efficient 8B parameter model
    
    Features:
    - OpenAI-compatible API (chat completions)
    - $5.00 free credits for new accounts
    - High-quality instruction-following models
    
    See: https://portal.nousresearch.com/api-docs
    """
    
    def __init__(
        self,
        model_name: str = "Hermes-3-Llama-70B",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: str = "https://portal.nousresearch.com/v1",
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the NousResearch client.

        Args:
            model_name: The model to use (default: "Hermes-3-Llama-70B")
                       Available models:
                       - "Hermes-3-Llama-70B": Powerful 70B instruction-following model
                       - "DeepHermes-3-8B-Preview": Efficient 8B preview model
            api_key: NousResearch API key (optional, falls back to NOUSRESEARCH_API_KEY environment variable)
            temperature: Sampling temperature (default: 0.7, range 0-2)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the NousResearch API (default: "https://portal.nousresearch.com/v1")
            tools: List of tools for function calling (default: None)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=False,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("NOUSRESEARCH_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "NousResearch API key must be provided either as 'api_key' parameter "
                "or via NOUSRESEARCH_API_KEY environment variable. "
                "Get your API key at https://portal.nousresearch.com"
            )
        
        self.base_url = base_url

        # Initialize the OpenAI client with NousResearch configuration
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.tools = tools

    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the NousResearch API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat completions API.
                     Supported kwargs include:
                     - response_format: For structured JSON outputs
                     - tools: List of tool definitions for function calling
                     - tool_choice: How to handle tool selection ("auto", "required", etc.)
                     - stream: Boolean to enable streaming responses

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }
            
            # Add tools if provided either in init or kwargs
            if self.tools and "tools" not in kwargs:
                params["tools"] = self.tools

            response = self.client.chat.completions.create(**params)
            
        except Exception as e:
            self.logger.error(f"Error during NousResearch API call: {e}")
            raise

        # Extract usage information if it exists
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        # Extract response content
        outputs = [choice.message.content for choice in response.choices]
        
        return outputs, usage

    def list_models(self):
        """
        List available models from the NousResearch API.
        
        Returns:
            Dict containing the models data from the NousResearch API response
        """
        try:
            response = self.client.models.list()
            return {
                "object": "list",
                "data": [model.model_dump() for model in response.data]
            }
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise

