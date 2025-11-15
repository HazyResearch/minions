import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class BasetenClient(MinionsClient):
    """
    Client for Baseten's OpenAI-compatible Model APIs.
    
    Baseten provides OpenAI-compatible API endpoints for various open-source LLMs.
    Supported models include:
    - deepseek-ai/DeepSeek-V3.1
    - deepseek-ai/DeepSeek-R1-0528
    - deepseek-ai/DeepSeek-V3-0324
    - Qwen/Qwen3-235B-A22B-Instruct-2507
    - Qwen/Qwen3-Coder-480B-A35B-Instruct
    - moonshotai/Kimi-K2-Instruct-0905
    - openai/gpt-oss-120b
    - zai-org/GLM-4.6
    
    See: https://docs.baseten.co/development/model-apis/overview
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3-0324",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        base_url: str = "https://inference.baseten.co/v1",
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Baseten client.

        Args:
            model_name: The model slug to use (default: "deepseek-ai/DeepSeek-V3-0324")
            api_key: Baseten API key (optional, falls back to BASETEN_API_KEY environment variable)
            temperature: Sampling temperature (default: 1.0, range 0-2)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Baseten API (default: "https://inference.baseten.co/v1")
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
        self.api_key = api_key or os.getenv("BASETEN_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Baseten API key must be provided either as 'api_key' parameter "
                "or via BASETEN_API_KEY environment variable"
            )
        
        self.base_url = base_url

        # Initialize the OpenAI client with Baseten configuration
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.tools = tools
        
        # Log warning if using DeepSeek R1 with tools (not recommended per docs)
        if tools and "DeepSeek-R1" in self.model_name:
            self.logger.warning(
                "DeepSeek R1 is not recommended for tool calling as it was not "
                "post-trained for this functionality. Consider using DeepSeek V3 instead."
            )

    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Baseten API.

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
            self.logger.error(f"Error during Baseten API call: {e}")
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
        List available models from the Baseten API.
        
        Returns:
            Dict containing the models data from the Baseten API response
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

