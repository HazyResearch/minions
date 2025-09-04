import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from openai import OpenAI

from minions.usage import Usage
from minions.clients.base import MinionsClient


class OsaurusClient(MinionsClient):
    """
    Client for Osaurus - Native, Apple Silicon-only local LLM server.
    
    Osaurus is similar to Ollama but built on Apple's MLX for maximum performance 
    on M-series chips. It provides OpenAI-compatible endpoints.
    
    This client uses the OpenAI SDK to communicate with the Osaurus server.

    Instructions for installing and running Osaurus: https://github.com/dinoki-ai/osaurus?tab=readme-ov-file#download
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.2-3b-instruct-4bit",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        port: int = 8080,
        local: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """
        Initialize the Osaurus client.
        
        Args:
            model_name: The Osaurus model to use (default: "llama-3.2-3b-instruct-4bit")
            api_key: API key (can be any placeholder for Osaurus, default: "osaurus")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            base_url: Base URL for the Osaurus server (default: http://127.0.0.1:{port}/v1)
            port: Port where Osaurus server is running (default: 8080)
            local: Whether this is a local client (default: True)
            tools: List of tools for function calling (default: None)
            tool_choice: Tool choice strategy - "auto", "none", or specific tool (default: "auto")
            **kwargs: Additional parameters passed to base class
        """
        # Set default base_url if not provided
        if base_url is None:
            base_url = f"http://127.0.0.1:{port}/v1"
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or "osaurus"  # Osaurus accepts any placeholder API key
        self.base_url = base_url
        self.port = port
        self.tools = tools
        self.tool_choice = tool_choice
        
        # Initialize the OpenAI client pointing to Osaurus
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Check if the Osaurus server is running
        if self.local:
            try:
                self.check_server_health()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Osaurus server at {self.base_url} is not running or reachable. "
                    f"Please start the Osaurus app and ensure the server is running on port {self.port}. "
                    f"Error: {e}"
                )
    
    @staticmethod
    def get_available_models():
        """
        Get a list of available Osaurus models.
        
        Note: This is a static method that returns common Osaurus model names.
        For dynamic model listing, use the list_models() instance method.
        
        Returns:
            List[str]: List of common model names
        """
        return [
            "llama-3.2-3b-instruct-4bit",
            "llama-3.2-1b-instruct-4bit", 
            "qwen2.5-3b-instruct-4bit",
            "gemma-2-2b-instruct-4bit",
            "phi-3.5-mini-instruct-4bit",
            "mistral-7b-instruct-v0.3-4bit",
            "deepseek-coder-6.7b-instruct-4bit"
        ]
    
    def check_server_health(self):
        """
        Check if the Osaurus server is running and reachable.
        
        Returns:
            Dict: Server response (typically model list)
            
        Raises:
            requests.exceptions.RequestException: If server is not reachable
        """
        try:
            # Try the models endpoint to check server health
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            # Fallback to base URL without /v1 prefix
            base_url_no_v1 = self.base_url.replace("/v1", "")
            resp = requests.get(f"{base_url_no_v1}/models", timeout=5)
            resp.raise_for_status()
            return resp.json()
    
    def list_models(self):
        """
        List available models from the Osaurus server.
        
        Returns:
            Dict: Models data from the Osaurus API response
        """
        try:
            response = self.client.models.list()
            return {
                "object": "list", 
                "data": [model.model_dump() for model in response.data]
            }
        except Exception as e:
            self.logger.error(f"Error listing Osaurus models: {e}")
            raise
    
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Osaurus server via OpenAI SDK.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys,
                     or a single message dictionary
            **kwargs: Additional arguments to pass to the chat completion
            
        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, 
            token usage, and finish reasons
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]
            
        assert len(messages) > 0, "Messages cannot be empty."
        
        try:
            # Prepare parameters for the API call
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs
            }
            
            # Add tools if provided
            if self.tools:
                params["tools"] = self.tools
                params["tool_choice"] = self.tool_choice
            
            # Make the API call
            response = self.client.chat.completions.create(**params)
            
            # Extract response content
            response_texts = []
            finish_reasons = []
            
            for choice in response.choices:
                response_texts.append(choice.message.content or "")
                finish_reasons.append(choice.finish_reason or "stop")
            
            # Extract usage information
            if response.usage is None:
                usage = Usage(prompt_tokens=0, completion_tokens=0)
            else:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
            
            # Handle tool calls if present
            if (response.choices and 
                response.choices[0].message.tool_calls and 
                self.tools):
                
                # For tool calls, we return the assistant message content
                # The tool_calls are available in response.choices[0].message.tool_calls
                # This follows OpenAI's pattern where the client handles tool execution
                pass
            
            if self.local:
                return response_texts, usage, finish_reasons
            else:
                return response_texts, usage
                
        except Exception as e:
            self.logger.error(f"Error during Osaurus API call: {e}")
            raise
    
    def stream_chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]], 
        **kwargs
    ):
        """
        Handle streaming chat completions using the Osaurus server.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys,
                     or a single message dictionary
            **kwargs: Additional arguments to pass to the chat completion
            
        Yields:
            Chat completion chunks from the streaming response
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]
            
        assert len(messages) > 0, "Messages cannot be empty."
        
        try:
            # Prepare parameters for the streaming API call
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                **kwargs
            }
            
            # Add tools if provided
            if self.tools:
                params["tools"] = self.tools
                params["tool_choice"] = self.tool_choice
            
            # Make the streaming API call
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error during Osaurus streaming API call: {e}")
            raise
    
    def embed(
        self,
        content: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using the Osaurus server.
        
        Note: Embedding support depends on the specific model loaded in Osaurus.
        
        Args:
            content: Text content to embed (single string or list of strings)
            **kwargs: Additional parameters for the embedding call
            
        Returns:
            List of embedding vectors
            
        Raises:
            NotImplementedError: If embeddings are not supported by the current model
        """
        try:
            if isinstance(content, str):
                content = [content]
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=content,
                **kwargs
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            self.logger.error(f"Error during Osaurus embedding call: {e}")
            # If embeddings are not supported, raise NotImplementedError
            if "not found" in str(e).lower() or "not supported" in str(e).lower():
                raise NotImplementedError(f"Embedding not supported by {self.model_name} in Osaurus")
            raise
