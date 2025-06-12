import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from openai import OpenAI

from minions.usage import Usage
from minions.clients.base import MinionsClient
from minions.clients.utils import ServerMixin


class ModularClient(MinionsClient, ServerMixin):
    """
    Client for Modular MAX using the OpenAI-compatible endpoint.
    
    This client starts a local Modular MAX server and communicates with it
    using the OpenAI-compatible API.
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        weights_path: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/smollm2-1.7b-instruct-q4_k_m.gguf",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        port: Optional[int] = None,
        capture_output: bool = False,
        base_url: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the Modular client.

        Args:
            model_name: Name of the model to use (default: "modularai/Llama-3.1-8B-Instruct-GGUF")
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            port: Port to use for the server (optional, will find a free port if not provided)
            capture_output: Whether to capture server output (default: False)
            base_url: Base URL for the Modular MAX API (optional, falls back to MODULAR_BASE_URL environment variable)
            verbose: Enable verbose logging (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            verbose=verbose,
            **kwargs
        )
        
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.weights_path = weights_path
        
        # Handle server setup
        if port is None:
            self.port = self.find_free_port()
            launch_command = f"max serve --model-path={model_name} --port={self.port} --weight-path={weights_path} --engine max"
            try:
                self.launch_server(launch_command, self.port, capture_output=capture_output)
            except Exception as e:
                raise ImportError(
                    f"Failed to start Modular MAX server. Make sure MAX is installed: {e}\n"
                    "Installation instructions: https://docs.modular.com/max/packages"
                    "please checkout the list of available models: https://builds.modular.com/?category=models"
                ) from e
        else:
            self.port = port
        
        # Get base URL from parameter, environment variable, or construct default from port
        default_base_url = f"http://0.0.0.0:{self.port}/v1"
        base_url = base_url or os.getenv("MODULAR_BASE_URL", default_base_url)
        
        # Initialize OpenAI client for the endpoint
        self.client = OpenAI(
            api_key="EMPTY",  # MAX doesn't require authentication
            base_url=base_url
        )
    
    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Modular MAX OpenAI-compatible endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the OpenAI chat completion

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        if not messages:
            raise ValueError("Messages cannot be empty.")

        try:
            # Prepare parameters for the OpenAI client
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs
            }
            
            self.logger.info(f"Sending chat request to Modular MAX server")
            
            response = self.client.chat.completions.create(**params)
            
            # Extract usage information
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )

            # Extract response content
            return [choice.message.content for choice in response.choices], usage, [choice.finish_reason for choice in response.choices]
            
        except Exception as e:
            self.logger.error(f"Error during Modular MAX API call: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of the client."""
        return f"ModularClient(model={self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the client."""
        attrs = [f"model_name='{self.model_name}'"]
        if hasattr(self, 'temperature'):
            attrs.append(f"temperature={self.temperature}")
        if hasattr(self, 'max_tokens'):
            attrs.append(f"max_tokens={self.max_tokens}")
        return f"ModularClient({', '.join(attrs)})" 