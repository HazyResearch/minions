"""
Distributed Inference client for Minions.
Supports both direct node access and network coordinator routing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from urllib.parse import quote

from minions.usage import Usage
from minions.clients.base import MinionsClient


class DistributedInferenceClient(MinionsClient):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "http://localhost:8080",
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize the Distributed Inference client.

        Args:
            model_name: Preferred model name (optional, coordinator will select if not specified)
            api_key: API key for network coordinator authentication (optional)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Network coordinator URL (default: "http://localhost:8080")
            timeout: Request timeout in seconds (default: 30)
            **kwargs: Additional parameters passed to base class
        """
        # For distributed inference, model_name is optional
        super().__init__(
            model_name=model_name or "auto",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("MINIONS_API_KEY")
        self.timeout = timeout
        
        # Set up headers for authenticated requests
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        try:
            kwargs.setdefault("timeout", self.timeout)
            kwargs.setdefault("headers", {}).update(self.headers)
            
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timed out after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def get_network_status(self) -> Dict[str, Any]:
        """Get network health status from coordinator."""
        url = f"{self.base_url}/health"
        response = self._make_request("GET", url)
        return response.json()

    def list_nodes(self) -> Dict[str, Any]:
        """List all nodes and their capabilities."""
        url = f"{self.base_url}/nodes"
        response = self._make_request("GET", url)
        return response.json()

    def register_node(self, node_url: str) -> Dict[str, Any]:
        """Register a new node with the network coordinator."""
        url = f"{self.base_url}/nodes"
        data = {"node_url": node_url}
        response = self._make_request("POST", url, json=data)
        return response.json()

    def remove_node(self, node_url: str) -> Dict[str, Any]:
        """Remove a node from the network."""
        encoded_url = quote(node_url, safe='')
        url = f"{self.base_url}/nodes/{encoded_url}"
        response = self._make_request("DELETE", url)
        return response.json()

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Distributed Inference API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments (not used by this API)

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        assert len(messages) > 0, "Messages cannot be empty."
        
        # For distributed inference API, we need to combine all messages into a single query
        # since the API only accepts a simple query parameter
        if len(messages) == 1 and messages[0].get("role") == "user":
            # Single user message - use it directly (this preserves full Minions formatting)
            query = messages[0].get("content", "")
        else:
            # Multiple messages - combine them meaningfully
            query_parts = []
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "system":
                    query_parts.append(f"System: {content}")
                elif role == "user":
                    query_parts.append(f"User: {content}")
                elif role == "assistant":
                    query_parts.append(f"Assistant: {content}")
                else:
                    query_parts.append(content)
            query = "\n\n".join(query_parts)
        
        try:
            # Use network coordinator
            url = f"{self.base_url}/chat"
            params = {"query": query}
            
            # Add model preference if specified and not "auto"
            if self.model_name and self.model_name != "auto":
                params["model"] = self.model_name
            
            response = self._make_request("POST", url, params=params)
            data = response.json()
            
            # Log which node was used
            if "node_url" in data:
                self.logger.info(f"Request routed to node: {data['node_url']}")
            if "model_used" in data:
                self.logger.info(f"Model used: {data['model_used']}")
            
            # Extract response and usage
            response_text = data.get("response", "")
            
            # Extract usage information
            usage_data = data.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0)
            )
            
            # Extract done reason if available, default to "stop"
            done_reason = data.get("done_reason", "stop")
            
            return [response_text], usage, [done_reason]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.error("No nodes available with requested model")
            elif e.response.status_code == 503:
                self.logger.error("No healthy nodes available")
            elif e.response.status_code == 504:
                self.logger.error("Request to node timed out")
            elif e.response.status_code == 401:
                self.logger.error("Authentication failed - check API key")
            raise
        except Exception as e:
            self.logger.error(f"Error during Distributed Inference API call: {e}")
            raise

    def embed(self, content: Any, **kwargs) -> List[List[float]]:
        """Embedding not supported by Distributed Inference API."""
        raise NotImplementedError("Embedding not supported by Distributed Inference API")

    def complete(self, prompts: Any, **kwargs) -> Tuple[List[str], Usage]:
        """
        Text completion using the chat endpoint.
        
        Args:
            prompts: Single prompt or list of prompts
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (List[str], Usage) containing completions and token usage
        """
        # Convert prompts to messages format
        if isinstance(prompts, str):
            prompts = [prompts]
        
        responses = []
        total_usage = Usage()
        
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            response, usage, _ = self.chat(messages, **kwargs)  # Ignore done_reasons
            responses.extend(response)
            total_usage += usage
        
        return responses, total_usage 