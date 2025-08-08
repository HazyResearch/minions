import requests
import aiohttp
import subprocess
import shutil
import platform
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from minions.clients.base import MinionsClient  
from minions.usage import Usage

class DockerModelRunnerClient(MinionsClient):
    def __init__(self, model_name: str, port: int = 12434, timeout: int = 60, local: bool = True, base_url: str = None, structured_output_schema: Optional[BaseModel] = None, **kwargs):
        super().__init__(model_name=model_name, local=local, **kwargs)
        
        # Check if Docker is installed before proceeding
        #self._check_docker_installation()
        self.model_name = model_name
        self.port = port
        self.base_url = base_url or f"http://localhost:{port}"
        self.timeout = timeout
        
        # Structured output formatting support
        self.format_structured_output = None
        if structured_output_schema is not None:
            self.format_structured_output = structured_output_schema.model_json_schema()

        # Check if model is available and pull if needed
        #self._ensure_model_available()

    @staticmethod
    def _raise_docker_not_installed():
        """Raise an exception with Docker installation instructions."""
        raise RuntimeError(
            "Docker is not installed or not found in PATH. "
            "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/ "
            "and ensure it's running."
        )

    @staticmethod
    def _check_docker_installation():
        """Check if Docker is installed and running, provide installation instructions if not."""
        # Check if docker command is available
        if not shutil.which("docker"):
            DockerModelRunnerClient._raise_docker_not_installed()
        
        # Check if Docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                print("Docker is not installed. Please navigate to https://www.docker.com/products/docker-desktop/ to install Docker.")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print("Docker is not running. Please start Docker Desktop.")

    def _check_model_available(self) -> bool:
        """Check if the model is available in Docker Model Runner."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["id"] for model in models_data.get("data", [])]
                return self.model_name in available_models
            else:
                print(f"Failed to check available models: HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to Docker Model Runner at {self.base_url}. Is Docker Desktop running with Model Runner enabled?")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error checking available models: {e}")
            return False

    def _pull_model(self):
        """Pull the model using Docker Model Runner."""
        try:
            print(f"Pulling model {self.model_name}...")
            # Use docker run command to pull the model through Docker Model Runner
            result = subprocess.run(
                ["docker", "model",  "pull", self.model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for model pull
            )
            
            if result.returncode == 0:
                print(f"Successfully pulled model {self.model_name}")
            else:
                print(f"Failed to pull model {self.model_name}: {result.stderr}")
                raise RuntimeError(f"Model pull failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timeout while pulling model {self.model_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull model {self.model_name}: {e}")

    def _ensure_model_available(self):
        """Ensure the model is available, pull it if necessary."""
        if not self._check_model_available():
            print(f"Model {self.model_name} not found locally. Attempting to pull...")
            self._pull_model()
            
            # Verify the model is now available
            if not self._check_model_available():
                raise RuntimeError(f"Model {self.model_name} is still not available after pull attempt")
        else:
            print(f"Model {self.model_name} is available")

    def _make_chat_request(self, messages, temperature=0.0, max_tokens=None):
        """Make a chat request to the Docker Model Runner API"""
        payload = {
            "model": self.model_name, 
            "messages": messages, 
            "temperature": temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if self.format_structured_output:
            payload["json_schema"] = self.format_structured_output
        
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=30)
            if resp.ok:
                return resp.json()
            else:
                resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Docker Model Runner at {self.base_url}. Is Docker Desktop running with Model Runner enabled?")
            

    def _chat(self, messages, temperature=0.0, max_tokens=None):
        return self._make_chat_request(messages, temperature, max_tokens)
    
    def chat(self, messages, **kwargs):
        result = self._chat(messages, **kwargs)
        if "choices" in result and len(result["choices"]) > 0:
            # Extract usage information from the top level of the response
            usage_data = result.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            )
            
            # Extract message content and finish reason
            choice = result["choices"][0]
            message_content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            
            if self.local:
                return [message_content], usage, [finish_reason]
            else:
                return [message_content], usage
        else:
            raise RuntimeError(f"Unexpected response format from Docker Model Runner: {result}")
    
    async def _make_chat_request_async(self, messages, temperature=0.0, max_tokens=None):
        """Make an async chat request to the Docker Model Runner API"""
        payload = {
            "model": self.model_name, 
            "messages": messages, 
            "temperature": temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if self.format_structured_output:
            payload["json_schema"] = self.format_structured_output
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.base_url}/chat/completions", json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        resp.raise_for_status()
        except aiohttp.ClientConnectorError:
            raise RuntimeError(f"Cannot connect to Docker Model Runner at {self.base_url}. Is Docker Desktop running with Model Runner enabled?")
        
    async def _achat(self, messages, temperature=0.0, max_tokens=None):
        return await self._make_chat_request_async(messages, temperature, max_tokens)

    async def achat(self, messages, **kwargs) -> Tuple[List[str], List[Usage], List[str]]:
        """
        Handle async chat completions.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments for the chat request
            
        Returns:
            Tuple of (List[str], List[Usage], List[str]) containing response strings, usage info, and finish reasons
        """
        result = await self._achat(messages, **kwargs)
        if "choices" in result and len(result["choices"]) > 0:
            # Extract usage information from the top level of the response
            usage_data = result.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            )
            
            # Extract message content and finish reason
            choice = result["choices"][0]
            message_content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            
            # Return format matches OllamaClient.achat - note List[Usage] instead of Usage
            if self.local:
                return [message_content], [usage], [finish_reason]
            else:
                return [message_content], [usage]
        else:
            raise RuntimeError(f"Unexpected response format from Docker Model Runner: {result}")

    def __del__(self):
        # No need to clean up processes since Docker Desktop manages the service
        pass
