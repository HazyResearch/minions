import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import requests

from minions.usage import Usage
from minions.clients.base import MinionsClient


class SarvamClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "sarvam-m",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        n: int = 1,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Sarvam AI client.

        Args:
            model_name: The name of the model to use (default: "sarvam-m")
            api_key: Sarvam API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: 100)
            top_p: Top-p sampling parameter (default: 1.0)
            n: Number of completions to generate (default: 1)
            base_url: Base URL for the Sarvam API (optional, falls back to SARVAM_BASE_URL environment variable or default URL)
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
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        self.top_p = top_p
        self.n = n
        self.base_url = base_url or os.getenv(
            "SARVAM_BASE_URL", "https://api.sarvam.ai/v1"
        )

        if not self.api_key:
            raise ValueError(
                "Sarvam API key is required. Set SARVAM_API_KEY environment variable or pass api_key parameter."
            )

        # Set up headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Sarvam AI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the API

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Prepare the payload for the API request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "n": self.n,
                **kwargs,
            }

            # Make the API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,  # 60 second timeout
            )

            # Check if the request was successful
            if response.status_code != 200:
                self.logger.error(f"Sarvam API request failed: {response.status_code} - {response.text}")
                raise Exception(f"Sarvam API request failed: {response.status_code} - {response.text}")

            response_data = response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during Sarvam API request: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during Sarvam API call: {e}")
            raise

        # Extract usage information
        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0)
        )

        # Extract response content and done reasons
        choices = response_data.get("choices", [])
        response_texts = []
        done_reasons = []

        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            response_texts.append(content)
            
            finish_reason = choice.get("finish_reason", "stop")
            done_reasons.append(finish_reason)

        return response_texts, usage

    def get_chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Low-level method to get the raw chat completion response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Raw response dictionary from the API or None if failed
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "n": self.n,
                **kwargs,
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Request failed: {response.status_code} {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error making request: {e}")
            return None 