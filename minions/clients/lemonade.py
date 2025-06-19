import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from minions.clients.openai import OpenAIClient
from minions.usage import Usage


class LemonadeClient(OpenAIClient):
    """Client for interacting with a local Lemonade inference server.

    This client uses the OpenAI compatible endpoints exposed by Lemonade
    Server. Additional Lemonade specific endpoints such as ``pull`` and
    ``load`` are also implemented via simple HTTP requests.
    """

    def __init__(
        self,
        model_name: str = "Llama-3.2-3B-Instruct-Hybrid",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        base_url = base_url or os.getenv("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs,
        )
        # Session for the custom endpoints
        self.session = requests.Session()
        self.base_url = base_url
        self.logger.setLevel(logging.INFO)

        # Validate server connection and model availability
        self._ensure_model_available()

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using direct HTTP requests to the lemonade service.
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Prepare the request payload for lemonade's OpenAI-compatible endpoint
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Make direct HTTP request to lemonade's chat completions endpoint
            response = self.session.post(
                f"{self.base_url.rstrip('/api/v1')}/api/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            self.logger.error(f"Error during Lemonade API call: {e}")
            raise

        # Extract responses from the lemonade response
        choices = response_data.get("choices", [])
        responses = [choice["message"]["content"] for choice in choices if "message" in choice]


        usage = Usage()

        usage += Usage(
            prompt_tokens=response_data.get('usage', 0)['prompt_tokens'],
            completion_tokens=response_data.get('usage', 0)['completion_tokens'],
        )

        done_reason = [choice.get("finish_reason", "stop") for choice in choices]

        return responses, usage, done_reason

    # ------------------------------------------------------------------
    # Lemonade specific helper APIs
    # ------------------------------------------------------------------
    def get_models(self) -> Dict[str, Any]:
        """Return models available on the server."""
        resp = self.session.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()

    def get_available_models(self) -> List[str]:
        """Return a list of model names available on the server."""
        models = self.get_models().get("data", [])
        return [model["id"] for model in models]
    
    def _ensure_model_available(self):
        """Ensure the specified model is available on the Lemonade server."""

        # Catch any connection issues when fetching available models
        # as that typically means the Lemonade server is not running.
        try:
            available_models = self.get_available_models()
        except requests.RequestException as e:
            msg = (f"Failed to fetch available models from Lemonade server."
                   f"Check if the Lemonade server is running")
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.model_name not in available_models:
            self.logger.info("Pulling model: %s", self.model_name)
            try:
                self.pull_model(self.model_name)
            except:
                msg = (f"Model '{self.model_name}' not found on Lemonade server and unable to pull.\n"
                    f"Available models: {available_models}")
                self.logger.error(msg)
                raise RuntimeError(msg)

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download and register a model on the server."""
        resp = self.session.post(f"{self.base_url}/pull", json={"model_name": model_name})
        resp.raise_for_status()
        return resp.json()

    def load_model(
        self,
        *,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        recipe: Optional[str] = None,
        reasoning: Optional[bool] = None,
        mmproj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly load a model into memory."""
        payload: Dict[str, Any] = {}
        if model_name:
            payload["model_name"] = model_name
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if recipe:
            payload["recipe"] = recipe
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if mmproj:
            payload["mmproj"] = mmproj
        resp = self.session.post(f"{self.base_url}/load", json=payload)
        resp.raise_for_status()
        return resp.json()

    def unload_model(self) -> Dict[str, Any]:
        """Unload the currently loaded model."""
        resp = self.session.post(f"{self.base_url}/unload")
        resp.raise_for_status()
        return resp.json()

    def set_params(self, **params: Any) -> Dict[str, Any]:
        """Set generation parameters that persist across requests."""
        resp = self.session.post(f"{self.base_url}/params", json=params)
        resp.raise_for_status()
        return resp.json()

    def get_health(self) -> Dict[str, Any]:
        """Check the health of the server."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict[str, Any]:
        """Return performance statistics from the last request."""
        resp = self.session.get(f"{self.base_url}/stats")
        resp.raise_for_status()
        return resp.json()