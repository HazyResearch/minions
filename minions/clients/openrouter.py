import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from minions.clients.openai import OpenAIClient
from minions.usage import Usage


class OpenRouterClient(OpenAIClient):
    """Client for OpenRouter API with unified access to various LLMs."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        verbosity: str = "medium",
        use_responses_api: bool = False,
        reasoning_effort: str = "low",
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize the OpenRouter client.

        Args:
            model_name: Primary model to use (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet")
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.
            base_url: API base URL. Falls back to OPENROUTER_BASE_URL env var or default.
            site_url: Site URL for openrouter.ai rankings (HTTP-Referer header).
            site_name: Site name for openrouter.ai rankings (X-Title header).
            verbosity: Response verbosity level: "low", "medium", "high".
            use_responses_api: Use responses API for reasoning models.
            reasoning_effort: Reasoning effort for reasoning models: "low", "medium", "high".
            fallback_models: List of fallback models if primary fails.
        """
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenRouter API key not provided and OPENROUTER_API_KEY not set."
                )

        if base_url is None:
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL")
        self.site_name = site_name or os.environ.get("OPENROUTER_SITE_NAME")
        self.fallback_models = fallback_models

        if verbosity not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid verbosity '{verbosity}'. Must be: low, medium, high")
        self.verbosity = verbosity

        # Auto-enable responses API for reasoning models
        self.use_responses_api = use_responses_api or any(
            x in model_name for x in ["o1", "o3", "o4"]
        )
        self.reasoning_effort = reasoning_effort

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            use_responses_api=False,
            **kwargs
        )

        # Reinitialize client with extra headers if needed
        extra_headers = self._get_extra_headers()
        if extra_headers:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=extra_headers
            )

        self.logger.info(
            f"Initialized OpenRouter client: model={model_name}, "
            f"verbosity={verbosity}, responses_api={self.use_responses_api}, "
            f"fallbacks={fallback_models}"
        )

    def _get_extra_headers(self) -> Dict[str, str]:
        """Get OpenRouter-specific headers."""
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def responses(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage]:
        """Handle chat completions using the Responses API for reasoning models."""
        assert len(messages) > 0, "Messages cannot be empty."

        if "response_format" in kwargs:
            kwargs["text"] = {"format": kwargs.pop("response_format")}

        # Convert system -> developer role for responses API
        for message in messages:
            if message["role"] == "system":
                message["role"] = "developer"

        params = {
            "model": self.model_name,
            "input": messages,
            "max_output_tokens": self.max_tokens,
            "stream": False,
            **kwargs,
        }

        if any(x in self.model_name for x in ["o1", "o3", "o4"]):
            params["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = self.client.responses.create(**params)
        except Exception as e:
            self.logger.error(f"OpenRouter Responses API error: {e}")
            raise

        # Extract output text
        outputs = []
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if item.type == "message" and hasattr(item, 'content'):
                    for part in item.content:
                        if part.type == "output_text":
                            outputs.append(part.text)

        usage = Usage(
            prompt_tokens=getattr(response.usage, 'input_tokens', 0) if response.usage else 0,
            completion_tokens=getattr(response.usage, 'output_tokens', 0) if response.usage else 0,
        )

        return outputs, usage

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """Handle chat completions using the OpenRouter API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional arguments for chat.completions.create.
                     Can include plugins for PDF processing, etc.
        """
        if self.use_responses_api:
            return self.responses(messages, **kwargs)

        assert len(messages) > 0, "Messages cannot be empty."

        params = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            **kwargs,
        }

        # Add fallback models if configured
        if self.fallback_models:
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["models"] = self.fallback_models

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {e}")
            raise

        usage = Usage(
            prompt_tokens=getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
            completion_tokens=getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
        )

        responses = []
        for choice in response.choices:
            content = choice.message.content
            if hasattr(choice.message, 'annotations'):
                responses.append({'content': content, 'annotations': choice.message.annotations})
            else:
                responses.append(content)

        return responses, usage

    def embed(
        self,
        content: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using the OpenRouter API.

        Args:
            content: Text to embed (string or list of strings).
            model: Embedding model override. Defaults to instance model_name.
            encoding_format: "float" or "base64".
        """
        if isinstance(content, str):
            content = [content]

        try:
            response = self.client.embeddings.create(
                input=content,
                model=model or self.model_name,
                encoding_format=encoding_format,
                **kwargs,
            )
            return [e.embedding for e in response.data]
        except Exception as e:
            self.logger.error(f"OpenRouter embeddings error: {e}")
            raise

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models from OpenRouter."""
        try:
            import requests

            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")

            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [m["id"] for m in response.json().get("data", [])]

        except Exception as e:
            logging.error(f"Failed to get OpenRouter models: {e}")
            return [
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/o4-mini",
                "openai/o1-preview",
                "openai/o3-mini",
                "anthropic/claude-3-5-sonnet",
                "anthropic/claude-3-5-haiku",
                "meta-llama/llama-3.1-405b-instruct",
                "google/gemini-2.0-flash",
                "mistralai/mistral-large",
            ]