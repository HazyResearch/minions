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
        reasoning_enabled: bool = False,
        reasoning_max_tokens: Optional[int] = None,
        fallback_models: Optional[List[str]] = None,
        variant: Optional[str] = None,
        response_healing: bool = False,
        auto_router: bool = False,
        auto_router_models: Optional[List[str]] = None,
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
            verbosity: Response verbosity level: "low", "medium", "high", or "max" (Opus 4.6 only).
                Controls response detail via output_config.effort.
            use_responses_api: Use responses API for reasoning models.
            reasoning_effort: Reasoning effort for OpenAI reasoning models (o1/o3/o4): "low", "medium", "high".
                NOTE: Ignored for Claude 4.6 Opus which uses adaptive thinking.
            reasoning_enabled: Enable reasoning/thinking for Claude models (default: False).
                For Claude 4.6 Opus, this enables adaptive thinking by default.
            reasoning_max_tokens: Max tokens for reasoning budget (optional).
                If set, uses budget-based thinking instead of adaptive for Claude 4.6.
            fallback_models: List of fallback models if primary fails.
            variant: Routing preference ("nitro", "online", etc.) Appends suffix to model_name.
            response_healing: Whether to enable OpenRouter's JSON repair feature by default.
            auto_router: Use OpenRouter's auto-router to select the best model per request.
                Sets model to "openrouter/auto". Use auto_router_models to restrict selection.
            auto_router_models: Glob patterns restricting which models the auto-router can pick.
                e.g. ["anthropic/*", "openai/gpt-5*"]. Implies auto_router=True.

        Note:
            For Claude 4.6 Opus, adaptive thinking is used by default when reasoning_enabled=True.
            To use budget-based thinking, explicitly set reasoning_max_tokens.
            The "max" verbosity level is only supported on Claude 4.6 Opus.
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
        # CHANGE: Store default preference
        self.response_healing = response_healing

        # Auto-router: auto_router_models implies auto_router=True
        if auto_router_models:
            auto_router = True
        self.auto_router = auto_router
        self.auto_router_models = auto_router_models
        if auto_router:
            model_name = "openrouter/auto"

        # Check if model is Claude 4.6 Opus
        self._is_claude_46_opus = "claude-4.6-opus" in model_name or "claude-4-6-opus" in model_name
        
        # Validate verbosity - "max" only supported on Claude 4.6 Opus
        valid_verbosity = ["low", "medium", "high", "max"] if self._is_claude_46_opus else ["low", "medium", "high"]
        if verbosity not in valid_verbosity:
            if verbosity == "max" and not self._is_claude_46_opus:
                raise ValueError(f"Verbosity 'max' is only supported on Claude 4.6 Opus. Use 'high' instead.")
            raise ValueError(f"Invalid verbosity '{verbosity}'. Must be: {', '.join(valid_verbosity)}")
        self.verbosity = verbosity
        
        # Store reasoning parameters for Claude models
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_max_tokens = reasoning_max_tokens

        # Handle model variant preference (nitro/free/exacto/extended/thinking)
        if variant:
            if variant not in ["nitro", "free", "exacto", "extended", "thinking", "online"]:
                raise ValueError("Variant must be either 'nitro', 'free', 'exacto', 'extended', 'thinking', or 'online'")
            
            # Append suffix if not already present
            suffix = f":{variant}"
            if not model_name.endswith(suffix):
                model_name = f"{model_name}{suffix}"

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
            f"fallbacks={fallback_models}, healing={response_healing}, "
            f"auto_router={auto_router}, auto_router_models={auto_router_models}"
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
            **kwargs: Additional arguments. 'response_healing' (bool) can be passed here.
        """
        if self.use_responses_api:
            return self.responses(messages, **kwargs)

        assert len(messages) > 0, "Messages cannot be empty."

        # CHANGE: Check for response healing override or default
        enable_healing = kwargs.pop("response_healing", self.response_healing)


        params = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            **kwargs,
        }

        # CHANGE: Apply Response Healing Logic
        if enable_healing:
            # 1. Ensure stream is False (healing incompatible with streaming)
            params["stream"] = False
            
            # 2. Ensure JSON mode is requested if not already present
            if "response_format" not in params:
                params["response_format"] = {"type": "json_object"}
            
            # 3. Add the healing plugin via extra_body
            extra_body = params.get("extra_body", {})
            plugins = extra_body.get("plugins", [])
            
            # Avoid adding duplicate plugin entries
            if not any(p.get("id") == "response-healing" for p in plugins):
                plugins.append({"id": "response-healing"})
            
            extra_body["plugins"] = plugins
            params["extra_body"] = extra_body

        # Add auto-router plugin with allowed models
        if self.auto_router and self.auto_router_models:
            extra_body = params.get("extra_body", {})
            plugins = extra_body.get("plugins", [])
            if not any(p.get("id") == "auto-router" for p in plugins):
                plugins.append({"id": "auto-router", "allowed_models": self.auto_router_models})
            extra_body["plugins"] = plugins
            params["extra_body"] = extra_body

        # Add fallback models if configured (and not already in extra_body)
        if self.fallback_models:
            params["extra_body"] = params.get("extra_body", {})
            if "models" not in params["extra_body"]:
                params["extra_body"]["models"] = self.fallback_models

        # Add reasoning support for Claude models (especially 4.6 Opus)
        if self.reasoning_enabled and "anthropic" in self.model_name.lower():
            params["extra_body"] = params.get("extra_body", {})
            
            if self.reasoning_max_tokens is not None:
                # Budget-based thinking (explicit token limit)
                params["extra_body"]["reasoning"] = {
                    "enabled": True,
                    "max_tokens": self.reasoning_max_tokens
                }
            else:
                # Adaptive thinking (recommended for Claude 4.6 Opus)
                params["extra_body"]["reasoning"] = {
                    "enabled": True
                }

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
                responses.append({'message': content, 'annotations': choice.message.annotations})
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
        """Generate embeddings using the OpenRouter API."""
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
                "anthropic/claude-4.6-opus",
                "anthropic/claude-4.5-opus",
                "anthropic/claude-4.5-sonnet",
                "anthropic/claude-3-5-haiku",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/o4-mini",
                "openai/o1-preview",
                "openai/o3-mini",
                "meta-llama/llama-3.1-405b-instruct",
                "google/gemini-2.0-flash",
                "mistralai/mistral-large",
            ]