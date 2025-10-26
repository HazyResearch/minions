import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os

from minions.clients.openai import OpenAIClient
from minions.usage import Usage


class VercelGatewayClient(OpenAIClient):
    """Client for Vercel AI Gateway (OpenAI-compatible API).

    The Vercel AI Gateway exposes an OpenAI-compatible `/v1` API and supports
    provider routing and configuration via `providerOptions`. Key features include:
    - Unified API for accessing hundreds of models
    - High reliability with automatic retries and fallbacks
    - Embeddings support for search and retrieval
    - Spend monitoring and budget controls

    Reference: https://vercel.com/docs/ai-gateway
    OpenAI Compatibility: https://vercel.com/docs/ai-gateway/openai-compatible-api
    """

    def __init__(
        self,
        model_name: str = "xai/grok-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize the Vercel AI Gateway client.

        Args:
            model_name: The model to use (e.g., "anthropic/claude-sonnet-4", "openai/gpt-4o-mini").
            api_key: Vercel AI Gateway API key. If not provided, uses AI_GATEWAY_API_KEY env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the Vercel AI Gateway. If not provided, uses
                VERCEL_AI_GATEWAY_BASE_URL env var or the default "https://ai-gateway.vercel.sh/v1".
            provider_options: Optional provider routing/config options to send via `providerOptions`.
            **kwargs: Additional parameters passed to base class.
        """

        # Resolve API key
        if api_key is None:
            api_key = os.environ.get("AI_GATEWAY_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Vercel AI Gateway API key not provided and AI_GATEWAY_API_KEY environment variable not set."
                )

        # Resolve base URL
        if base_url is None:
            base_url = os.environ.get("VERCEL_AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")

        self.provider_options = provider_options

        # Initialize via OpenAI-compatible client
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized Vercel AI Gateway client with model: {model_name}")
        self.logger.info(f"Using gateway URL: {base_url}")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """Handle chat completions via the Vercel AI Gateway.

        Supports sending `providerOptions` using the OpenAI SDK `extra_body` parameter.
        You can pass `provider_options` (snake_case) or `providerOptions` (camelCase)
        in kwargs to override the instance-level provider options.
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # Pull provider options from kwargs if provided
        provider_options = (
            kwargs.pop("provider_options", None)
            or kwargs.pop("providerOptions", None)
            or self.provider_options
        )

        # Build params consistent with OpenAIClient
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": getattr(self, "max_tokens", None),
        }

        # Only apply temperature for non-reasoning models
        if "o1" not in self.model_name and "o3" not in self.model_name:
            params["temperature"] = getattr(self, "temperature", None)
        else:
            # Reasoning models: pass through effort if provided
            if "reasoning_effort" in kwargs:
                params["reasoning_effort"] = kwargs.pop("reasoning_effort")

        # Merge remaining kwargs (e.g., tools, tool_choice, response_format, stream)
        params.update(kwargs)

        # Attach providerOptions via extra_body if present
        if provider_options is not None:
            # If caller already provided extra_body, merge into it
            extra_body: Dict[str, Any] = params.pop("extra_body", {}) or {}
            extra_body["providerOptions"] = provider_options
            params["extra_body"] = extra_body

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Vercel AI Gateway API call: {e}")
            raise

        # Extract usage
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return [choice.message.content for choice in response.choices], usage

    def embed(
        self,
        content: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Vercel AI Gateway.

        Args:
            content: Text content to embed (single string or list of strings).
            model: Embedding model to use (e.g., "openai/text-embedding-3-small").
                   If not provided, uses the instance model_name.
            encoding_format: Format of embeddings ("float" or "base64", default: "float").
            **kwargs: Additional parameters for the embeddings API.

        Returns:
            List of embedding vectors.
        """
        try:
            # Ensure content is a list
            if isinstance(content, str):
                content = [content]

            # Use provided model or fall back to instance model
            embedding_model = model or self.model_name

            response = self.client.embeddings.create(
                input=content,
                model=embedding_model,
                encoding_format=encoding_format,
                **kwargs
            )

            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            self.logger.error(f"Error generating embeddings via Vercel AI Gateway: {e}")
            raise

    def list_models(self) -> Dict[str, Any]:
        """List available models from the Vercel AI Gateway.

        Returns a list of all models available through the gateway, including
        models from all supported providers (OpenAI, Anthropic, xAI, etc.).

        Returns:
            Dict containing the models data with structure:
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "provider/model-name",
                            "object": "model",
                            "created": timestamp,
                            "owned_by": "provider"
                        },
                        ...
                    ]
                }

        Reference: https://vercel.com/docs/ai-gateway/openai-compat#list-models
        """
        try:
            response = self.client.models.list()
            return {
                "object": "list",
                "data": [model.model_dump() for model in response.data]
            }
        except Exception as e:
            self.logger.error(f"Error listing models via Vercel AI Gateway: {e}")
            raise

    def retrieve_model(self, model_id: str) -> Dict[str, Any]:
        """Retrieve information about a specific model from the Vercel AI Gateway.

        Args:
            model_id: The model identifier (e.g., "anthropic/claude-sonnet-4", "openai/gpt-4o").

        Returns:
            Dict containing the model information:
                {
                    "id": "provider/model-name",
                    "object": "model",
                    "created": timestamp,
                    "owned_by": "provider"
                }

        Reference: https://vercel.com/docs/ai-gateway/openai-compat#retrieve-model
        """
        try:
            response = self.client.models.retrieve(model_id)
            return response.model_dump()
        except Exception as e:
            self.logger.error(f"Error retrieving model '{model_id}' via Vercel AI Gateway: {e}")
            raise


