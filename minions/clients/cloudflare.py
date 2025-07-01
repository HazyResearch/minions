import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from openai import OpenAI
from minions.clients.openai import OpenAIClient

from minions.usage import Usage


class CloudflareGatewayClient(OpenAIClient):
    """Client for Cloudflare AI Gateway.

    Cloudflare AI Gateway uses the OpenAI API format, so we can inherit from OpenAIClient.
    Provides access to OpenAI models through Cloudflare's AI Gateway with enhanced
    observability, caching, rate limiting, and other features.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        gateway_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Cloudflare AI Gateway client.

        Args:
            model_name: The OpenAI model to use (default: "gpt-4o-mini")
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var.
            account_id: Cloudflare account ID. If not provided, will look for CLOUDFLARE_ACCOUNT_ID env var.
            gateway_id: Cloudflare AI Gateway ID. If not provided, will look for CLOUDFLARE_GATEWAY_ID env var.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            base_url: Base URL for the Cloudflare AI Gateway. If provided, overrides account_id/gateway_id.
            **kwargs: Additional parameters passed to base class
        """
        # Get API key from environment if not provided (this is still the OpenAI API key)
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set. "
                    "Cloudflare AI Gateway requires your OpenAI API key to proxy requests."
                )

        # Determine base URL
        if base_url is None:
            # Get account ID and gateway ID from environment if not provided
            if account_id is None:
                account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
                if account_id is None:
                    raise ValueError(
                        "Cloudflare account ID not provided and CLOUDFLARE_ACCOUNT_ID environment variable not set."
                    )

            if gateway_id is None:
                gateway_id = os.environ.get("CLOUDFLARE_GATEWAY_ID")
                if gateway_id is None:
                    raise ValueError(
                        "Cloudflare gateway ID not provided and CLOUDFLARE_GATEWAY_ID environment variable not set."
                    )

            # Construct the Cloudflare AI Gateway URL
            base_url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat"

        # Store Cloudflare-specific parameters
        self.account_id = account_id
        self.gateway_id = gateway_id

        # Call parent constructor
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )

        self.logger.info(f"Initialized Cloudflare AI Gateway client with model: {model_name}")
        self.logger.info(f"Using gateway URL: {base_url}")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Cloudflare AI Gateway.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Only add temperature if NOT using reasoning models (e.g., o1, o3 models)
            if "o1" in self.model_name or "o3" in self.model_name:
                params.pop("temperature", None)

            response = self.client.chat.completions.create(**params)
            
        except Exception as e:
            self.logger.error(f"Error during Cloudflare AI Gateway API call: {e}")
            raise

        # Extract usage information
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        # Return response content
        return [choice.message.content for choice in response.choices], usage

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available models through Cloudflare AI Gateway.
        
        Note: Since Cloudflare AI Gateway proxies to multiple providers, this list
        includes models from OpenAI, Anthropic, Groq, Mistral, Cohere, Perplexity,
        Workers AI, Google AI Studio, Grok, DeepSeek, and Cerebras.
        
        Returns:
            List[str]: List of model names available through the gateway
        """
        return [
            # OpenAI Models
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "openai/gpt-3.5-turbo-instruct",
            "openai/o1",
            "openai/o1-mini",
            "openai/o1-preview",
            
            # Anthropic Models
            "anthropic/claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-sonnet-20240620",
            "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-sonnet-20240229",
            "anthropic/claude-3-haiku-20240307",
            "anthropic/claude-2.1",
            "anthropic/claude-2.0",
            "anthropic/claude-instant-1.2",
            
            # Groq Models
            "groq/llama-3.1-405b-reasoning",
            "groq/llama-3.1-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/llama3-groq-70b-8192-tool-use-preview",
            "groq/llama3-groq-8b-8192-tool-use-preview",
            "groq/llama-3.2-90b-text-preview",
            "groq/llama-3.2-11b-text-preview",
            "groq/llama-3.2-3b-preview",
            "groq/llama-3.2-1b-preview",
            "groq/mixtral-8x7b-32768",
            "groq/gemma-7b-it",
            "groq/gemma2-9b-it",
            
            # Mistral AI Models
            "mistral/mistral-large-latest",
            "mistral/mistral-large-2407",
            "mistral/mistral-medium-latest",
            "mistral/mistral-small-latest",
            "mistral/mistral-tiny",
            "mistral/mixtral-8x7b-instruct",
            "mistral/mixtral-8x22b-instruct",
            "mistral/codestral-latest",
            "mistral/codestral-mamba-latest",
            "mistral/pixtral-12b-2409",
            
            # Cohere Models
            "cohere/command-r",
            "cohere/command-r-plus",
            "cohere/command",
            "cohere/command-light",
            "cohere/command-nightly",
            "cohere/command-light-nightly",
            
            # Perplexity Models
            "perplexity/llama-3.1-sonar-small-128k-online",
            "perplexity/llama-3.1-sonar-small-128k-chat",
            "perplexity/llama-3.1-sonar-large-128k-online",
            "perplexity/llama-3.1-sonar-large-128k-chat",
            "perplexity/llama-3.1-sonar-huge-128k-online",
            
            # Workers AI Models (Cloudflare's own)
            "workers-ai/@cf/meta/llama-3.1-8b-instruct",
            "workers-ai/@cf/meta/llama-3.1-70b-instruct",
            "workers-ai/@cf/meta/llama-3.2-1b-instruct",
            "workers-ai/@cf/meta/llama-3.2-3b-instruct",
            "workers-ai/@cf/meta/llama-3.2-11b-vision-instruct",
            "workers-ai/@cf/mistral/mistral-7b-instruct-v0.1",
            "workers-ai/@cf/google/gemma-7b-it",
            "workers-ai/@cf/qwen/qwen1.5-7b-chat-awq",
            "workers-ai/@cf/deepseek-ai/deepseek-math-7b-instruct",
            
            # Google AI Studio Models
            "google/gemini-1.5-pro",
            "google/gemini-1.5-flash",
            "google/gemini-1.0-pro",
            "google/gemini-1.0-pro-vision",
            "google/gemini-2.0-flash-exp",
            "google/palm-2-chat-bison",
            "google/palm-2-codechat-bison",
            
            # Grok Models (xAI)
            "grok/grok-beta",
            "grok/grok-vision-beta",
            
            # DeepSeek Models
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder",
            "deepseek/deepseek-r1-distill-qwen-32b",
            "deepseek/deepseek-r1-distill-llama-70b",
            
            # Cerebras Models
            "cerebras/llama3.1-8b",
            "cerebras/llama3.1-70b",
            "cerebras/llama3.3-70b",
        ] 