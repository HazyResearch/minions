from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
import logging
import os

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    raise ImportError(
        "cerebras-cloud-sdk is required for CerebrasClient. "
        "Install it with: pip install cerebras-cloud-sdk"
    )


class CerebrasClient:
    # Valid service tier values
    VALID_SERVICE_TIERS = ("priority", "default", "auto", "flex")
    
    def __init__(
        self,
        model_name: str = "llama3.1-8b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        reasoning_effort: str = "low",  # low, medium, high
        service_tier: Optional[str] = None,
        queue_threshold: Optional[int] = None,
        local: bool = False,
    ):
        '''
        Initialize the Cerebras client.

        Args:
            model_name: The name of the model to use (default: "llama3.1-8b")
            api_key: Cerebras API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the API (optional, uses default if not provided)
            reasoning_effort: Reasoning effort level - "low", "medium", or "high" (default: "low")
            service_tier: Request prioritization tier (default: None, uses "default"):
                - "priority": Highest priority, processed first (dedicated endpoints only)
                - "default": Standard priority processing
                - "auto": Automatically uses highest available tier
                - "flex": Lowest priority, processed towards the end
            queue_threshold: Maximum acceptable queue time in milliseconds for flex/auto tiers.
                Valid range: 50-20000ms. If exceeded, request is rejected instead of waiting.
            local: Whether running locally (default: False)
        '''
        self.model_name = model_name
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self.logger = logging.getLogger("CerebrasClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local = local
        self.reasoning_effort = reasoning_effort
        
        # Validate and store service tier
        if service_tier is not None and service_tier not in self.VALID_SERVICE_TIERS:
            raise ValueError(
                f"Invalid service_tier: {service_tier}. "
                f"Must be one of: {', '.join(self.VALID_SERVICE_TIERS)}"
            )
        self.service_tier = service_tier
        
        # Validate and store queue threshold
        if queue_threshold is not None:
            if not (50 <= queue_threshold <= 20000):
                raise ValueError(
                    f"Invalid queue_threshold: {queue_threshold}. "
                    "Must be between 50 and 20000 milliseconds."
                )
            if service_tier not in ("auto", "flex"):
                self.logger.warning(
                    "queue_threshold only applies to 'auto' or 'flex' service tiers. "
                    f"Current tier: {service_tier or 'default'}"
                )
        self.queue_threshold = queue_threshold

        # Initialize the Cerebras client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = Cerebras(**client_kwargs)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        '''
        Handle chat completions using the Cerebras API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to cerebras.chat.completions.create
                - service_tier: Override instance service_tier for this request
                - queue_threshold: Override instance queue_threshold for this request

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        '''
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Get service tier from kwargs or instance default
            service_tier = kwargs.pop("service_tier", self.service_tier)
            queue_threshold = kwargs.pop("queue_threshold", self.queue_threshold)
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
                **kwargs,
            }
            
            # Add service tier if specified
            if service_tier is not None:
                params["service_tier"] = service_tier
            
            # Add queue threshold via extra_headers if specified
            extra_headers = {}
            if queue_threshold is not None:
                extra_headers["queue_threshold"] = str(queue_threshold)
            
            if extra_headers:
                response = self.client.chat.completions.create(
                    **params,
                    extra_headers=extra_headers
                )
            else:
                response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Cerebras API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # Extract finish reasons
        finish_reasons = ["stop"] * len(response.choices)
        
        # Extract response content
        if self.local:
            return [choice.message.content for choice in response.choices], usage, finish_reasons
        else:
            return [choice.message.content for choice in response.choices], usage 