import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai
import requests

from minions.usage import Usage
from minions.clients.base import MinionsClient


# TODO: define one dataclass for what is returned from all the clients
class OpenAIClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        use_responses_api: bool = False,
        local: bool = False,
        tools: List[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
        conversation_id: Optional[str] = None,
        service_tier: Optional[str] = None,
        verbosity: Optional[str] = None,
        compact_threshold: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI client.

        Args:
            model_name: The name of the model to use (default: "gpt-4o")
            api_key: OpenAI API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the OpenAI API (optional, falls back to OPENAI_BASE_URL environment variable or default URL)
            use_responses_api: Whether to use responses API (default: False)
            tools: List of tools for function calling (default: None)
            reasoning_effort: Reasoning effort level for reasoning models. Valid values depend on model:
                - gpt-5, gpt-5-mini, gpt-5-nano: "minimal", "low", "medium", "high"
                - gpt-5-codex, gpt-5.1-codex: "low", "medium", "high" (no "minimal")
                - gpt-5.1-codex-max: "none", "medium", "high", "xhigh"
                - gpt-5.2: "none", "low", "medium", "high", "xhigh"
                (default: "low")
            conversation_id: Conversation ID for responses API (optional, only used when use_responses_api=True)
            service_tier: Service tier for request processing - "auto" or "priority" (default: None, which uses standard processing)
            verbosity: Verbosity level for responses API - "low", "medium", or "high" (default: None)
            compact_threshold: Token threshold for server-side compaction (optional, responses API only).
                When the rendered token count crosses this threshold, the server automatically
                compacts the conversation context. The response includes an encrypted compaction
                item that carries forward key state using fewer tokens.
                See: https://developers.openai.com/api/docs/guides/compaction
            local: If this is communicating with a local client (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

        # Initialize the client
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.use_responses_api = use_responses_api

        if not conversation_id and use_responses_api:
            self.conversation = self.client.conversations.create()
            self.conversation_id = self.conversation.id
        else:
            self.conversation_id = conversation_id

        self.tools = tools
        self.reasoning_effort = reasoning_effort
        
        # Validate reasoning_effort for the specific model
        self._validate_reasoning_effort()
        
        # Priority processing support via service_tier
        self.service_tier = service_tier
        if self.service_tier and self.service_tier not in ["auto", "priority"]:
            self.logger.warning(f"Invalid service_tier '{self.service_tier}'. Valid values are 'auto' or 'priority'. Using standard processing.")
            self.service_tier = None
        
        # Verbosity level for responses API
        self.verbosity = verbosity
        if self.verbosity and self.verbosity not in ["low", "medium", "high"]:
            self.logger.warning(f"Invalid verbosity '{self.verbosity}'. Valid values are 'low', 'medium', or 'high'. Using default.")
            self.verbosity = None
        
        # Server-side compaction for long-running conversations
        self.compact_threshold = compact_threshold

        # If we are using a local client, we want to check to see if the
        # local server is running or not
        if self.local:
            try:
                self.check_local_server_health()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(("Local OpenAI server at {} is "
                    "not running or reachable.".format(self.base_url)))

    def _is_reasoning_model(self) -> bool:
        """
        Check if the current model is a reasoning model (GPT-5 family).
        
        Returns:
            True if the model supports reasoning effort, False otherwise.
        """
        model_lower = self.model_name.lower()
        return "gpt-5" in model_lower or "gpt5" in model_lower

    def _validate_reasoning_effort(self):
        """
        Validate and warn about reasoning_effort compatibility with the model.
        
        Different GPT-5 model variants support different reasoning effort values:
        - gpt-5, gpt-5-mini, gpt-5-nano: "minimal", "low", "medium", "high"
        - gpt-5-codex, gpt-5.1-codex: "low", "medium", "high" (no "minimal")
        - gpt-5.1-codex-max: "none", "medium", "high", "xhigh"
        - gpt-5.2: "none", "low", "medium", "high", "xhigh"
        """
        if not self._is_reasoning_model():
            return
            
        model_lower = self.model_name.lower()
        
        # Define supported reasoning efforts per model variant
        if "gpt-5.2" in model_lower or "gpt5.2" in model_lower:
            supported = ["none", "low", "medium", "high", "xhigh"]
        elif "gpt-5.1-codex-max" in model_lower or "gpt5.1-codex-max" in model_lower:
            supported = ["none", "medium", "high", "xhigh"]
        elif "codex" in model_lower:
            # gpt-5-codex, gpt-5.1-codex variants (but not codex-max which is handled above)
            supported = ["low", "medium", "high"]
        else:
            # General gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1
            supported = ["minimal", "low", "medium", "high"]
        
        if self.reasoning_effort not in supported:
            self.logger.warning(
                f"reasoning_effort '{self.reasoning_effort}' may not be supported for model '{self.model_name}'. "
                f"Supported values are: {supported}. This may cause an API error."
            )

    def get_conversation_id(self):
        return self.conversation_id

    def responses(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage]:

        assert len(messages) > 0, "Messages cannot be empty."

        if "response_format" in kwargs:
            # handle new format of structure outputs from openai
            kwargs["text"] = {"format": kwargs["response_format"]}
            del kwargs["response_format"]
            if self.tools:
                del kwargs["text"]

        try:

            # replace an messages that have "system" with "developer"
            for message in messages:
                if message["role"] == "system":
                    message["role"] = "developer"

            params = {
                "model": self.model_name,
                "input": messages,
                "max_output_tokens": self.max_tokens,
                "tools": self.tools,
                "prompt_cache_key": "minions-v1",
                **kwargs,
            }
            
            # Add reasoning effort for GPT-5 reasoning models
            if self._is_reasoning_model():
                params["reasoning"] = {"effort": self.reasoning_effort}
            
            # Add conversation_id if provided
            if self.conversation_id is not None:
                params["conversation"] = self.conversation_id
            
            # Add service_tier for priority processing if specified
            if self.service_tier is not None:
                params["service_tier"] = self.service_tier
            
            # Add verbosity to text parameter if specified
            if self.verbosity is not None:
                if "text" not in params:
                    params["text"] = {}
                params["text"]["verbosity"] = self.verbosity
            
            # Add server-side compaction if threshold is set
            if self.compact_threshold is not None:
                params["context_management"] = [
                    {"type": "compaction", "compact_threshold": self.compact_threshold}
                ]

            response = self.client.responses.create(
                **params,
            )
            output_text = response.output

        except Exception as e:
            self.logger.error(f"Error during OpenAI API call: {e}")
            raise

        outputs = [output_text[0].content[0].text]

        # Extract usage information if it exists
        if response.usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=0)
        else:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

        return outputs, usage

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        if self.use_responses_api:
            return self.responses(messages, **kwargs)
        else:
            assert len(messages) > 0, "Messages cannot be empty."

            try:
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_completion_tokens": self.max_tokens,
                    **kwargs,
                }

                # Add temperature for non-reasoning models
                if not self._is_reasoning_model():
                    params["temperature"] = self.temperature
                
                # Add reasoning_effort for GPT-5 reasoning models
                if self._is_reasoning_model():
                    params["reasoning_effort"] = self.reasoning_effort
                
                # Add service_tier for priority processing if specified
                if self.service_tier is not None:
                    params["service_tier"] = self.service_tier

                response = self.client.chat.completions.create(**params)
            except Exception as e:
                self.logger.error(f"Error during OpenAI API call: {e}")
                raise

            # Extract usage information if it exists
            if response.usage is None:
                usage = Usage(prompt_tokens=0, completion_tokens=0)
            else:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

            # The content is now nested under message
            if self.local:
                return [choice.message.content for choice in response.choices], usage, [choice.finish_reason for choice in response.choices]
            else:
                return [choice.message.content for choice in response.choices], usage


    def check_local_server_health(self):
        """
        If we are using a local client, we want to be able
        to check if the local server is running or not
        """
        resp = requests.get(f"{self.base_url}/health") if "api" in self.base_url else requests.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()

    def list_models(self):
        """
        List available models from the OpenAI API.
        
        Returns:
            Dict containing the models data from the OpenAI API response
        """
        try:
            response = self.client.models.list()
            return {
                "object": "list",
                "data": [model.model_dump() for model in response.data]
            }
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise