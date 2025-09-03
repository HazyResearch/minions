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
            use_responses_api: Whether to use responses API for o1-pro models (default: False)
            tools: List of tools for function calling (default: None)
            reasoning_effort: Reasoning effort level for o1 models (default: "low")
            conversation_id: Conversation ID for responses API (optional, only used when use_responses_api=True)
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
        if "o1-pro" in self.model_name:
            self.use_responses_api = True
        else:
            self.use_responses_api = use_responses_api

        if not conversation_id and use_responses_api:
            self.conversation = self.client.conversations.create()
            self.conversation_id = self.conversation.id
        else:
            self.conversation_id = conversation_id

        self.tools = tools
        self.reasoning_effort = reasoning_effort
        if "gpt5" in self.model_name and self.reasoning_effort == "low":
            self.reasoning_effort = "minimal"

        # If we are using a local client, we want to check to see if the
        # local server is running or not
        if self.local:
            try:
                self.check_local_server_health()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(("Local OpenAI server at {} is "
                    "not running or reachable.".format(self.base_url)))


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
                **kwargs,
            }
            if "o1" in self.model_name or "o3" in self.model_name:
                params["reasoning"] = {"effort": self.reasoning_effort}
                # delete "tools" from params
                del params["tools"]
            
            # Add conversation_id if provided
            if self.conversation_id is not None:
                params["conversation"] = self.conversation_id


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

                # Only add temperature if NOT using the reasoning models (e.g., o3-mini model)
                if "o1" not in self.model_name and "o3" not in self.model_name:
                    params["temperature"] = self.temperature
                if "o1" in self.model_name or "o3" in self.model_name:
                    params["reasoning_effort"] = self.reasoning_effort

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