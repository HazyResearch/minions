import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage


# TODO: define one dataclass for what is returned from all the clients
class OpenAIClient:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        use_responses_api: bool = False,
        tools: List[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            model_name: The name of the model to use (default: "gpt-4o")
            api_key: OpenAI API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the OpenAI API (optional, falls back to OPENAI_BASE_URL environment variable or default URL)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.logger = logging.getLogger("OpenAIClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

        # Initialize the client
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.use_responses_api = use_responses_api
        self.tools = tools

    def responses(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage]:

        assert len(messages) > 0, "Messages cannot be empty."

        if "response_format" in kwargs:
            # handle new format of structure outputs from openai
            kwargs["text"] = {"format": kwargs["response_format"]}
            del kwargs["response_format"]

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

            response = self.client.responses.create(
                **params,
            )
            output_text = response.output

        except Exception as e:
            self.logger.error(f"Error during OpenAI API call: {e}")
            raise

        # extract the text from the output_text
        outputs = []
        for output in output_text:
            for content in output.content:
                outputs.append(content.text)

        usage = response.usage.input_tokens

        # Extract usage information
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

                response = self.client.chat.completions.create(**params)
            except Exception as e:
                self.logger.error(f"Error during OpenAI API call: {e}")
                raise

            # Extract usage information
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

            # The content is now nested under message
            return [choice.message.content for choice in response.choices], usage
