import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class GroqClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        use_responses_api: bool = False,
        local: bool = False,
        tools: List[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
        **kwargs
    ):
        """
        Initialize the Groq client using OpenAI compatibility.

        Args:
            model_name: The name of the model to use (default: "llama-3.3-70b-versatile")
            api_key: Groq API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            base_url: Base URL for the Groq API (default: "https://api.groq.com/openai/v1")
            use_responses_api: Whether to use responses API (default: False)
            tools: List of tools for function calling or MCP (default: None)
            reasoning_effort: Reasoning effort level for reasoning models (default: "low")
            local: Whether this is a local client (default: False)
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
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        
        # Initialize OpenAI client with Groq base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Responses API configuration
        self.use_responses_api = use_responses_api
        self.tools = tools
        self.reasoning_effort = reasoning_effort

    def responses(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle completions using the Groq Responses API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.responses.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        if "response_format" in kwargs:
            # Handle new format of structured outputs
            kwargs["text"] = {"format": kwargs["response_format"]}
            del kwargs["response_format"]
            if self.tools:
                del kwargs["text"]

        try:
            # Replace messages that have "system" with "developer" for Responses API
            for message in messages:
                if message["role"] == "system":
                    message["role"] = "developer"

            params = {
                "model": self.model_name,
                "input": messages,
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Add tools if provided
            if self.tools:
                params["tools"] = self.tools

            # Add reasoning configuration if needed
            if "reasoning" in kwargs or self.reasoning_effort:
                params["reasoning"] = {"effort": self.reasoning_effort}

            response = self.client.responses.create(**params)
            output_text = response.output

        except Exception as e:
            self.logger.error(f"Error during Groq Responses API call: {e}")
            raise

        # Extract the text output from the response
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
        Handle chat completions using the Groq API via OpenAI compatibility.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        # Use Responses API if enabled
        if self.use_responses_api:
            return self.responses(messages, **kwargs)
        
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Groq API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # Extract finish reasons
        finish_reasons = [choice.finish_reason for choice in response.choices]

        if self.local:
            return [choice.message.content for choice in response.choices], usage, finish_reasons
        else:
            return [choice.message.content for choice in response.choices], usage 