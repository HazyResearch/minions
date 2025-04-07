import asyncio
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Tuple
import os

from minions.usage import Usage


class GeminiClient:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        structured_output_schema: Optional[BaseModel] = None,
        use_async: bool = False,
        tool_calling: bool = False,
        system_instruction: Optional[str] = None,
    ):
        """Initialize Gemini Client."""
        self.model_name = model_name
        self.logger = logging.getLogger("GeminiClient")
        self.logger.setLevel(logging.INFO)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.use_async = use_async
        self.return_tools = tool_calling
        self.system_instruction = system_instruction

        # If we want structured schema output:
        self.format_structured_output = None
        if structured_output_schema:
            self.format_structured_output = structured_output_schema.model_json_schema()

        # Initialize the Google Generative AI client
        try:
            from google import genai
            from google.genai import types

            self.client = genai.Client(api_key=self.api_key)
            self.genai = genai
            self.types = types
        except ImportError:
            self.logger.error(
                "Failed to import google.genai. Please install it with 'pip install -q -U google-genai'"
            )
            raise

    @staticmethod
    def get_available_models():
        """
        Get a list of available Gemini models

        Returns:
            List[str]: List of model names
        """
        try:
            from google import genai

            client = genai.Client()
            models = client.list_models()
            # Extract model names from the list
            model_names = [model.name for model in models if "gemini" in model.name]
            return model_names
        except Exception as e:
            logging.error(f"Failed to get Gemini model list: {e}")
            return [
                "gemini-2.0-flash",
                "gemini-2.0-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

    def _prepare_generation_config(self):
        """Common generation config for both sync and async calls."""
        config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        return config

    def _format_content(self, messages: List[Dict[str, Any]]):
        """Format messages for Gemini API using the types module."""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Extract system instruction
            if role == "system":
                system_instruction = content
                continue

            # Map roles to Gemini format
            if role == "user":
                contents.append(
                    self.types.Content(
                        role="user", parts=[self.types.Part.from_text(text=content)]
                    )
                )
            elif role == "assistant" or role == "model":
                contents.append(
                    self.types.Content(
                        role="model", parts=[self.types.Part.from_text(text=content)]
                    )
                )

        return contents, system_instruction

    #
    #  ASYNC
    #
    def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], List[Usage], List[str]]:
        """
        Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )

        try:
            return asyncio.run(self._achat_internal(messages, **kwargs))
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Create a new event loop and set it as the current one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._achat_internal(messages, **kwargs)
                    )
                finally:
                    loop.close()
            raise

    async def _achat_internal(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle async chat with multiple messages in parallel.
        """
        # If the user provided a single dictionary, wrap it in a list.
        if isinstance(messages, dict):
            messages = [messages]

        # Now we have a list of dictionaries. We'll call them in parallel.
        generation_config = self._prepare_generation_config()

        async def process_one(msg):
            # Convert to Gemini format
            if isinstance(msg, dict):
                msg = [msg]

            contents, system_instruction = self._format_content(msg)

            # Use instance system_instruction as fallback
            if not system_instruction:
                system_instruction = self.system_instruction

            # Create a new event loop for this async task
            loop = asyncio.get_event_loop()

            # Prepare kwargs with generation config
            call_kwargs = {**kwargs}
            if generation_config:
                call_kwargs["config"] = self.types.GenerationConfig(**generation_config)

            # Add system instruction if present
            if system_instruction:
                call_kwargs["system_instruction"] = system_instruction

            # Run the synchronous API call in a thread pool
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=self.max_tokens,
                    ),
                ),
            )

            # Extract usage information
            usage = Usage(
                prompt_tokens=getattr(response, "usage_metadata", {}).get(
                    "prompt_token_count", 0
                ),
                completion_tokens=getattr(response, "usage_metadata", {}).get(
                    "candidates_token_count", 0
                ),
            )

            return {
                "text": response.text,
                "usage": usage,
                "finish_reason": "stop",  # Gemini doesn't provide this directly
            }

        # Run them all in parallel
        results = await asyncio.gather(*(process_one(m) for m in messages))

        # Gather them back
        texts = []
        usage_total = Usage()
        done_reasons = []
        for r in results:
            texts.append(r["text"])
            usage_total += r["usage"]
            done_reasons.append(r["finish_reason"])

        return texts, usage_total

    def schat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions.
        """
        # If the user provided a single dictionary, wrap it
        if isinstance(messages, dict):
            messages = [messages]

        # Prepare generation config
        generation_config = self._prepare_generation_config()

        responses = []
        usage_total = Usage()
        done_reasons = []
        tools = []

        try:
            # Format messages for Gemini API
            contents, system_instruction = self._format_content(messages)

            # Use instance system_instruction as fallback
            if not system_instruction:
                system_instruction = self.system_instruction

            # Prepare kwargs with generation config
            call_kwargs = {**kwargs}
            if generation_config:
                call_kwargs["config"] = self.types.GenerationConfig(**generation_config)

            # Add system instruction if present
            if system_instruction:
                call_kwargs["system_instruction"] = system_instruction

            print(call_kwargs["config"])

            # Make the API call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                # **call_kwargs,
            )

            responses.append(response.text)

            # Extract usage information
            usage_total += Usage(
                prompt_tokens=response.usage_metadata.total_token_count
                - response.usage_metadata.candidates_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
            )

        except Exception as e:
            self.logger.error(f"Error during Gemini API call: {e}")
            raise

        return responses, usage_total

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions, routing to async or sync implementation.
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)
