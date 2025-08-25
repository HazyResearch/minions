import asyncio
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import re

from minions.usage import Usage
from minions.clients.base import MinionsClient


class GeminiClient(MinionsClient):
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
        use_openai_api: bool = False,
        thinking_budget: Optional[int] = None,
        url_context: bool = False,
        use_search: bool = False,
        local: bool = False,
        **kwargs
    ):
        """Initialize Gemini Client.

        Args:
            model_name: The Gemini model to use.
            temperature: The temperature to use for generation. Higher values make output more random.
            max_tokens: The maximum number of tokens to generate.
            api_key: The API key to use. If not provided, it will be read from the GOOGLE_API_KEY environment variable.
            structured_output_schema: Optional Pydantic model for structured output.
            use_async: Whether to use async API calls.
            tool_calling: Whether to support tool calling.
            system_instruction: Optional system instruction to use for all calls.
            use_openai_api: Whether to use OpenAI-compatible API endpoint for Gemini models.
            thinking_budget: Optional thinking budget for reasoning models.
            url_context: Whether to enable URL context retrieval tool. When enabled, the model can
                       access content from URLs mentioned in messages (supports up to 20 URLs per request).
                       URLs are automatically detected and the tool is enabled dynamically if needed.
            use_search: Whether to enable Google Search tool. Can be combined with URL context for 
                       powerful search-then-analyze workflows.
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_async = use_async
        self.return_tools = tool_calling
        self.system_instruction = system_instruction
        self.use_openai_api = use_openai_api
        self.thinking_budget = thinking_budget
        self.url_context = url_context
        self.google_search = use_search
        self.last_url_context_metadata = None

        # If we want structured schema output:
        self.format_structured_output = None
        if structured_output_schema:
            self.format_structured_output = structured_output_schema.model_json_schema()

        # Initialize the client based on the chosen API
        if self.use_openai_api:
            if self.url_context or self.google_search:
                raise ValueError("URL context and Google Search are not supported with OpenAI-compatible API. Use native Gemini API instead.")
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self.logger.info("Initialized OpenAI-compatible client for Gemini API")
            except ImportError:
                self.logger.error(
                    "Failed to import openai. Please install it with 'pip install openai'"
                )
                raise
        else:
            # Initialize the Google Generative AI client
            try:
                from google import genai
                from google.genai import types

                self.client = genai.Client(api_key=self.api_key)
                self.genai = genai
                self.types = types
                self.logger.info("Initialized native Gemini API client")
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

            # Try to use API key from environment if available
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logging.warning("No GOOGLE_API_KEY found in environment variables")
                # Return default models if no API key
                return [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                    "gemini-2.0-pro",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ]

            client = genai.Client(api_key=api_key)
            models = client.list_models()
            
            # Extract model names and filter for Gemini models
            model_names = []
            for model in models:
                model_name = model.name
                # Remove the 'models/' prefix if present
                if model_name.startswith('models/'):
                    model_name = model_name[7:]
                
                # Only include Gemini models that support text generation
                if "gemini" in model_name.lower():
                    # Check if the model supports generateContent
                    supported_methods = getattr(model, 'supported_generation_methods', [])
                    if 'generateContent' in supported_methods or not supported_methods:
                        model_names.append(model_name)
            
            # Sort models with newest versions first
            model_names.sort(reverse=True)
            
            # If we got models from API, return them
            if model_names:
                return model_names
            
            # Fallback to default models if API returned empty
            return [
                "gemini-2.5-pro",
                "gemini-2.5-flash", 
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.0-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
            
        except Exception as e:
            logging.error(f"Failed to get Gemini model list: {e}")
            # Return default models including the new Flash-Lite
            return [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
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

    def _create_url_context_tool(self):
        """Create URL context tool for retrieving content from URLs."""
        return self.types.Tool(
            url_context=self.types.UrlContext()
        )

    def _create_google_search_tool(self):
        """Create Google Search tool for web search capabilities."""
        if not self.google_search:
            return None
        
        return self.types.Tool(
            google_search=self.types.GoogleSearch()
        )

    def _detect_urls_in_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if there are URLs in the message content.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if URLs are found, False otherwise
        """
        # More comprehensive URL pattern to match http/https URLs
        # This pattern matches URLs more accurately and handles edge cases better
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:[?](?:[;&\w\-.,@?^=%&:/~+#])*)?(?:[#](?:[\w\-.,@?^=%&:/~+#])*)?)?'
        
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                if isinstance(content, str) and re.search(url_pattern, content):
                    return True
        return False

    def extract_urls_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract all URLs from message content.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List[str]: List of URLs found in the messages
        """
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:[?](?:[;&\w\-.,@?^=%&:/~+#])*)?(?:[#](?:[\w\-.,@?^=%&:/~+#])*)?)?'
        urls = []
        
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                if isinstance(content, str):
                    found_urls = re.findall(url_pattern, content)
                    urls.extend(found_urls)
        
        return list(set(urls))  # Remove duplicates

    def _validate_url_context_support(self) -> bool:
        """
        Validate if the current model supports URL context feature.
        
        Returns:
            bool: True if model supports URL context, False otherwise
        """
        # Models that support URL context according to the documentation
        supported_models = {
            "gemini-2.5-pro",
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite",
            "gemini-live-2.5-flash-preview",
            "gemini-2.0-flash-live-001"
        }
        
        if self.model_name not in supported_models:
            self.logger.warning(f"Model '{self.model_name}' may not support URL context feature. Supported models: {supported_models}")
            return False
        return True

    def _prepare_tools(self, messages: Optional[List[Dict[str, Any]]] = None):
        """Prepare tools list for generation."""
        tools = []
        
        # Check if we should enable URL context automatically
        should_enable_url_context = self.url_context
        if not should_enable_url_context and messages:
            # Auto-detect URLs and enable URL context if found
            should_enable_url_context = self._detect_urls_in_messages(messages)
            if should_enable_url_context:
                # Validate URL count limit (max 20 URLs per request)
                urls = self.extract_urls_from_messages(messages)
                if len(urls) > 20:
                    self.logger.warning(f"Found {len(urls)} URLs in messages, but Gemini API supports max 20 URLs per request. Some URLs may not be processed.")
                else:
                    self.logger.info(f"URLs detected in messages ({len(urls)} URLs), automatically enabling URL context tool")
        
        if should_enable_url_context:
            # Validate model support for URL context
            if self._validate_url_context_support():
                url_tool = self._create_url_context_tool()
                if url_tool:
                    tools.append(url_tool)
            else:
                self.logger.warning("URL context requested but model may not support it, skipping URL context tool")
        
        if self.google_search:
            search_tool = self._create_google_search_tool()
            if search_tool:
                tools.append(search_tool)
        
        return tools if tools else None

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

    def _format_openai_messages(self, messages: List[Dict[str, Any]]):
        """Format messages for OpenAI API format."""
        formatted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Map roles to OpenAI format (which is already similar)
            if role == "assistant":
                role = "assistant"
            elif role == "model":
                role = "assistant"

            formatted_messages.append({"role": role, "content": content})

        return formatted_messages

    #
    #  ASYNC
    #
    def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a running event loop (e.g., in Streamlit)
                # Create a new loop in a separate thread to avoid conflicts
                import threading
                import concurrent.futures

                # Use a thread to run our async code
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_in_new_loop, messages, **kwargs)
                    return future.result()
            else:
                # We have a loop but it's not running
                return loop.run_until_complete(self._achat_internal(messages, **kwargs))
        except RuntimeError:
            # No event loop exists, create one (the normal case)
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

    def _run_in_new_loop(self, messages, **kwargs):
        """Run the async chat in a new event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._achat_internal(messages, **kwargs))
        finally:
            loop.close()

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

            if self.use_openai_api:
                # Format messages for OpenAI API
                formatted_messages = self._format_openai_messages(msg)

                # Create a new event loop for this async task
                loop = asyncio.get_event_loop()

                # Run the OpenAI API call in a thread pool
                response = await loop.run_in_executor(
                    None,
                    lambda: self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        system_content=self.system_instruction,
                    ),
                )

                # Extract usage information
                usage = Usage(
                    prompt_tokens=getattr(response, "usage", {}).get(
                        "prompt_tokens", 0
                    ),
                    completion_tokens=getattr(response, "usage", {}).get(
                        "completion_tokens", 0
                    ),
                )

                return {
                    "text": response.choices[0].message.content,
                    "usage": usage,
                    "finish_reason": response.choices[0].finish_reason or "stop",
                }
            else:
                # Use native Gemini API
                contents, system_instruction = self._format_content(msg)

                # Use instance system_instruction as fallback
                if not system_instruction:
                    system_instruction = self.system_instruction

                # Create a new event loop for this async task
                loop = asyncio.get_event_loop()

                # Prepare kwargs with generation config
                call_kwargs = {**kwargs}
                if generation_config:
                    call_kwargs["config"] = self.types.GenerationConfig(
                        **generation_config
                    )

                # Add system instruction if present
                if system_instruction:
                    call_kwargs["system_instruction"] = system_instruction

                # Prepare tools
                tools = self._prepare_tools(messages=msg)
                
                # Create GenerateContentConfig with tools and other settings
                config_kwargs = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
                
                # Add thinking config if specified
                if self.thinking_budget is not None:
                    config_kwargs["thinking_config"] = self.types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    )
                
                # Add tools if available
                if tools:
                    config_kwargs["tools"] = tools
                
                config = self.types.GenerateContentConfig(**config_kwargs)

                # Run the synchronous API call in a thread pool
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config,
                        system_instruction=system_instruction,
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

                # Extract URL context metadata if available
                url_context_metadata = None
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'url_context_metadata'):
                        url_context_metadata = candidate.url_context_metadata

                return {
                    "text": response.text,
                    "usage": usage,
                    "finish_reason": "stop",  # Gemini doesn't provide this directly
                    "url_context_metadata": url_context_metadata,
                }

        # Run them all in parallel
        results = await asyncio.gather(*(process_one(m) for m in messages))

        # Gather them back
        texts = []
        usage_total = Usage()
        done_reasons = []
        url_context_metadata = None
        for r in results:
            texts.append(r["text"])
            usage_total += r["usage"]
            done_reasons.append(r["finish_reason"])
            # Collect URL context metadata if available
            if r.get("url_context_metadata"):
                url_context_metadata = r["url_context_metadata"]

        # Store URL context metadata for later retrieval
        self.last_url_context_metadata = url_context_metadata
        
        if self.local:
            return texts, usage_total, done_reasons
        else:
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
        url_context_metadata = None

        try:
            if self.use_openai_api:
                # Use OpenAI-compatible API endpoint
                formatted_messages = self._format_openai_messages(messages)

                # Add system instruction if present
                system_content = self.system_instruction
                if system_content:
                    # Check if there's already a system message
                    has_system = any(
                        msg["role"] == "system" for msg in formatted_messages
                    )
                    if not has_system:
                        # Add system message at the beginning
                        formatted_messages.insert(
                            0, {"role": "system", "content": system_content}
                        )

                # Make the API call
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Extract text
                responses.append(response.choices[0].message.content)

                # Add finish reason
                done_reasons.append(response.choices[0].finish_reason or "stop")

                # Extract usage information
                usage_total += Usage(
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(response.usage, "completion_tokens", 0),
                )
            else:
                # Use native Gemini API
                # Format messages for Gemini API
                contents, system_instruction = self._format_content(messages)

                # Use instance system_instruction as fallback
                if not system_instruction:
                    system_instruction = self.system_instruction

                # Prepare tools
                tools = self._prepare_tools(messages=messages)
                
                # Create GenerateContentConfig with tools and other settings
                config_kwargs = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
                
                # Add thinking config if specified
                if self.thinking_budget is not None:
                    config_kwargs["thinking_config"] = self.types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    )
                
                # Add tools if available
                if tools:
                    config_kwargs["tools"] = tools

                config_kwargs["system_instruction"] = system_instruction
                
                config = self.types.GenerateContentConfig(**config_kwargs)

                # Make the API call
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                responses.append(response.text)

                # Extract URL context metadata if available
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'url_context_metadata'):
                        url_context_metadata = candidate.url_context_metadata
                        self.logger.info(f"URL context metadata: {url_context_metadata}")

                # Extract usage information
                usage_total += Usage(
                    prompt_tokens=response.usage_metadata.total_token_count
                    - response.usage_metadata.candidates_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                )

        except Exception as e:
            self.logger.error(f"Error during API call: {e}")
            raise

        # Store URL context metadata for later retrieval
        self.last_url_context_metadata = url_context_metadata
        
        if self.local:
            return responses, usage_total, done_reasons
        else:
            return responses, usage_total

    def get_url_context_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get URL context metadata from the last response.
        
        Returns:
            Optional[Dict[str, Any]]: URL context metadata if available, None otherwise
        """
        return self.last_url_context_metadata

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions, routing to async or sync implementation.
        
        The client will automatically detect URLs in the message content and enable
        URL context retrieval if URLs are found (unless explicitly disabled).
        
        After completion, you can retrieve URL context metadata using:
        get_url_context_metadata()
        
        Args:
            messages: List of message dictionaries or single message dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple[List[str], Usage, List[str]]: Response texts, usage info, and finish reasons
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)
