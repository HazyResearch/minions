from typing import Any, Dict, List, Optional, Tuple, Union
from minions.usage import Usage
from minions.clients.base import MinionsClient
import logging
import os
import openai


class GrokClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "grok-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.x.ai/v1",
        local: bool = False,
        enable_reasoning_output: bool = False,
        use_search: bool = False,
        search_allowed_domains: Optional[List[str]] = None,
        search_excluded_domains: Optional[List[str]] = None,
        search_enable_image_understanding: bool = False,
        use_native_sdk: bool = False,
        file_ids: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the Grok client.
        
        Available models (as of Nov 2025):
            - grok-4: Flagship model for natural language, math, and reasoning (256K context)
            - grok-4-fast-reasoning: Cost-efficient reasoning model (2M context)
            - grok-4-0709: Specific Grok 4 version (256K context)
            - grok-code-fast-1: Optimized for agentic coding tasks (256K context)
            - grok-3-mini: Smaller efficient model
            - grok-3-mini-fast: Faster version of mini model

        Args:
            model_name: The name of the model to use (default: "grok-4")
            api_key: Grok API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Grok API (default: "https://api.x.ai/v1")
            reasoning_effort: Reasoning effort level for reasoning models ("low", "medium", "high", default: None)
            enable_reasoning_output: Whether to include reasoning traces in output (default: False)
            use_search: Whether to enable web search capabilities (default: False).
                       When enabled, the model can perform real-time web searches.
                       Note: This requires the xai_sdk package and uses the native xAI SDK.
            search_allowed_domains: Optional list of domains to restrict web searches to.
                       Example: ["wikipedia.org", "arxiv.org"]
            search_excluded_domains: Optional list of domains to exclude from web searches.
                       Example: ["reddit.com"]
            search_enable_image_understanding: Whether to enable image understanding during web searches.
            use_native_sdk: Whether to use the native xai_sdk instead of OpenAI-compatible API.
                       Automatically enabled when use_search or file_ids is set.
            file_ids: Optional list of file IDs to attach to all chat messages.
                       Files enable Grok's document_search tool for reasoning over documents.
                       Use upload_file() to upload documents and get file IDs.
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
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        openai.api_key = self.api_key
        self.enable_reasoning_output = enable_reasoning_output
        
        # Web search configuration
        self.use_search = use_search
        self.search_allowed_domains = search_allowed_domains
        self.search_excluded_domains = search_excluded_domains
        self.search_enable_image_understanding = search_enable_image_understanding
        self.last_search_results = None
        
        # File search configuration
        self.file_ids = file_ids or []
        self.uploaded_files = {}  # Cache of uploaded file info {file_id: file_info}
        
        # Use native SDK if search, files, or explicitly requested
        self.use_native_sdk = use_native_sdk or use_search or bool(file_ids)
        
        # Initialize native xAI SDK client if needed
        if self.use_native_sdk:
            try:
                from xai_sdk import Client as XAIClient
                from xai_sdk.tools import web_search
                from xai_sdk.chat import user as xai_user, file as xai_file
                
                self.xai_client = XAIClient(api_key=self.api_key)
                self.xai_web_search = web_search
                self.xai_user = xai_user
                self.xai_file = xai_file
                self.logger.info("Initialized native xAI SDK client for web search and file support")
            except ImportError:
                if use_search or file_ids:
                    raise ImportError(
                        "Web search and file features require the xai_sdk package. "
                        "Install it with: pip install xai_sdk"
                    )
                self.logger.warning(
                    "xai_sdk not installed. Web search and file features will not be available. "
                    "Install with: pip install xai_sdk"
                )
                self.use_native_sdk = False
                self.use_search = False
                self.file_ids = []

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available Grok models from the X.AI API.

        Returns:
            List[str]: List of model names available through X.AI
        """
        try:
            import requests
            
            # Try to use API key from environment or provided key
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                logging.warning("No XAI_API_KEY found in environment variables")
                # Return default models if no API key
                return []

            # Make API call to list models
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.x.ai/v1/models", headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            model_names = [model["id"] for model in models_data.get("data", [])]
            
            # If we got models from API, return them
            if model_names:
                return sorted(model_names, reverse=True)  # Sort with newest first
            # Fallback to default models if API returned empty
            return []
        except Exception as e:
            logging.error(f"Failed to get Grok model list: {e}")
            # Return fallback models on error
            return []

    def _is_reasoning_model(self, model_name: str) -> bool:
        """
        Check if the given model supports reasoning.
        
        Args:
            model_name: The model name to check
            
        Returns:
            bool: True if the model supports reasoning
        """
        reasoning_models = [
            "grok-3-mini", 
            "grok-3-mini-fast", 
            "grok-4",
            "grok-4-fast-reasoning",
            "grok-4-0709",
            "grok-code-fast-1"
        ]
        return any(reasoning_model in model_name.lower() for reasoning_model in reasoning_models)

    def _create_web_search_tool(self):
        """
        Create the web search tool for xAI SDK.
        
        Returns:
            The configured web_search tool instance
        """
        if not self.use_native_sdk or not hasattr(self, 'xai_web_search'):
            return None
        
        # Build web search configuration
        kwargs = {}
        if self.search_allowed_domains:
            kwargs['allowed_domains'] = self.search_allowed_domains
        if self.search_excluded_domains:
            kwargs['excluded_domains'] = self.search_excluded_domains
        if self.search_enable_image_understanding:
            kwargs['enable_image_understanding'] = self.search_enable_image_understanding
        
        return self.xai_web_search(**kwargs) if kwargs else self.xai_web_search()

    def _prepare_tools(self) -> Optional[List]:
        """
        Prepare the list of tools for the xAI SDK.
        
        Returns:
            List of tools or None if no tools are configured
        """
        tools = []
        
        if self.use_search:
            web_search_tool = self._create_web_search_tool()
            if web_search_tool:
                tools.append(web_search_tool)
                self.logger.info("Web search tool enabled")
        
        # Note: document_search tool is automatically enabled when files are attached
        # No need to explicitly add it to tools list
        
        return tools if tools else None

    def _find_first_user_message_index(self, messages: List[Dict[str, Any]]) -> int:
        """
        Find the index of the first user message in the messages list.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Index of first user message, or -1 if not found
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                return i
        return -1

    def get_search_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the search results from the last response.
        
        Returns:
            Optional[Dict[str, Any]]: Search results metadata if available, None otherwise
        """
        return self.last_search_results

    def enable_web_search(
        self,
        allowed_domains: Optional[List[str]] = None,
        excluded_domains: Optional[List[str]] = None,
        enable_image_understanding: bool = False
    ):
        """
        Enable web search functionality.
        
        Args:
            allowed_domains: Optional list of domains to restrict searches to
            excluded_domains: Optional list of domains to exclude from searches
            enable_image_understanding: Whether to enable image understanding
            
        Raises:
            ImportError: If xai_sdk is not installed
            
        Example:
            client.enable_web_search(allowed_domains=["wikipedia.org", "arxiv.org"])
        """
        self._ensure_native_sdk()
        
        self.use_search = True
        self.search_allowed_domains = allowed_domains
        self.search_excluded_domains = excluded_domains
        self.search_enable_image_understanding = enable_image_understanding
        self.logger.info("Web search enabled")

    def disable_web_search(self):
        """
        Disable web search functionality.
        
        Example:
            client.disable_web_search()
        """
        self.use_search = False
        self.logger.info("Web search disabled")

    def set_search_domains(
        self,
        allowed_domains: Optional[List[str]] = None,
        excluded_domains: Optional[List[str]] = None
    ):
        """
        Update the domain restrictions for web search.
        
        Args:
            allowed_domains: Optional list of domains to restrict searches to
            excluded_domains: Optional list of domains to exclude from searches
            
        Example:
            client.set_search_domains(
                allowed_domains=["wikipedia.org"],
                excluded_domains=["reddit.com"]
            )
        """
        self.search_allowed_domains = allowed_domains
        self.search_excluded_domains = excluded_domains
        self.logger.info(f"Search domains updated: allowed={allowed_domains}, excluded={excluded_domains}")

    # ==================== File Search Methods ====================
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a file to xAI for use with Grok's document_search tool.
        
        Supported formats: .txt, .md, .py, .js, .java, .csv, .json, .pdf, and other text-based formats.
        Maximum file size: 48 MB.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            Dict containing file information including 'id', 'filename', 'size', 'created_at'
            
        Raises:
            ImportError: If xai_sdk is not installed
            FileNotFoundError: If the file doesn't exist
            
        Example:
            file_info = client.upload_file("/path/to/document.pdf")
            print(f"Uploaded file ID: {file_info['id']}")
            
            # Use the file in chat
            client.add_file(file_info['id'])
        """
        self._ensure_native_sdk()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            file = self.xai_client.files.upload(file_path)
            file_info = {
                'id': file.id,
                'filename': getattr(file, 'filename', os.path.basename(file_path)),
                'size': getattr(file, 'size', None),
                'created_at': getattr(file, 'created_at', None),
            }
            # Cache the file info
            self.uploaded_files[file.id] = file_info
            self.logger.info(f"Uploaded file: {file_info['filename']} (ID: {file.id})")
            return file_info
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            raise

    def upload_file_bytes(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Upload file content as bytes to xAI.
        
        Args:
            content: File content as bytes
            filename: Name to assign to the file
            
        Returns:
            Dict containing file information including 'id', 'filename', 'size', 'created_at'
            
        Example:
            with open("document.pdf", "rb") as f:
                file_info = client.upload_file_bytes(f.read(), "document.pdf")
        """
        self._ensure_native_sdk()
        
        try:
            file = self.xai_client.files.upload(content, filename=filename)
            file_info = {
                'id': file.id,
                'filename': filename,
                'size': getattr(file, 'size', len(content)),
                'created_at': getattr(file, 'created_at', None),
            }
            self.uploaded_files[file.id] = file_info
            self.logger.info(f"Uploaded file: {filename} (ID: {file.id})")
            return file_info
        except Exception as e:
            self.logger.error(f"Error uploading file bytes: {e}")
            raise

    def list_files(self, limit: int = 10, order: str = "desc", sort_by: str = "created_at") -> List[Dict[str, Any]]:
        """
        List uploaded files.
        
        Args:
            limit: Maximum number of files to return (default: 10)
            order: Sort order - "asc" or "desc" (default: "desc")
            sort_by: Field to sort by (default: "created_at")
            
        Returns:
            List of file information dictionaries
            
        Example:
            files = client.list_files(limit=20)
            for f in files:
                print(f"File: {f['filename']} (ID: {f['id']})")
        """
        self._ensure_native_sdk()
        
        try:
            response = self.xai_client.files.list(limit=limit, order=order, sort_by=sort_by)
            files = []
            for file in response.data:
                file_info = {
                    'id': file.id,
                    'filename': getattr(file, 'filename', 'unknown'),
                    'size': getattr(file, 'size', None),
                    'created_at': getattr(file, 'created_at', None),
                }
                files.append(file_info)
                self.uploaded_files[file.id] = file_info
            return files
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            raise

    def get_file(self, file_id: str) -> Dict[str, Any]:
        """
        Get information about a specific file.
        
        Args:
            file_id: The ID of the file to retrieve
            
        Returns:
            Dict containing file information
            
        Example:
            file_info = client.get_file("file_abc123")
            print(f"Filename: {file_info['filename']}")
        """
        self._ensure_native_sdk()
        
        try:
            file = self.xai_client.files.retrieve(file_id)
            file_info = {
                'id': file.id,
                'filename': getattr(file, 'filename', 'unknown'),
                'size': getattr(file, 'size', None),
                'created_at': getattr(file, 'created_at', None),
            }
            self.uploaded_files[file_id] = file_info
            return file_info
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_id}: {e}")
            raise

    def delete_file(self, file_id: str) -> bool:
        """
        Delete an uploaded file.
        
        Args:
            file_id: The ID of the file to delete
            
        Returns:
            True if deletion was successful
            
        Example:
            client.delete_file("file_abc123")
        """
        self._ensure_native_sdk()
        
        try:
            self.xai_client.files.delete(file_id)
            # Remove from cache and active file IDs
            self.uploaded_files.pop(file_id, None)
            if file_id in self.file_ids:
                self.file_ids.remove(file_id)
            self.logger.info(f"Deleted file: {file_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {e}")
            raise

    def add_file(self, file_id: str):
        """
        Add a file ID to be attached to chat messages.
        When files are attached, Grok's document_search tool is automatically enabled.
        
        Args:
            file_id: The ID of the file to attach
            
        Example:
            file_info = client.upload_file("report.pdf")
            client.add_file(file_info['id'])
            response = client.chat([{"role": "user", "content": "What is the main conclusion?"}])
        """
        if file_id not in self.file_ids:
            self.file_ids.append(file_id)
            self.use_native_sdk = True
            self.logger.info(f"Added file to chat: {file_id}")

    def remove_file(self, file_id: str):
        """
        Remove a file ID from chat attachments (does not delete the file).
        
        Args:
            file_id: The ID of the file to remove from attachments
        """
        if file_id in self.file_ids:
            self.file_ids.remove(file_id)
            self.logger.info(f"Removed file from chat: {file_id}")

    def clear_files(self):
        """
        Clear all file attachments from chat (does not delete the files).
        """
        self.file_ids = []
        self.logger.info("Cleared all file attachments")

    def get_attached_files(self) -> List[str]:
        """
        Get the list of file IDs currently attached to chat.
        
        Returns:
            List of file IDs
        """
        return self.file_ids.copy()

    def _ensure_native_sdk(self):
        """
        Ensure the native xAI SDK is initialized.
        
        Raises:
            ImportError: If xai_sdk is not installed
        """
        if not hasattr(self, 'xai_client'):
            try:
                from xai_sdk import Client as XAIClient
                from xai_sdk.tools import web_search
                from xai_sdk.chat import user as xai_user, file as xai_file
                
                self.xai_client = XAIClient(api_key=self.api_key)
                self.xai_web_search = web_search
                self.xai_user = xai_user
                self.xai_file = xai_file
                self.use_native_sdk = True
                self.logger.info("Initialized native xAI SDK client")
            except ImportError:
                raise ImportError(
                    "File operations require the xai_sdk package. "
                    "Install it with: pip install xai_sdk"
                )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Union[Tuple[List[str], Usage], Tuple[List[str], Usage, List[str]], Tuple[List[str], Usage, List[str], List[Optional[str]]]]:
        """
        Handle chat completions using the Grok API.
        
        When use_search is enabled, the model can perform real-time web searches
        to provide up-to-date information. Search results can be retrieved using
        get_search_results() after the call.
        
        When files are attached (via file_ids or add_file()), the document_search
        tool is automatically enabled, allowing Grok to search and reason over
        your documents.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the API
                - file_ids: Optional list of file IDs to attach to this specific request

        Returns:
            Tuple containing response strings, token usage, and optionally finish reasons and reasoning content
        """
        assert len(messages) > 0, "Messages cannot be empty."
        
        # Check for per-request file_ids
        request_file_ids = kwargs.pop('file_ids', None)
        
        # Use native xAI SDK if web search or files are enabled
        if self.use_native_sdk and (self.use_search or self.file_ids or request_file_ids):
            return self._chat_with_native_sdk(messages, file_ids=request_file_ids, **kwargs)
        else:
            return self._chat_with_openai_api(messages, **kwargs)

    def _chat_with_native_sdk(self, messages: List[Dict[str, Any]], file_ids: Optional[List[str]] = None, **kwargs) -> Union[Tuple[List[str], Usage], Tuple[List[str], Usage, List[str]]]:
        """
        Handle chat completions using the native xAI SDK with web search and file support.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            file_ids: Optional list of file IDs to attach to user messages (overrides instance file_ids)
            **kwargs: Additional arguments

        Returns:
            Tuple containing response strings, token usage, and optionally finish reasons
        """
        try:
            # Prepare tools
            tools = self._prepare_tools()
            
            # Create chat session with tools
            chat_kwargs = {
                "model": self.model_name,
            }
            if tools:
                chat_kwargs["tools"] = tools
            
            chat = self.xai_client.chat.create(**chat_kwargs)
            
            # Determine which file IDs to use (per-request takes precedence)
            active_file_ids = file_ids if file_ids is not None else self.file_ids
            
            # Convert messages to xAI SDK format and append to chat
            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    # Attach files to the first user message if we have file IDs
                    if active_file_ids and i == self._find_first_user_message_index(messages):
                        # Create file attachments
                        file_attachments = [self.xai_file(fid) for fid in active_file_ids]
                        chat.append(self.xai_user(content, *file_attachments))
                        self.logger.info(f"Attached {len(active_file_ids)} file(s) to chat")
                    else:
                        chat.append(self.xai_user(content))
                elif role == "system":
                    # System messages are handled differently in xai_sdk
                    # Prepend to first user message or handle as system instruction
                    pass  # xai_sdk handles system context differently
                # Assistant messages are part of conversation history
            
            # Get the response
            response = chat.sample()
            
            # Extract content
            response_content = [response.content] if response.content else [""]
            
            # Try to extract usage information (may not be available in all SDK versions)
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage'):
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            
            # Store any search results from tool calls
            self.last_search_results = None
            if hasattr(response, 'tool_calls') and response.tool_calls:
                search_results = []
                for tool_call in response.tool_calls:
                    if hasattr(tool_call, 'function') and tool_call.function.name == 'web_search':
                        search_results.append({
                            'tool_call_id': getattr(tool_call, 'id', None),
                            'arguments': getattr(tool_call.function, 'arguments', {}),
                        })
                if search_results:
                    self.last_search_results = search_results
            
            # Return appropriate tuple
            if self.local:
                finish_reasons = ["stop"]  # xai_sdk doesn't provide finish_reason directly
                return response_content, usage, finish_reasons
            else:
                return response_content, usage
                
        except Exception as e:
            self.logger.error(f"Error during xAI SDK call: {e}")
            raise

    def _chat_with_openai_api(self, messages: List[Dict[str, Any]], **kwargs) -> Union[Tuple[List[str], Usage], Tuple[List[str], Usage, List[str]], Tuple[List[str], Usage, List[str], List[Optional[str]]]]:
        """
        Handle chat completions using the OpenAI-compatible API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to grok.chat.completions.create

        Returns:
            Tuple containing response strings, token usage, and optionally finish reasons and reasoning content
        """
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                **kwargs,
            }

            # Handle reasoning parameters for reasoning models
            is_reasoning_model = self._is_reasoning_model(self.model_name)
            
            if is_reasoning_model:
                # Reasoning models don't use temperature the same way
                if "temperature" not in kwargs:
                    # Only add temperature if not explicitly provided and model doesn't use reasoning
                    pass  # Don't set temperature for reasoning models unless explicitly requested
            else:
                # Regular models use temperature normally
                params["temperature"] = self.temperature

            client = openai.OpenAI(api_key=openai.api_key, base_url=self.base_url)
            response = client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Grok API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # Extract finish reasons
        finish_reasons = [choice.finish_reason for choice in response.choices]
        
        # Extract response content
        response_content = [choice.message.content for choice in response.choices]
        
        # Extract reasoning content if available and enabled
        reasoning_content = []
        if self.enable_reasoning_output and is_reasoning_model:
            for choice in response.choices:
                reasoning = getattr(choice.message, 'reasoning_content', None)
                reasoning_content.append(reasoning)

        # Return appropriate tuple based on what's requested
        if self.local:
            if self.enable_reasoning_output and reasoning_content and any(r is not None for r in reasoning_content):
                return f"{reasoning_content} \n {response_content}", usage, finish_reasons
            else:
                return response_content, usage, finish_reasons
        else:
            if self.enable_reasoning_output and reasoning_content and any(r is not None for r in reasoning_content):
                return f"{reasoning_content} \n {response_content}", usage
            else:
                return response_content, usage

   
