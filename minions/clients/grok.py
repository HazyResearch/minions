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
        region: Optional[str] = None,
        local: bool = False,
        enable_reasoning_output: bool = False,
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
            base_url: Base URL for the Grok API (default: "https://api.x.ai/v1").
                       Ignored if 'region' is specified.
            region: Optional region for data residency requirements (e.g., "us-east-1", "us-west-2", "eu-west-1").
                       
            enable_reasoning_output: Whether to include reasoning traces in output (default: False)
            use_native_sdk: Whether to use the native xai_sdk instead of OpenAI-compatible API.
                       Automatically enabled when file_ids is set.
            file_ids: Optional list of file IDs to attach to all chat messages.
                       Files enable Grok's document_search tool for reasoning over documents.
                       Use upload_file() to upload documents and get file IDs.
            **kwargs: Additional parameters passed to base class
        """
        # Construct regional endpoint URL if region is specified
        self.region = region
        if region:
            base_url = f"https://{region}.api.x.ai/v1"
        
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

        # File search configuration
        self.file_ids = file_ids or []
        self.uploaded_files = {}  # Cache of uploaded file info {file_id: file_info}

        # Use native SDK if files are provided or explicitly requested
        self.use_native_sdk = use_native_sdk or bool(file_ids)

        # Initialize native xAI SDK client if needed
        if self.use_native_sdk:
            try:
                from xai_sdk import Client as XAIClient
                from xai_sdk.chat import user as xai_user, file as xai_file

                # Configure native SDK with regional endpoint if specified
                sdk_kwargs = {"api_key": self.api_key}
                if self.region:
                    # Native SDK uses api_host without the https:// prefix
                    sdk_kwargs["api_host"] = f"{self.region}.api.x.ai"
                
                self.xai_client = XAIClient(**sdk_kwargs)
                self.xai_user = xai_user
                self.xai_file = xai_file
                region_info = f" (region: {self.region})" if self.region else ""
                self.logger.info(f"Initialized native xAI SDK client for file support{region_info}")
            except ImportError:
                if file_ids:
                    raise ImportError(
                        "File features require the xai_sdk package. "
                        "Install it with: pip install xai_sdk"
                    )
                self.logger.warning(
                    "xai_sdk not installed. File features will not be available. "
                    "Install with: pip install xai_sdk"
                )
                self.use_native_sdk = False
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

    def _find_first_user_message_index(self, messages: List[Dict[str, Any]]) -> int:
        """
        Find the index of the first user message in the messages list.
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                return i
        return -1

    # ==================== File Search Methods ====================

    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a file to xAI for use with Grok's document_search tool.

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
                from xai_sdk.chat import user as xai_user, file as xai_file

                # Configure native SDK with regional endpoint if specified
                sdk_kwargs = {"api_key": self.api_key}
                if self.region:
                    # Native SDK uses api_host without the https:// prefix
                    sdk_kwargs["api_host"] = f"{self.region}.api.x.ai"
                
                self.xai_client = XAIClient(**sdk_kwargs)
                self.xai_user = xai_user
                self.xai_file = xai_file
                self.use_native_sdk = True
                region_info = f" (region: {self.region})" if self.region else ""
                self.logger.info(f"Initialized native xAI SDK client{region_info}")
            except ImportError:
                raise ImportError(
                    "File operations require the xai_sdk package. "
                    "Install it with: pip install xai_sdk"
                )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Union[Tuple[List[str], Usage], Tuple[List[str], Usage, List[str]], Tuple[List[str], Usage, List[str], List[Optional[str]]]]:
        """
        Handle chat completions using the Grok API.

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

        # Use native xAI SDK if files are enabled
        if self.use_native_sdk and (self.file_ids or request_file_ids):
            return self._chat_with_native_sdk(messages, file_ids=request_file_ids, **kwargs)
        else:
            return self._chat_with_openai_api(messages, **kwargs)

    def _chat_with_native_sdk(self, messages: List[Dict[str, Any]], file_ids: Optional[List[str]] = None, **kwargs) -> Union[Tuple[List[str], Usage], Tuple[List[str], Usage, List[str]]]:
        """
        Handle chat completions using the native xAI SDK with file support.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            file_ids: Optional list of file IDs to attach to user messages (overrides instance file_ids)
            **kwargs: Additional arguments

        Returns:
            Tuple containing response strings, token usage, and optionally finish reasons
        """
        try:
            # Create chat session
            chat_kwargs = {
                "model": self.model_name,
            }

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