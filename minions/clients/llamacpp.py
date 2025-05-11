import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from llama_cpp import Llama
except ImportError:
    print(
        "WARNING: llama-cpp-python is not installed. Please install it with `pip install llama-cpp-python`."
    )

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print(
        "WARNING: huggingface_hub is not installed. Please install it with `pip install huggingface-hub`."
    )

from minions.usage import Usage


class LlamaCppClient:
    def __init__(
        self,
        model_path: str = None,
        chat_format: str = "chatml",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        embedding: bool = False,
        json_output: bool = False,
        tool_calling: bool = False,
        model_repo_id: Optional[str] = None,
        model_file_pattern: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs,
    ):
        """Initialize LlamaCpp Client with a simpler interface.

        Args:
            model_path: Path to the model file
            chat_format: Chat format to use (default: "chatml")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            n_ctx: Context window size (default: 4096)
            n_gpu_layers: Number of layers to offload to GPU (default: 0)
            embedding: Enable embedding generation (default: False)
            json_output: Whether to format responses as JSON (default: False)
            tool_calling: Support for tool calling (default: False)
            model_repo_id: Hugging Face model repo ID for downloading models
            model_file_pattern: File pattern for HF model (e.g., "*q4_0.gguf")
            hf_token: Hugging Face token for accessing gated models
            **kwargs: Additional arguments for Llama constructor
        """
        self.logger = logging.getLogger("LlamaCppClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.chat_format = chat_format
        self.embedding = embedding
        self.json_output = json_output
        self.return_tools = tool_calling
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Ensure either model_path or model_repo_id is provided
        if model_path is None and model_repo_id is None:
            raise ValueError("Either model_path or model_repo_id must be provided")

        # If model_repo_id is provided, download the model from Hugging Face
        if model_repo_id:
            model_path = self._download_from_hf(model_repo_id, model_file_pattern)

        self.model_path = model_path
        self.logger.info(f"Loading model from {model_path}")

        # Initialize the Llama model
        try:
            self.llm = Llama(
                model_path=model_path,
                chat_format=chat_format,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                embedding=embedding,
                **kwargs,
            )
            self.logger.info(f"Successfully loaded model {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _download_from_hf(
        self, repo_id: str, file_pattern: Optional[str] = None
    ) -> str:
        """Download a model from Hugging Face Hub."""
        try:
            self.logger.info(f"Downloading model from Hugging Face: {repo_id}")

            # List files in the repo
            files = list_repo_files(repo_id, token=self.hf_token)

            # Find the appropriate model file
            if file_pattern:
                import fnmatch

                matching_files = [f for f in files if fnmatch.fnmatch(f, file_pattern)]
                if not matching_files:
                    raise ValueError(
                        f"No files matching '{file_pattern}' found in {repo_id}"
                    )
                target_file = matching_files[0]
            else:
                # Default to first .gguf file
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if not gguf_files:
                    raise ValueError(f"No .gguf files found in {repo_id}")
                target_file = gguf_files[0]

            self.logger.info(f"Downloading file: {target_file}")
            model_path = hf_hub_download(
                repo_id=repo_id, filename=target_file, token=self.hf_token
            )

            self.logger.info(f"Downloaded model to {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Error downloading from Hugging Face: {e}")
            raise

    def chat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """Handle chat completions using the given messages.

        Args:
            messages: List of message dictionaries or a single message dictionary
            **kwargs: Additional args to pass to create_chat_completion

        Returns:
            Tuple of (responses, usage, done_reasons)
        """
        # If a single dictionary is provided, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]

        # for each messages in messages convert {"role": "", "content": "", "image": ""} to {"role": "", "content": [{"type": "text", "text": ""}, {"type": "image_url", "image_url": {"url": ""}}]}
        for message in messages:
            if "image" in message:
                message["content"] = [
                    {"type": "text", "text": message["content"]},
                    {"type": "image_url", "image_url": {"url": message["image"]}},
                ]
                del message["image"]
            else:
                message["content"] = [{"type": "text", "text": message["content"]}]

        responses = []
        usage_total = Usage()
        done_reasons = []
        tools = []

        try:
            # Set up the completion parameters
            completion_kwargs = {
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Add JSON response format if requested
            if self.json_output or kwargs.get("json_output", False):
                completion_kwargs["response_format"] = {"type": "json_object"}

            # Add stop sequences if provided
            if "stop" in kwargs:
                completion_kwargs["stop"] = kwargs["stop"]

            print(messages)

            # Create the chat completion
            response = self.llm.create_chat_completion(**completion_kwargs)

            # Extract content from response
            content = response["choices"][0]["message"]["content"]
            responses.append(content)

            # Extract tool calls if present and requested
            if self.return_tools and "tool_calls" in response["choices"][0]["message"]:
                tools.append(response["choices"][0]["message"]["tool_calls"])

            # Track token usage
            usage_total += Usage(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
            )

            # Track completion reason
            done_reasons.append(response["choices"][0]["finish_reason"])

        except Exception as e:
            self.logger.error(f"Error in chat completion: {e}")
            raise

        if self.return_tools:
            return responses, usage_total, done_reasons, tools
        else:
            return responses, usage_total, done_reasons

    def complete(
        self, prompts: Union[str, List[str]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """Generate text completions for prompts.

        Args:
            prompts: String or list of strings to complete
            **kwargs: Additional args to pass to create_completion

        Returns:
            Tuple of (responses, usage, done_reasons)
        """
        # If a single string is provided, wrap it in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        responses = []
        usage_total = Usage()
        done_reasons = []

        try:
            for prompt in prompts:
                # Set up completion parameters
                completion_kwargs = {
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }

                # Add stop sequences if provided
                if "stop" in kwargs:
                    completion_kwargs["stop"] = kwargs["stop"]

                # Create the completion
                response = self.llm.create_completion(**completion_kwargs)

                # Extract content and append to responses
                text = response["choices"][0]["text"]
                responses.append(text)

                # Track token usage
                usage_total += Usage(
                    prompt_tokens=response["usage"]["prompt_tokens"],
                    completion_tokens=response["usage"]["completion_tokens"],
                )

                # Track completion reason
                done_reasons.append(response["choices"][0]["finish_reason"])

        except Exception as e:
            self.logger.error(f"Error in text completion: {e}")
            raise

        return responses, usage_total, done_reasons

    def embed(self, content: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given content.

        Args:
            content: Text string or list of strings to embed
            **kwargs: Additional args for embedding

        Returns:
            List of embedding vectors
        """
        if not self.embedding:
            raise ValueError(
                "Embedding functionality not enabled. Initialize with embedding=True"
            )

        # If a single string is provided, wrap it in a list
        if isinstance(content, str):
            content = [content]

        embeddings = []

        try:
            for text in content:
                embedding = self.llm.embed(text)
                embeddings.append(embedding)

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

        return embeddings
