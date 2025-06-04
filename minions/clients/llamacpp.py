import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

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
from minions.clients.base import MinionsClient


class LlamaCppClient(MinionsClient):
    def __init__(
        self,
        model_path: str = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        chat_format: str = "llama-3",
        n_ctx: int = 0,
        n_gpu_layers: int = 0,
        embedding: bool = False,
        json_output: bool = False,
        tool_calling: bool = False,
        model_repo_id: Optional[str] = None,
        model_file_pattern: Optional[str] = "*.gguf",
        hf_token: Optional[str] = None,
        verbose: Optional[bool] = None,
        seed: int = -1,
        logits_all: bool = False,
        n_threads: Optional[int] = None,
        n_batch: int = 512,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        mul_mat_q: bool = True,
        offload_kqv: bool = True,
        flash_attn: bool = False,
        **kwargs,
    ):
        """Initialize LlamaCpp Client with a simpler interface.

        Args:
            model_path: Path to the model file
            model_name: Explicit model name if different from path/repo
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            chat_format: Chat format to use (default: "llama-3")
            n_ctx: Context window size (default: 0, from model)
            n_gpu_layers: Number of layers to offload to GPU (default: 0)
            embedding: Enable embedding generation (default: False)
            json_output: Whether to format responses as JSON (default: False)
            tool_calling: Support for tool calling (default: False)
            model_repo_id: Hugging Face model repo ID for downloading models
            model_file_pattern: File pattern for HF model (e.g., "*q4_0.gguf")
            hf_token: Hugging Face token for accessing gated models
            verbose: Explicitly declare verbose for super()
            seed: Seed for random number generation
            logits_all: Whether to return all logits
            n_threads: Number of threads to use
            n_batch: Batch size for processing
            rope_freq_base: Rope frequency base
            rope_freq_scale: Rope frequency scale
            mul_mat_q: Whether to multiply matrix q
            offload_kqv: Whether to offload kqv
            flash_attn: Whether to use flash attention
            **kwargs: Additional arguments for Llama constructor
        """
        # LlamaCpp uses model_path instead of model_name, so we need special handling
        effective_model_name = model_name or model_path or model_repo_id or "llamacpp-model"
        
        super().__init__(
            model_name=effective_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=model_path,
            chat_format=chat_format,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=embedding,
            json_output=json_output,
            tool_calling=tool_calling,
            model_repo_id=model_repo_id,
            model_file_pattern=model_file_pattern,
            hf_token=hf_token,
            verbose=verbose,
            seed=seed,
            logits_all=logits_all,
            n_threads=n_threads,
            n_batch=n_batch,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            mul_mat_q=mul_mat_q,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn,
            **kwargs
        )
        
        # Client-specific configuration
        # self.logger.setLevel(logging.INFO) # Logger level is set by super() if verbose is True

        # Attributes like self.n_ctx, self.chat_format, self.temperature, etc.,
        # are now set by super().__init__()
        
        # Prepare arguments for Llama constructor from self's attributes
        llama_constructor_args = {}
        
        # List of valid Llama constructor parameters expected to be set on self
        valid_llama_params = [
            "model_path", "n_ctx", "n_gpu_layers", "embedding", "chat_format", 
            "logits_all", "verbose", "seed", "n_threads", "n_batch", 
            "rope_freq_base", "rope_freq_scale", "mul_mat_q", "offload_kqv", "flash_attn",
            "main_gpu", "tensor_split", "vocab_only", "use_mmap", "use_mlock" # Add more as needed
        ]

        for param_name in valid_llama_params:
            if hasattr(self, param_name) and getattr(self, param_name) is not None:
                # Special handling for verbose: Llama expects bool, base sets self.verbose (which can be None)
                if param_name == "verbose":
                     llama_constructor_args[param_name] = bool(self.verbose) if self.verbose is not None else (logging.root.level == logging.DEBUG) # Llama's default for verbose
                else:
                    llama_constructor_args[param_name] = getattr(self, param_name)
        
        # Ensure model_path is present if not downloading
        effective_model_path = getattr(self, 'model_path', None)

        if self.model_repo_id and not (effective_model_path and Path(effective_model_path).exists()):
            self.logger.info(
                f"Model path {effective_model_path} not found or not provided. Downloading from {self.model_repo_id}"
            )
            downloaded_path = hf_hub_download(
                repo_id=self.model_repo_id,
                filename=self.model_file_pattern, # This might need to be more specific if multiple files match
                local_dir=os.path.join(os.getcwd(), "models"), # Ensure 'models' dir exists or is created
                local_dir_use_symlinks=False,
                token=getattr(self, 'hf_token', None),
            )
            llama_constructor_args["model_path"] = downloaded_path
            self.model_path = downloaded_path # Update self.model_path for consistency
            self.logger.info(f"Model downloaded to {downloaded_path}")
        elif not effective_model_path:
            raise ValueError(
                "LlamaCppClient requires either a valid 'model_path' or 'model_repo_id' for model download."
            )
        else:
            # Use the existing model_path if it's valid and no download is needed
            llama_constructor_args["model_path"] = effective_model_path


        self.model = Llama(**llama_constructor_args)

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
            response = self.model.create_chat_completion(**completion_kwargs)

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
                response = self.model.create_completion(**completion_kwargs)

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
                embedding = self.model.embed(text)
                embeddings.append(embedding)

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

        return embeddings
