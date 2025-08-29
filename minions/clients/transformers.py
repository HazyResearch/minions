import logging
import os

from typing import Any, Dict, List, Optional, Tuple, Union


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
except ImportError:
    print(
        "WARNING: Transformers is not installed. Please install it with `pip install transformers`."
    )

try:
    from PIL import Image
except ImportError:
    Image = None  # PIL is optional; required only for vision models

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print(
        "WARNING: PyTorch is not installed. Please install it with `pip install torch`."
    )

try:
    from peft import PeftModel, PeftConfig
except ImportError:
    print("WARNING: PEFT is not installed. Please install it with `pip install peft`.")

try:
    from huggingface_hub import login
except ImportError:
    print(
        "WARNING: HuggingFace Hub is not installed. Please install it with `pip install huggingface_hub`."
    )

from minions.usage import Usage
from minions.clients.base import MinionsClient


class TransformersClient(MinionsClient):
    """
    Transformers client for local inference with HuggingFace models.
    
    Features:
    - Local model inference using transformers library
    - Support for LoRA adapters and PEFT models
    - Special support for NVIDIA Nemotron models with reasoning control
    - Special support for Hunyuan models with thinking mode
    - Hardware-optimized inference (CUDA/MPS/CPU)
    - Tool calling and embedding support
    - Vision support for Apple FastVLM models (single-image VLM chat)
    
    Nemotron Models:
    - Automatic detection of Nemotron models (nvidia/NVIDIA-Nemotron-*)
    - Reasoning control via /think and /no_think system prompts
    - Reasoning budget control (max thinking tokens)
    - Optimized parameters for reasoning vs non-reasoning modes
    
    Hunyuan Models:
    - Chain-of-Thought reasoning with <think> and <answer> tags
    - Access thinking process via get_thinking_content()
    
    Example usage with Nemotron:
        client = TransformersClient(
            model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            reasoning_enabled=True,
            reasoning_budget=500
        )
        
        response = client.chat([{"role": "user", "content": "Solve 2+2*3"}])
        
        # Or use nemotron_chat for explicit control
        response = client.nemotron_chat(
            messages=[{"role": "user", "content": "Complex reasoning task"}],
            reasoning_enabled=True,
            reasoning_budget=1000
        )
    """
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        hf_token: Optional[str] = None,
        tool_calling: bool = False,
        embedding_model: Optional[str] = None,
        enable_thinking: bool = False,  # for qwen and hunyuan models
        reasoning_enabled: bool = True,  # for nemotron models
        reasoning_budget: Optional[int] = None,  # for nemotron models
        local: bool = True,
        **kwargs
    ):
        """
        Initialize the Transformers client for local HuggingFace models.

        Args:
            model_name: The Hugging Face model identifier or local path.
                E.g., "EleutherAI/gpt-neox-20b", "/local/path/to/checkpoint", or "hf://mistralai/Mistral-7B-v0.1"
                For Hunyuan models (e.g., "tencent/Hunyuan-1.8B-Instruct"), special thinking mode is supported.
                For Nemotron models (e.g., "nvidia/NVIDIA-Nemotron-Nano-9B-v2"), reasoning control is supported.
            temperature: Sampling temperature for generation (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 1024)
            top_p: Top-p sampling parameter (default: 1.0)
            min_p: Minimum probability threshold for sampling (default: 0.0)
            repetition_penalty: Penalty for repeated tokens (default: 1.0)
            do_sample: Whether to use sampling for generation (default: False)
            hf_token: Optional Hugging Face token for accessing gated models
            tool_calling: Whether to support tool calling (default: False)
            embedding_model: Optional separate model for embeddings (default: None, uses main model)
            enable_thinking: Whether to enable thinking mode for qwen and hunyuan models (default: False)
                For Hunyuan models, this enables Chain-of-Thought reasoning with <think> and <answer> tags.
                Use get_thinking_content() to access the thinking process separately.
            reasoning_enabled: Whether to enable reasoning for Nemotron models (default: True)
                Controls whether /think or /no_think is added to system prompts for Nemotron models.
            reasoning_budget: Maximum tokens allowed for reasoning (Nemotron models only, default: None)
                Limits the number of "thinking" tokens the model can use before providing final answer.
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.return_tools = tool_calling
        self.embedding_model_name = embedding_model
        self.enable_thinking = enable_thinking
        
        # Nemotron-specific configuration
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_budget = reasoning_budget
        self.is_nemotron = self._is_nemotron_model(model_name)
        self.is_fastvlm = self._is_fastvlm_model(model_name)
        
        if self.is_nemotron:
            self.logger.info(f"Detected Nemotron model: {model_name}")
            self.logger.info(f"Reasoning enabled: {reasoning_enabled}")
            if reasoning_budget:
                self.logger.info(f"Reasoning budget: {reasoning_budget} tokens")

        # Check device availability
        self.device, self.dtype = self._get_device_and_dtype()
        self.logger.info(f"Using device: {self.device}, dtype: {self.dtype}")

        # Authenticate with Hugging Face if token is provided
        self._authenticate_huggingface()

        # Load model and tokenizer
        self.model, self.tokenizer = self._build_model_and_tokenizer()

        # Load embedding model if specified
        self.embedding_model = None
        self.embedding_tokenizer = None
        if self.embedding_model_name:
            self._load_embedding_model()

        # Store last Hunyuan thinking content for access via get_thinking_content()
        self._last_hunyuan_thinking = ""

        self.logger.info(f"Loaded Hugging Face model: {self.model_name}")

    def _is_fastvlm_model(self, model_name: str) -> bool:
        """
        Detect Apple FastVLM models which require special image handling.
        """
        name = (model_name or "").lower()
        return "fastvlm" in name or "apple/fastvlm" in name

    def _is_hunyuan_model(self) -> bool:
        """
        Check if the current model is a Hunyuan model.
        
        Returns:
            bool: True if the model is a Hunyuan model, False otherwise
        """
        return "hunyuan" in self.model_name.lower()

    def _is_nemotron_model(self, model_name: str) -> bool:
        """
        Check if the model is a Nemotron model.
        
        Args:
            model_name: The model name to check
            
        Returns:
            True if the model is a Nemotron model, False otherwise
        """
        nemotron_patterns = [
            "nemotron",
            "NVIDIA-Nemotron",
            "nvidia/NVIDIA-Nemotron"
        ]
        return any(pattern.lower() in model_name.lower() for pattern in nemotron_patterns)

    def _extract_first_image(self, messages: List[Dict[str, Any]]) -> Optional["Image.Image"]:
        """
        Extract and load the first image referenced in messages.
        Supports keys: 'images' (list/str), 'image' (str), 'image_url' (data URL or path).
        Returns a PIL Image if available.
        """
        if not messages:
            return None

        def _open_image_from_path(path: str) -> Optional["Image.Image"]:
            if not path:
                return None
            try:
                if Image is None:
                    self.logger.error("Pillow (PIL) is required for vision models. Install with `pip install pillow`.")
                    return None
                return Image.open(path).convert("RGB")
            except Exception as e:
                self.logger.error(f"Failed to open image '{path}': {e}")
                return None

        def _open_image_from_data_url(data_url: str) -> Optional["Image.Image"]:
            try:
                if not data_url.startswith("data:"):
                    return None
                import base64, io
                header, b64 = data_url.split(",", 1)
                img_bytes = base64.b64decode(b64)
                if Image is None:
                    self.logger.error("Pillow (PIL) is required for vision models. Install with `pip install pillow`.")
                    return None
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                self.logger.error(f"Failed to decode image data URL: {e}")
                return None

        for msg in messages:
            # 'images': list[str] | str
            if isinstance(msg, dict) and "images" in msg:
                images_val = msg.get("images")
                if isinstance(images_val, list) and images_val:
                    return _open_image_from_path(images_val[0])
                if isinstance(images_val, str):
                    return _open_image_from_path(images_val)
            # 'image': str
            if isinstance(msg, dict) and "image" in msg and isinstance(msg["image"], str):
                return _open_image_from_path(msg["image"])
            # 'image_url': data URL or path
            if isinstance(msg, dict) and "image_url" in msg and isinstance(msg["image_url"], str):
                url = msg["image_url"]
                if url.startswith("data:"):
                    img = _open_image_from_data_url(url)
                    if img is not None:
                        return img
                else:
                    # Treat as local path (best effort)
                    return _open_image_from_path(url)
        return None

    def _parse_hunyuan_response(self, completion_text: str) -> Dict[str, str]:
        """
        Parse Hunyuan model response to extract thinking content and answer content.
        
        Args:
            completion_text: Raw completion text from the model
            
        Returns:
            Dict containing 'thinking', 'answer', and 'full_response' keys
        """
        import re
        
        result = {
            'thinking': '',
            'answer': '',
            'full_response': completion_text
        }
        
        # Extract thinking content
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, completion_text, re.DOTALL)
        if think_matches:
            result['thinking'] = think_matches[0].strip()
        
        # Extract answer content
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, completion_text, re.DOTALL)
        if answer_matches:
            result['answer'] = answer_matches[0].strip()
        
        return result

    def get_thinking_content(self) -> str:
        """
        Get the thinking content from the last Hunyuan model response.
        
        Returns:
            str: The thinking content extracted from <think> tags, empty string if none
        """
        return self._last_hunyuan_thinking

    def set_reasoning_enabled(self, enabled: bool) -> None:
        """
        Enable or disable reasoning for Nemotron models.
        
        Args:
            enabled: Whether to enable reasoning
        """
        if not self.is_nemotron:
            self.logger.warning("Reasoning control is only available for Nemotron models")
            return
            
        self.reasoning_enabled = enabled
        self.logger.info(f"Reasoning {'enabled' if enabled else 'disabled'}")

    def set_reasoning_budget(self, budget: Optional[int]) -> None:
        """
        Set the reasoning budget (maximum thinking tokens) for Nemotron models.
        
        Args:
            budget: Maximum tokens allowed for reasoning, or None for no limit
        """
        if not self.is_nemotron:
            self.logger.warning("Reasoning budget control is only available for Nemotron models")
            return
            
        self.reasoning_budget = budget
        if budget:
            self.logger.info(f"Reasoning budget set to {budget} tokens")
        else:
            self.logger.info("Reasoning budget removed (no limit)")

    def _prepare_nemotron_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare messages for Nemotron models by adding appropriate reasoning control.
        
        Args:
            messages: Original messages
            
        Returns:
            Modified messages with reasoning control
        """
        if not self.is_nemotron:
            return messages
        
        # Create a copy to avoid modifying the original
        prepared_messages = messages.copy()
        
        # Check if there's already a system message with reasoning control
        has_reasoning_control = False
        for msg in prepared_messages:
            if (msg.get("role") == "system" and 
                isinstance(msg.get("content"), str) and 
                ("/think" in msg["content"] or "/no_think" in msg["content"])):
                has_reasoning_control = True
                break
        
        # If no reasoning control is present, add it
        if not has_reasoning_control:
            reasoning_prompt = "/think" if self.reasoning_enabled else "/no_think"
            
            # Check if there's already a system message
            system_msg_idx = None
            for i, msg in enumerate(prepared_messages):
                if msg.get("role") == "system":
                    system_msg_idx = i
                    break
            
            if system_msg_idx is not None:
                # Append to existing system message
                existing_content = prepared_messages[system_msg_idx]["content"]
                prepared_messages[system_msg_idx]["content"] = f"{reasoning_prompt}\n{existing_content}"
            else:
                # Add new system message at the beginning
                system_msg = {"role": "system", "content": reasoning_prompt}
                prepared_messages.insert(0, system_msg)
        
        return prepared_messages

    def _prepare_nemotron_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare kwargs for Nemotron models, including reasoning budget control.
        
        Args:
            **kwargs: Original kwargs
            
        Returns:
            Modified kwargs with Nemotron-specific parameters
        """
        if not self.is_nemotron:
            return kwargs
        
        # Create a copy to avoid modifying the original
        prepared_kwargs = kwargs.copy()
        
        # Add reasoning budget if specified and not already present
        if self.reasoning_budget is not None and "max_thinking_tokens" not in prepared_kwargs:
            prepared_kwargs["max_thinking_tokens"] = self.reasoning_budget
        
        # Set recommended parameters for Nemotron models if not specified
        if self.reasoning_enabled:
            # For reasoning mode, use higher temperature and different sampling if not specified
            if "temperature" not in prepared_kwargs:
                prepared_kwargs["temperature"] = 0.6
            if "top_p" not in prepared_kwargs:
                prepared_kwargs["top_p"] = 0.95
            if "do_sample" not in prepared_kwargs:
                prepared_kwargs["do_sample"] = True
            if "max_completion_tokens" not in prepared_kwargs and "max_tokens" not in prepared_kwargs:
                prepared_kwargs["max_completion_tokens"] = 1024
        else:
            # For non-reasoning mode, use greedy search if not specified
            if "temperature" not in prepared_kwargs:
                prepared_kwargs["temperature"] = 0.0
            if "do_sample" not in prepared_kwargs:
                prepared_kwargs["do_sample"] = False
        
        return prepared_kwargs

    def _get_device_and_dtype(self):
        """
        Determine the appropriate device and dtype to use.

        Returns:
            tuple: (device, dtype)
        """
        if torch.cuda.is_available():
            self.logger.info("CUDA is available, using GPU")
            return "cuda", torch.bfloat16
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.logger.info("MPS is available, using Apple Silicon GPU")
            # Note: bfloat16 may not be supported on all MPS devices,
            # but float16 is generally available
            return "mps", torch.bfloat16
        else:
            self.logger.info("No GPU available, using CPU")
            return "cpu", torch.float32

    def _authenticate_huggingface(self):
        """
        Authenticate with Hugging Face using the provided token or environment variable.
        """
        if self.hf_token:
            self.logger.info("Authenticating with Hugging Face...")
            login(token=self.hf_token, write_permission=False)
            self.logger.info("Successfully authenticated with Hugging Face")
        else:
            self.logger.warning(
                "No Hugging Face token provided. Gated models may not be accessible."
            )

    def _load_embedding_model(self):
        """
        Load a separate model for generating embeddings.
        """
        self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name, token=self.hf_token
            )

            # For embedding models we typically use AutoModel instead of AutoModelForCausalLM
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name, token=self.hf_token
            )

            # Move to appropriate device
            self.embedding_model.to(self.device)

            self.embedding_model.eval()
            self.logger.info(
                f"Successfully loaded embedding model: {self.embedding_model_name}"
            )
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise

    def _build_model_and_tokenizer(self):
        """
        Build and return the model and tokenizer from the checkpoint path.
        Supports HF Hub models, local paths, and PEFT adapters.

        Returns:
            tuple: (model, tokenizer)
        """
        ckpt_path = self.model_name
        self.logger.info(f"Loading model from {ckpt_path}...")

        # Check if this is a HF Hub model by looking for the hf:// prefix
        is_hf_model = ckpt_path.startswith("hf://") or not ckpt_path.startswith("/")

        # If it's an HF model with the prefix, remove the prefix
        if ckpt_path.startswith("hf://"):
            ckpt_path = ckpt_path[5:]

        # Check if this is a LoRA adapter by looking for adapter_config.json
        adapter_config_path = (
            os.path.join(ckpt_path, "adapter_config.json") if not is_hf_model else None
        )
        is_lora_adapter = (
            False
            if is_hf_model
            else (os.path.exists(adapter_config_path) if adapter_config_path else False)
        )

        # Determine if we should use device_map=auto for higher-level device management
        use_device_map = self.device in ["cuda", "mps"] and not is_lora_adapter

        if is_lora_adapter:
            self.logger.info(f"Detected LoRA adapter at {ckpt_path}")
            try:
                # Load the adapter config to get the base model name
                adapter_config = PeftConfig.from_pretrained(ckpt_path)
                base_model_name = adapter_config.base_model_name_or_path
                self.logger.info(f"Base model: {base_model_name}")

                # First load the tokenizer from the base model
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, token=self.hf_token
                )

                # Add pad token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

                self.logger.info(f"Loading base model: {base_model_name}")
                # Load the base model with explicit device mapping
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=self.dtype,
                    device_map="auto" if use_device_map else None,
                    token=self.hf_token,
                    trust_remote_code=True,
                )

                # Resize token embeddings to match the tokenizer
                model.resize_token_embeddings(len(tokenizer))

                # Load the LoRA adapter
                self.logger.info(f"Loading LoRA adapter: {ckpt_path}")
                model = PeftModel.from_pretrained(model, ckpt_path)

                # Merge weights for better performance
                self.logger.info("Merging LoRA weights into base model")
                model = model.merge_and_unload()
                self.logger.info("LoRA weights merged into base model")
            except Exception as e:
                self.logger.error(f"Error loading LoRA adapter: {str(e)}")
                raise
        else:
            self.logger.info(f"Loading full model: {ckpt_path}")
            # Original code path for loading a full model
            # First load the tokenizer

            tokenizer = AutoTokenizer.from_pretrained(ckpt_path, token=self.hf_token)

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=self.dtype,
                device_map="auto" if use_device_map else None,
                token=self.hf_token,
                trust_remote_code=True,
            )

            # Resize token embeddings to match the tokenizer
            model.resize_token_embeddings(len(tokenizer))

        # Move model to device if device_map wasn't used
        if not use_device_map and not isinstance(
            getattr(model, "hf_device_map", None), dict
        ):
            self.logger.info(f"Moving model to {self.device} device")
            model = model.to(self.device)

        # Set model to eval mode
        model.eval()

        # Log model device information
        self.logger.info(f"Model device map: {getattr(model, 'hf_device_map', None)}")
        self.logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch, "mps"):
            self.logger.info(f"Is MPS available: {torch.backends.mps.is_available()}")
        device_info = next(model.parameters()).device
        self.logger.info(f"Model is on device: {device_info}")

        return model, tokenizer

    def complete(
        self, prompts: Union[str, List[str]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Generate completions for the given text prompts.

        Args:
            prompts: String or list of strings to generate completions for
            **kwargs: Additional arguments to pass to model.generate

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        # If a single string is provided, wrap it in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        responses = []
        done_reasons = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)

        try:
            for prompt in prompts:
                # Tokenize the prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                prompt_token_count = input_ids.shape[1]
                usage.prompt_tokens += prompt_token_count

                # Move input tokens to the correct device
                input_ids = input_ids.to(self.model.device)

                max_tokens = kwargs.get("max_completion_tokens", self.max_tokens)
                temperature = kwargs.get("temperature", self.temperature)
                top_p = kwargs.get("top_p", self.top_p)
                min_p = kwargs.get("min_p", self.min_p)
                repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)
                do_sample = kwargs.get("do_sample", self.do_sample)

                # Generate completion
                with torch.no_grad():
                    generation_kwargs = {
                        "max_new_tokens": max_tokens,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p,
                        "return_dict_in_generate": True,
                        "output_logits": True,
                    }
                    
                    # Only add min_p and repetition_penalty if they're not default values
                    if min_p > 0.0:
                        generation_kwargs["min_p"] = min_p
                    if repetition_penalty != 1.0:
                        generation_kwargs["repetition_penalty"] = repetition_penalty
                    
                    gen_out = self.model.generate(
                        input_ids,
                        **generation_kwargs
                    )

                    # Extract token IDs for the completion
                    output_ids = gen_out.sequences[0]
                    completion_ids = output_ids[prompt_token_count:]

                    completion_text = self.tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    )

                    # Handle stop sequences
                    stop_sequences = kwargs.get("stop", [])
                    for s in stop_sequences:
                        if s in completion_text:
                            completion_text = completion_text.split(s, 1)[0]
                            break

                    completion_token_count = len(completion_ids)
                    usage.completion_tokens += completion_token_count

                    responses.append(completion_text)
                    done_reasons.append(
                        "stop_string"
                        if any(s in completion_text for s in stop_sequences)
                        else "max_tokens"
                    )

        except Exception as e:
            self.logger.error(f"Error during completion generation: {e}")
            raise

        return responses, usage, done_reasons

    def nemotron_chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        reasoning_enabled: Optional[bool] = None,
        reasoning_budget: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Chat with Nemotron models with explicit reasoning control.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys or a single message dictionary
            reasoning_enabled: Override the default reasoning setting for this request
            reasoning_budget: Override the default reasoning budget for this request
            **kwargs: Additional arguments to pass to model.generate
            
        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
            If tool_calling is enabled, returns (List[str], Usage, List[str], List[tool_calls])
            
        Example:
            # Enable reasoning for this specific request
            response = client.nemotron_chat(
                messages=[{"role": "user", "content": "Solve this math problem: 2+2*3"}],
                reasoning_enabled=True,
                reasoning_budget=100
            )
        """
        if not self.is_nemotron:
            self.logger.warning("nemotron_chat should only be used with Nemotron models")
            
        # Temporarily override settings if provided
        original_reasoning = self.reasoning_enabled
        original_budget = self.reasoning_budget
        
        if reasoning_enabled is not None:
            self.reasoning_enabled = reasoning_enabled
        if reasoning_budget is not None:
            self.reasoning_budget = reasoning_budget
            
        try:
            result = self.chat(messages, **kwargs)
        finally:
            # Restore original settings
            self.reasoning_enabled = original_reasoning
            self.reasoning_budget = original_budget
            
        return result

    def chat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Transformers model.
        
        For Hunyuan models with enable_thinking=True, responses are automatically parsed to extract
        the answer from <answer> tags, while thinking content from <think> tags is stored and can
        be accessed via get_thinking_content().

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys or a single message dictionary
                For Hunyuan models, you can add "/think" or "/no_think" prefixes to force thinking behavior.
            **kwargs: Additional arguments to pass to model.generate

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
            If tool_calling is enabled, returns (List[str], Usage, List[str], List[tool_calls])
        """
        # If the user provided a single dictionary, wrap it
        if isinstance(messages, dict):
            messages = [messages]

        # Prepare messages and kwargs for Nemotron models
        prepared_messages = self._prepare_nemotron_messages(messages)
        prepared_kwargs = self._prepare_nemotron_kwargs(**kwargs)

        responses = []
        done_reasons = []
        tools = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)

        try:
            # Prepare inputs, including FastVLM vision path if applicable
            px = None  # pixel values for image if any

            if self.is_fastvlm:
                # For FastVLM, if an image is present, we must splice in the <image> token
                pil_img = self._extract_first_image(prepared_messages)
                if pil_img is not None:
                    # Insert a placeholder into the first user message content if not already present
                    messages_for_template = []
                    image_inserted = False
                    for m in prepared_messages:
                        m_copy = dict(m)
                        if not image_inserted and m_copy.get("role") == "user":
                            content = m_copy.get("content", "")
                            if isinstance(content, str):
                                if "<image>" not in content:
                                    m_copy["content"] = f"<image>\n{content}"
                                image_inserted = True
                        messages_for_template.append(m_copy)

                    rendered = self.tokenizer.apply_chat_template(
                        messages_for_template,
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                    if "<image>" not in rendered:
                        # As a fallback, prepend at the very start
                        rendered = f"<image>\n{rendered}"

                    pre, post = rendered.split("<image>", 1)

                    pre_ids = self.tokenizer(
                        pre,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids
                    post_ids = self.tokenizer(
                        post,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids

                    IMAGE_TOKEN_INDEX = -200  # FastVLM expects this special token id
                    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
                    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)

                    # Preprocess image via the model's own processor
                    try:
                        vision_tower = self.model.get_vision_tower()
                        processor = getattr(vision_tower, "image_processor", None)
                        if processor is None:
                            raise AttributeError("FastVLM vision tower missing image_processor")
                        px = processor(images=pil_img, return_tensors="pt")["pixel_values"]
                        px = px.to(self.model.device, dtype=self.model.dtype)
                    except Exception as e:
                        self.logger.error(f"Failed to prepare image for FastVLM: {e}")
                        raise

                    prompt_token_count = input_ids.shape[1]
                    usage.prompt_tokens += prompt_token_count

                    input_ids = input_ids.to(self.model.device)
                else:
                    # No image present; fall back to text-only path
                    input_ids = self.tokenizer.apply_chat_template(
                        prepared_messages,
                        tokenize=True,
                        return_tensors="pt",
                        enable_thinking=self.enable_thinking,
                        add_generation_prompt=True,
                    )
                    prompt_token_count = input_ids.shape[1]
                    usage.prompt_tokens += prompt_token_count
                    input_ids = input_ids.to(self.model.device)
            else:
                # Non-FastVLM path (text-only or models without special vision handling)
                if self.model_name != "kyutai/helium-1-2b":
                    input_ids = self.tokenizer.apply_chat_template(
                        prepared_messages,
                        tokenize=True,
                        return_tensors="pt",
                        enable_thinking=self.enable_thinking,
                        add_generation_prompt=True,
                    )
                else:
                    messages_str = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in prepared_messages]
                    )

                    input_ids = self.tokenizer(
                        messages_str,
                        return_tensors="pt",
                    )["input_ids"]

                prompt_token_count = input_ids.shape[1]
                usage.prompt_tokens += prompt_token_count

                input_ids = input_ids.to(self.model.device)

            max_tokens = prepared_kwargs.get("max_completion_tokens", self.max_tokens)
            temperature = prepared_kwargs.get("temperature", self.temperature)
            top_p = prepared_kwargs.get("top_p", self.top_p)
            min_p = prepared_kwargs.get("min_p", self.min_p)
            repetition_penalty = prepared_kwargs.get("repetition_penalty", self.repetition_penalty)
            do_sample = prepared_kwargs.get("do_sample", self.do_sample)

            # Generate response
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_dict_in_generate": True,
                    "output_logits": True,
                }
                
                # Only add min_p and repetition_penalty if they're not default values
                if min_p > 0.0:
                    generation_kwargs["min_p"] = min_p
                if repetition_penalty != 1.0:
                    generation_kwargs["repetition_penalty"] = repetition_penalty
                
                # Add Nemotron-specific parameters if present
                if "max_thinking_tokens" in prepared_kwargs:
                    generation_kwargs["max_thinking_tokens"] = prepared_kwargs["max_thinking_tokens"]
                
                if px is not None:
                    gen_out = self.model.generate(
                        inputs=input_ids,
                        images=px,
                        **generation_kwargs,
                    )
                else:
                    gen_out = self.model.generate(
                        input_ids,
                        **generation_kwargs
                    )

                # Extract token IDs for the completion
                output_ids = gen_out.sequences[0]
                completion_ids = output_ids[prompt_token_count:]

                completion_text = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )

                # Enhanced prefix removal for assistant responses
                cleaned_text = completion_text.lstrip()
                assistant_prefixes = [
                    "assistant:",
                    "assistant ",
                    "ASSISTANT:",
                    "ASSISTANT ",
                    "<assistant>",
                    "assistant\n\n",
                ]

                for prefix in assistant_prefixes:
                    if cleaned_text.lower().startswith(prefix.lower()):
                        # Remove the prefix and any whitespace after it
                        cleaned_text = cleaned_text[len(prefix) :].lstrip()
                        break

                # If we're left with nothing after removing prefixes, use the original text
                completion_text = cleaned_text if cleaned_text else completion_text
                
                # Handle Hunyuan model special thinking format
                hunyuan_parsed = None
                if self._is_hunyuan_model() and self.enable_thinking:
                    hunyuan_parsed = self._parse_hunyuan_response(completion_text)
                    # Store thinking content for later access
                    self._last_hunyuan_thinking = hunyuan_parsed['thinking']
                    # Use the answer content as the main response if available
                    if hunyuan_parsed['answer']:
                        completion_text = hunyuan_parsed['answer']
                    # If there's thinking content but no answer tags, use the full response
                    elif hunyuan_parsed['thinking'] and not hunyuan_parsed['answer']:
                        # This might happen if the model only outputs thinking without answer tags
                        completion_text = completion_text

                # Parse tool calls if present in the completion
                if self.return_tools:
                    # Simple regex-based tool call extraction (this is a simplification)
                    # In a real implementation, you would use a more robust parser
                    import re

                    tool_call_pattern = r"<tool>(.*?)</tool>"
                    tool_matches = re.findall(tool_call_pattern, completion_text)
                    if tool_matches:
                        tool_data = [{"content": match} for match in tool_matches]
                        tools.append(tool_data)
                        # Remove the tool calls from the completion text
                        completion_text = re.sub(
                            tool_call_pattern, "", completion_text
                        ).strip()

                # Handle stop sequences
                stop_sequences = prepared_kwargs.get(
                    "stop", ["<|end_of_text|>", "</s>", "<|eot_id|>"]
                )
                for s in stop_sequences:
                    if s in completion_text:
                        completion_text = completion_text.split(s, 1)[0]
                        break

                completion_token_count = len(completion_ids)
                usage.completion_tokens += completion_token_count

                responses.append(completion_text)
                done_reasons.append(
                    "stop_string"
                    if any(s in completion_text for s in stop_sequences)
                    else "max_tokens"
                )

        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

        if self.return_tools:
            return responses, usage, done_reasons, tools
        else:
            if self.local:
                return responses, usage, done_reasons
            else:
                return responses, usage

    def embed(self, content: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for the given text content.

        Args:
            content: Text string or list of text strings to embed
            **kwargs: Additional kwargs to pass to the embedding model

        Returns:
            List[List[float]]: List of embedding vectors
        """
        # If a single string is provided, wrap it in a list
        if isinstance(content, str):
            content = [content]

        # Use the dedicated embedding model if available, otherwise use the main model
        model = self.embedding_model if self.embedding_model else self.model
        tokenizer = (
            self.embedding_tokenizer if self.embedding_tokenizer else self.tokenizer
        )

        try:
            # Tokenize the input text
            inputs = tokenizer(
                content, padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

                # Different models have different output formats for embeddings
                if hasattr(outputs, "pooler_output"):
                    # BERT-like models use pooler_output for the [CLS] token embedding
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    # For other models, use the mean of the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    raise ValueError(
                        "Model output format not recognized for embeddings."
                    )

                # Normalize embeddings (optional, but often helpful)
                embeddings = F.normalize(embeddings, p=2, dim=1)

                # Convert to list of lists
                embeddings_list = embeddings.cpu().tolist()

                return embeddings_list

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def get_sequence_probs(
        self, sequence: Union[str, List[int]], **kwargs
    ) -> Dict[str, Any]:
        """
        Compute log probabilities for each token in a sequence by performing
        a single forward pass through the model.

        Args:
            sequence: A string or list of token IDs to compute probabilities for
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dict[str, Any]: Dictionary containing token IDs, tokens, and their log probabilities
                {
                    'tokens': List of decoded tokens,
                    'token_ids': List of token IDs,
                    'log_probs': List of log probabilities for each token,
                    'top_tokens': (Optional) List of top token predictions for each position,
                    'top_token_probs': (Optional) List of probabilities for top token predictions
                }
        """
        try:
            # Convert to token IDs if input is a string
            if isinstance(sequence, str):
                input_ids = self.tokenizer.encode(sequence, return_tensors="pt")
            else:
                input_ids = torch.tensor([sequence], dtype=torch.long)

            # Move input tokens to the correct device
            input_ids = input_ids.to(self.model.device)

            # Get number of tokens to handle in results
            seq_len = input_ids.shape[1]

            # Get tokens corresponding to the IDs
            tokens = [
                self.tokenizer.decode(token_id.item()) for token_id in input_ids[0]
            ]

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    output_logits=True,
                    echo=True,
                    max_tokens=1,
                    return_dict=True,
                )

                # Get the logits
                logits = outputs.logits

                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)

                # Extract log probabilities for the selected tokens (excluding the last position)
                log_probs = []

                # For each position (excluding the last one), get probability of the next token
                for i in range(seq_len - 1):
                    next_token_id = input_ids[0, i + 1].item()
                    next_token_prob = probs[0, i, next_token_id].item()
                    next_token_log_prob = torch.log(probs[0, i, next_token_id]).item()
                    log_probs.append(next_token_log_prob)

                # Add None for the last position since we don't have the next token
                log_probs.append(None)

                # Collect results
                result = {
                    "tokens": tokens,
                    "token_ids": input_ids[0].cpu().tolist(),
                    "log_probs": log_probs,
                }

                # Optionally provide top predictions at each position
                if kwargs.get("return_top_tokens", False):
                    top_k = kwargs.get("top_k", 5)
                    top_tokens = []
                    top_token_probs = []

                    for i in range(seq_len):
                        # Get top-k token predictions at this position
                        topk_values, topk_indices = torch.topk(probs[0, i], top_k)

                        # Convert to tokens and probabilities
                        position_top_tokens = [
                            self.tokenizer.decode(idx.item()) for idx in topk_indices
                        ]
                        position_top_probs = topk_values.cpu().tolist()

                        top_tokens.append(position_top_tokens)
                        top_token_probs.append(position_top_probs)

                    result["top_tokens"] = top_tokens
                    result["top_token_probs"] = top_token_probs

                return result

        except Exception as e:
            self.logger.error(f"Error computing sequence probabilities: {e}")
            raise

