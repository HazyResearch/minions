"""
LMCache client for Minions - Local inference with KV cache management.

This client provides efficient KV cache offloading and sharing capabilities
using LMCache integrated with vLLM for improved performance with shared prefixes.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from minions.usage import Usage
from minions.clients.base import MinionsClient


class LMCacheClient(MinionsClient):
    """
    LMCache client for local inference with advanced KV cache management.
    
    This client integrates LMCache with vLLM to provide:
    - KV cache offloading to CPU/storage
    - Cache sharing across multiple inference requests
    - Performance improvements for shared prefixes
    - Configurable memory management
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_model_len: int = 8000,
        gpu_memory_utilization: float = 0.8,
        chunk_size: int = 256,
        local_cpu: bool = True,
        max_local_cpu_size: float = 5.0,
        enable_prefix_caching: bool = False,
        config_file: Optional[str] = None,
        local: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the LMCache client.

        Args:
            model_name: The model identifier (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            max_model_len: Maximum sequence length for the model (default: 8000)
            gpu_memory_utilization: GPU memory utilization ratio (default: 0.8)
            chunk_size: Token chunk size for LMCache (default: 256)
            local_cpu: Enable CPU memory backend for cache offloading (default: True)
            max_local_cpu_size: Maximum CPU memory size in GB for caching (default: 5.0)
            enable_prefix_caching: Enable vLLM prefix caching alongside LMCache (default: False)
            config_file: Path to LMCache configuration file (optional)
            local: Mark as local inference client (default: True)
            verbose: Enable verbose logging (default: False)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            local=local,
            verbose=verbose,
            **kwargs
        )
        
        # LMCache-specific configuration
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.chunk_size = chunk_size
        self.local_cpu = local_cpu
        self.max_local_cpu_size = max_local_cpu_size
        self.enable_prefix_caching = enable_prefix_caching
        self.config_file = config_file
        
        # Set up logging
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # Lazy import and initialization
        self._llm = None
        self._sampling_params = None
        
        self.logger.info(f"LMCache client initialized for model: {model_name}")
    
    def _setup_lmcache_environment(self) -> None:
        """Configure LMCache environment variables."""
        env_vars = {
            "LMCACHE_CHUNK_SIZE": str(self.chunk_size),
            "LMCACHE_LOCAL_CPU": str(self.local_cpu).lower(),
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(self.max_local_cpu_size),
        }
        
        # Set config file if provided
        if self.config_file and os.path.exists(self.config_file):
            env_vars["LMCACHE_CONFIG_FILE"] = self.config_file
        
        for key, value in env_vars.items():
            os.environ[key] = value
            self.logger.info(f"Set {key}={value}")
    
    def _initialize_llm(self) -> None:
        """Initialize vLLM with LMCache integration."""
        if self._llm is not None:
            return
            
        try:
            from vllm import LLM, SamplingParams
            from vllm.config import KVTransferConfig
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            from lmcache.integration.vllm.utils import ENGINE_NAME
        except ImportError as e:
            raise ImportError(
                "LMCache dependencies not found. Please install with:\n"
                "pip install lmcache vllm\n"
                "For detailed installation instructions, see: https://docs.lmcache.ai/getting_started/installation.html"
            ) from e
        
        # Setup environment variables
        self._setup_lmcache_environment()
        
        # Configure KV cache transfer
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",  # Both store and load KV cache
        )
        
        try:
            # Initialize LLM with LMCache configuration
            self._llm = LLM(
                model=self.model_name,
                kv_transfer_config=ktc,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enable_prefix_caching=self.enable_prefix_caching,
            )
            
            # Initialize sampling parameters
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=0.95,
                max_tokens=self.max_tokens,
            )
            
            self.logger.info(f"Successfully initialized LMCache-enabled vLLM for {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LMCache vLLM: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions with LMCache KV cache management.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        if not messages:
            raise ValueError("Messages cannot be empty.")
        
        # Initialize LLM on first use
        if self._llm is None:
            self._initialize_llm()
        
        try:
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)
            
            # Update sampling parameters with any kwargs
            sampling_params = self._sampling_params
            if kwargs:
                temp_params = {}
                if "temperature" in kwargs:
                    temp_params["temperature"] = kwargs["temperature"]
                if "max_tokens" in kwargs:
                    temp_params["max_tokens"] = kwargs["max_tokens"]
                if "top_p" in kwargs:
                    temp_params["top_p"] = kwargs["top_p"]
                
                if temp_params:
                    from vllm import SamplingParams
                    sampling_params = SamplingParams(
                        temperature=temp_params.get("temperature", self.temperature),
                        top_p=temp_params.get("top_p", 0.95),
                        max_tokens=temp_params.get("max_tokens", self.max_tokens),
                    )
            
            # Generate response using vLLM with LMCache
            outputs = self._llm.generate([prompt], sampling_params)
            
            # Extract generated text and usage information
            responses = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            for output in outputs:
                if output.outputs:
                    generated_text = output.outputs[0].text
                    responses.append(generated_text)
                    
                    # Accumulate token usage
                    if hasattr(output, 'usage') and output.usage:
                        total_prompt_tokens += getattr(output.usage, 'prompt_tokens', 0)
                        total_completion_tokens += getattr(output.usage, 'completion_tokens', 0)
            
            # Create usage object
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )
            
            if not responses:
                responses = [""]  # Ensure we always return at least one response
            
            self.logger.info(f"Generated {len(responses)} responses with LMCache")
            return responses, usage
            
        except Exception as e:
            self.logger.error(f"Error during LMCache inference: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat messages to a single prompt string.
        
        This is a simple implementation - you may want to use proper chat templates
        for specific models.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")
        
        # Add assistant prompt for generation
        prompt_parts.append("<|assistant|>\n")
        
        return "".join(prompt_parts)
    
    def cleanup(self) -> None:
        """Clean up LMCache resources."""
        if self._llm is not None:
            try:
                from lmcache.v1.cache_engine import LMCacheEngineBuilder
                from lmcache.integration.vllm.utils import ENGINE_NAME
                
                LMCacheEngineBuilder.destroy(ENGINE_NAME)
                self.logger.info("LMCache engine destroyed successfully")
                
            except Exception as e:
                self.logger.warning(f"Error during LMCache cleanup: {e}")
            finally:
                self._llm = None
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of commonly supported models for LMCache.
        
        Returns:
            List[str]: List of model names that work well with LMCache
        """
        return [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ] 