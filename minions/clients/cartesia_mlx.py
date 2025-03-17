import logging
from typing import Any, Dict, List, Optional, Tuple, Iterator

import cartesia_mlx as cmx
import mlx.core as mx
from minions.usage import Usage
from transformers import AutoTokenizer


class CartesiaMLXClient:
    def __init__(
        self,
        model_name: str = "cartesia-ai/Llamba-1B-4bit-mlx",
        temperature: float = 0.001,
        max_tokens: int = 100,
        verbose: bool = False,
        dtype: str = "float32",
    ):
        """
        Initialize the Cartesia MLX client.

        Args:
            model_name: The name or path of the model to use (default: "cartesia-ai/Llamba-1B-4bit-mlx")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 1000)
            verbose: Whether to print tokens and timing information (default: False)
            dtype: Data type for model computation (default: "float32")
        """
        self.model_name = model_name
        self.logger = logging.getLogger("CartesiaMLXClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.dtype = dtype

        # Load the model
        self.logger.info(f"Loading Cartesia MLX model: {model_name}")
        self.model = cmx.from_pretrained(model_name)

        # if Llamba-1B in model name, set tokenizer to meta-llama/Llama-3.2-3B-Instruct
        if "Llamba-1B" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct"
            )
        elif "Llamba-3B" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct"
            )
        elif "Llamba-8B" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct"
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # set EOS token
        self.model.eos = self.tokenizer.eos_token_id
        # Set the data type
        if dtype == "float32":
            self.model.set_dtype(mx.float32)
            self.dtype = mx.float32
        elif dtype == "float16":
            self.model.set_dtype(mx.float16)
            self.dtype = mx.float16
        elif dtype == "bfloat16":
            self.model.set_dtype(mx.bfloat16)
            self.dtype = mx.bfloat16
        self.logger.info(f"Model {model_name} loaded successfully with dtype {dtype}")

    def chat(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, str]:
        """
        Handle chat completions using the Cartesia MLX client.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, str) containing response strings, token usage, and finish reason
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                temp=self.temperature,
            )

            # decode prompt
            prompt_toks = self.tokenizer.decode(prompt)
            # Set default parameters
            params = {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "verbose": self.verbose,
                **kwargs,
            }
            return self._generate(prompt, params)

        except Exception as e:
            self.logger.error(f"Error during Cartesia MLX generation: {e}")
            raise

    def _generate(
        self, prompt_ids: List[int], params: Dict[str, Any]
    ) -> Tuple[List[str], Usage, str]:
        """Generate text without streaming."""
        # Set eval_every_n to max_tokens to get the final result at once
        prompt_mlx_tensor = mx.array(prompt_ids)

        tokens = []
        for n_tokens, token in enumerate(
            self.model.generate_tokens(prompt_mlx_tensor, **params)
        ):
            tokens.append(token)

        output_text = self.tokenizer.decode(tokens)

        prompt_tokens = len(prompt_ids)
        completion_tokens = len(tokens)

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return [output_text], usage, "stop"
