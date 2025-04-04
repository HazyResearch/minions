import logging
from typing import Any, Dict, List, Tuple

# wrap mlx_parallm import in a try catch
try:
    from mlx_parallm.utils import load, generate
except ImportError:
    raise ImportError(
        "mlx_parallm is not installed. Please install it using 'pip install git+https://github.com/akhileshvb/mlx_parallm.git'."
    )

from minions.usage import Usage


class MLXParallmClient:
    def __init__(
        self,
        model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        temperature: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
    ):
        """
        Initialize the MLX PARALLM client.

        Args:
            model_name: The name or identifier of the MLX PARALLM model.
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 100)
            verbose: Whether to print verbose output (default: False)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        self.logger = logging.getLogger("MLXParallmClient")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Loading MLX PARALLM model: {model_name}")

        self.model, self.tokenizer = load(model_name)
        self.logger.info(f"Model {model_name} loaded successfully")

    def chat(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, str]:
        """
        Generate a response for a chat conversation using the MLX PARALLM model.

        Args:
            messages: List of message dictionaries, each with 'role' and 'content' keys.
            **kwargs: Additional keyword arguments to pass to the generate function.

        Returns:
            A tuple containing:
              - List of response strings.
              - Usage object with prompt and completion token counts.
        """
        assert len(messages) > 0, "Messages cannot be empty."

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, temp=self.temperature
        )

        params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "verbose": self.verbose,
            "temp": self.temperature,
            **kwargs,
        }

        response = generate(**params)

        # messages are structured as list of lists in run.py
        # extract prompt tokens across all messages
        prompt_tokens = sum(len(prompt_i) for prompt_i in prompt)

        try:
            encoded = self.tokenizer.encode(response)
            completion_tokens = len(encoded)
        except Exception as e:
            self.logger.error(f"Error during token encoding: {e}")
            completion_tokens = len(response)

        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

        return [response], usage, "END_OF_TEXT"
