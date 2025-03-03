import logging
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm import generate, load
from minions.usage import Usage


class MLXLMClient:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize the MLX LM client.

        Args:
            model_name: The name or path of the model to use (default: "mistralai/Mistral-7B-Instruct-v0.3")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 1000)
            verbose: Whether to print tokens and timing information (default: False)
        """
        self.model_name = model_name
        self.logger = logging.getLogger("MLXLMClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Load the model and tokenizer
        self.logger.info(f"Loading MLX LM model: {model_name}")
        self.model, self.tokenizer = load(path_or_hf_repo=model_name)
        self.logger.info(f"Model {model_name} loaded successfully")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the MLX LM client.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Apply the chat template to the messages
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, temp=self.temperature
            )

            # Generate response
            params = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "verbose": self.verbose,
                **kwargs,
            }

            response = generate(**params)

            # Since MLX LM doesn't provide token usage information directly,
            # we'll estimate it based on the input and output lengths
            prompt_tokens = len(prompt)
            completion_tokens = len(self.tokenizer.encode(response))

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            return [response], usage, "END_OF_TEXT"

        except Exception as e:
            self.logger.error(f"Error during MLX LM generation: {e}")
            raise
