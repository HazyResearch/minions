import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from mlx_lm import generate, load
from minions.usage import Usage


class MLXLMClient:
    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        verbose: bool = False,
        use_async: bool = False,
    ):
        """
        Initialize the MLX LM client.

        Args:
            model_name: The name or path of the model to use (default: "mlx-community/Llama-3.2-3B-Instruct")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 1000)
            verbose: Whether to print tokens and timing information (default: False)
            use_async: Whether to use async mode (default: False)
        """
        self.model_name = model_name
        self.logger = logging.getLogger("MLXLMClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.use_async = use_async

        # Load the model and tokenizer
        self.logger.info(f"Loading MLX LM model: {model_name}")
        self.model, self.tokenizer = load(path_or_hf_repo=model_name)
        self.logger.info(f"Model {model_name} loaded successfully")

    def _prepare_params(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare parameters for generation.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Dictionary of parameters for generation
        """
        # Apply the chat template to the messages
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, temp=self.temperature
        )

        # Generate response params
        params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "verbose": self.verbose,
            **kwargs,
        }

        return params, prompt

    def chat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the MLX LM client.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)

    def schat(
        self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]

        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params, prompt = self._prepare_params(messages, **kwargs)
            response = generate(**params)

            # Since MLX LM doesn't provide token usage information directly,
            # we'll estimate it based on the input and output lengths
            prompt_tokens = len(prompt)
            completion_tokens = len(self.tokenizer.encode(response))

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            return [response], usage, ["stop"]

        except Exception as e:
            self.logger.error(f"Error during MLX LM generation: {e}")
            raise

    def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )

        # Check if we're already in an event loop
        try:
            print("Checking if we're already in an event loop")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                print("We're in a running event loop")
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

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     or a single message dictionary
            **kwargs: Additional arguments to pass to generate function

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings,
            token usage, and completion reasons
        """
        # If the user provided a single dictionary, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]

        assert len(messages) > 0, "Messages cannot be empty."

        async def process_one(msg):
            # We need to run the generation in a thread pool since MLX LM's generate
            # function is synchronous
            params, prompt = self._prepare_params([msg], **kwargs)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: generate(**params))
            print(response)

            prompt_tokens = len(prompt)
            completion_tokens = len(self.tokenizer.encode(response))

            return response, prompt_tokens, completion_tokens

        # Run tasks in parallel
        tasks = [process_one(m) for m in messages]
        results = await asyncio.gather(*tasks)

        # Gather results
        texts = []
        usage_total = Usage()
        done_reasons = []

        for response, prompt_tokens, completion_tokens in results:
            texts.append(response)
            usage_total += Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            done_reasons.append("stop")

        return texts, usage_total, done_reasons
