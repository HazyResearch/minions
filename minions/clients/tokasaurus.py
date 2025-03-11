import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from openai import OpenAI

from minions.usage import Usage
from minions.clients.utils import ServerMixin


# TODO: define one dataclass for what is returned from all the clients
class TokasaurusClient(ServerMixin):
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        port: Optional[int] = None,
        capture_output: bool = False,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Tokasaurus client.

        Args:
            model_name: The name of the model to use (default: "meta-llama/Llama-3.2-1B-Instruct")
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            port: Port to use for the server (optional, will find a free port if not provided)
            capture_output: Whether to capture server output (default: False)
            base_url: Base URL for the Tokasaurus API (optional, falls back to TOKASAURUS_BASE_URL environment variable or default URL)
        """
        self.model_name = model_name
        self.logger = logging.getLogger("OpenAIClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens

        if port is None:
            self.port = self.find_free_port()
            launch_command = f"""tksrs \
            port={self.port} \
            model={model_name} \
            torch_compile=T \
            """
            self.launch_server(launch_command, self.port, capture_output=capture_output)
        else:
            self.port = port
        
        # Get base URL from parameter, environment variable, or construct default from port
        default_base_url = f"http://0.0.0.0:{self.port}/v1"
        base_url = base_url or os.getenv("TOKASAURUS_BASE_URL", default_base_url)
        
        self.client = OpenAI(
            api_key='fake-key',
            base_url=base_url
        )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:

            # Only add temperature if NOT using the reasoning models (e.g., o3-mini model)
            if "o1" not in self.model_name and "o3" not in self.model_name:
                kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        except Exception as e:
            self.logger.error(f"Error during OpenAI API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage, ""
