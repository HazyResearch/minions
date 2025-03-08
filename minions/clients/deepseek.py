from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
import logging
import os
import openai 

class DeepSeekClient:
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.deepseek.com",
    ):
        '''
        Initialize the DeepSeek client.

        Args:
            model_name: The name of the model to use (default: "deepseek-chat")
            api_key: DeepSeek API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
        '''
        self.model_name = model_name
        openai.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.logger = logging.getLogger("DeepSeekClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        '''
        Handle chat completions using the DeepSeek API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to deepseek.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        '''
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            # Only add temperature if NOT using the reasoning models
            if "reasoner" not in self.model_name:
                params["temperature"] = self.temperature

            client = openai.OpenAI(api_key=openai.api_key, base_url=self.base_url)
            response = client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during DeepSeek API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage
