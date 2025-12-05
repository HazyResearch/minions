from typing import Any, Dict, List, Optional, Tuple
import os
from together import Together

from minions.usage import Usage
from minions.clients.base import MinionsClient


class TogetherClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        local: bool = False,
        **kwargs
    ):
        """
        Initialize the Together client.

        Args:
            model_name: The name of the model to use (default: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
            api_key: Together API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            local=local,
            **kwargs
        )
        
        # Client-specific configuration
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.client = Together(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Together API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.chat.completions.create

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Together API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
        # Extract done reasons (finish_reason in OpenAI-compatible APIs)
        done_reasons = [choice.finish_reason for choice in response.choices]
        return [choice.message.content for choice in response.choices], usage, done_reasons

    def get_sequence_probs(
        self,
        prompt: str,
        completion: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute log probabilities for the prompt and optional completion tokens
        using the Together completions API.

        Args:
            prompt: The prompt text to compute probabilities for
            completion: Optional completion text to compute probabilities for
            **kwargs: Additional arguments:
                - logprobs: Number of top logprobs to return per token (default: 1)
                - model: Override the default model

        Returns:
            Dict[str, Any]: Dictionary containing token log probabilities
                {
                    'prompt_tokens': List of prompt tokens,
                    'prompt_logprobs': List of log probabilities for prompt tokens,
                    'completion_tokens': List of completion tokens (if completion provided),
                    'completion_logprobs': List of log probabilities for completion tokens,
                    'top_logprobs': (Optional) List of top logprob dicts per position
                }
        """
        try:
            # Build the full text (prompt + optional completion)
            full_text = prompt
            if completion:
                full_text = prompt + completion

            logprobs = kwargs.get("logprobs", 1)
            model = kwargs.get("model", self.model_name)

            # Get logprobs for the full sequence using echo=True
            response = self.client.completions.create(
                model=model,
                prompt=full_text,
                max_tokens=0,
                echo=True,
                logprobs=logprobs,
            )

            # Extract the response data
            choice = response.choices[0]
            
            result = {
                'prompt_tokens': [],
                'prompt_logprobs': [],
                'completion_tokens': [],
                'completion_logprobs': [],
            }

            # Get tokens and logprobs from the response
            if hasattr(choice, 'logprobs') and choice.logprobs:
                tokens = choice.logprobs.tokens if hasattr(choice.logprobs, 'tokens') else []
                token_logprobs = choice.logprobs.token_logprobs if hasattr(choice.logprobs, 'token_logprobs') else []
                
                # If completion was provided, we need to split tokens between prompt and completion
                if completion:
                    # Find approximate split point by tokenizing prompt separately
                    prompt_response = self.client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=0,
                        echo=True,
                        logprobs=logprobs,
                    )
                    prompt_choice = prompt_response.choices[0]
                    
                    if hasattr(prompt_choice, 'logprobs') and prompt_choice.logprobs:
                        prompt_token_count = len(prompt_choice.logprobs.tokens) if hasattr(prompt_choice.logprobs, 'tokens') else 0
                        
                        # Split tokens and logprobs
                        result['prompt_tokens'] = tokens[:prompt_token_count]
                        result['prompt_logprobs'] = token_logprobs[:prompt_token_count]
                        result['completion_tokens'] = tokens[prompt_token_count:]
                        result['completion_logprobs'] = token_logprobs[prompt_token_count:]
                else:
                    # No completion, all tokens are prompt tokens
                    result['prompt_tokens'] = tokens
                    result['prompt_logprobs'] = token_logprobs

                # Include top logprobs if available and requested
                if logprobs > 1 and hasattr(choice.logprobs, 'top_logprobs'):
                    result['top_logprobs'] = choice.logprobs.top_logprobs

            return result

        except Exception as e:
            self.logger.error(f"Error computing sequence probabilities: {e}")
            raise