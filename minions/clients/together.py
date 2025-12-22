from typing import Any, Dict, List, Optional, Tuple
import os
# CHANGE: Import TogetherError for v2 exception handling
from together import Together, TogetherError

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
        # v2.0: Client instantiation remains the same
        self.client = Together(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Together API.
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # v2.0: Enforce keyword arguments via dictionary unpacking
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
            
        # CHANGE: Catch specific TogetherError instead of generic Exception
        except TogetherError as e:
            self.logger.error(f"Together API Error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during Together call: {e}")
            raise

        # v2.0: Response is a Pydantic model, so attribute access (.usage) is correct
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
        # v2.0: .finish_reason access is correct
        done_reasons = [choice.finish_reason for choice in response.choices]
        return [choice.message.content for choice in response.choices], usage, done_reasons

    def get_sequence_probs(
        self,
        prompt: str,
        completion: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute log probabilities using the Together completions API.
        """
        try:
            full_text = prompt
            if completion:
                full_text = prompt + completion

            logprobs = kwargs.get("logprobs", 1)
            model = kwargs.get("model", self.model_name)

            # v2.0: Ensure all arguments are passed as keywords
            response = self.client.completions.create(
                model=model,
                prompt=full_text,
                max_tokens=0,
                echo=True,
                logprobs=logprobs,
            )

            choice = response.choices[0]
            
            result = {
                'prompt_tokens': [],
                'prompt_logprobs': [],
                'completion_tokens': [],
                'completion_logprobs': [],
            }

            # v2.0: Pydantic object access for logprobs
            if hasattr(choice, 'logprobs') and choice.logprobs:
                # Access attributes safely
                tokens = getattr(choice.logprobs, 'tokens', [])
                token_logprobs = getattr(choice.logprobs, 'token_logprobs', [])
                
                if completion:
                    # Helper call to split prompt/completion
                    # CHANGE: Ensure kwargs usage here as well
                    prompt_response = self.client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=0,
                        echo=True,
                        logprobs=logprobs,
                    )
                    prompt_choice = prompt_response.choices[0]
                    
                    if hasattr(prompt_choice, 'logprobs') and prompt_choice.logprobs:
                        prompt_tokens_list = getattr(prompt_choice.logprobs, 'tokens', [])
                        prompt_token_count = len(prompt_tokens_list)
                        
                        result['prompt_tokens'] = tokens[:prompt_token_count]
                        result['prompt_logprobs'] = token_logprobs[:prompt_token_count]
                        result['completion_tokens'] = tokens[prompt_token_count:]
                        result['completion_logprobs'] = token_logprobs[prompt_token_count:]
                else:
                    result['prompt_tokens'] = tokens
                    result['prompt_logprobs'] = token_logprobs

                if logprobs > 1 and hasattr(choice.logprobs, 'top_logprobs'):
                    result['top_logprobs'] = choice.logprobs.top_logprobs

            return result

        # CHANGE: Specific error handling
        except TogetherError as e:
            self.logger.error(f"Together API Error in get_sequence_probs: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error computing sequence probabilities: {e}")
            raise