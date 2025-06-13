from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tiktoken

@dataclass
class Usage: 
    completion_tokens: int = 0
    prompt_tokens: int = 0

    # Some clients explicitly tell us whether or not we are using cached 
    # prompt tokens and being charged for less for them. This is distinct from 
    # seen_prompt_tokens since we don't know exactly how they determine if there's
    # a cache hit
    cached_prompt_tokens: int = 0
    
    # We keep track of the prompt tokens that have been seen in the 
    # conversation history.
    seen_prompt_tokens: int = 0

    @property
    def new_prompt_tokens(self) -> int:
        if self.seen_prompt_tokens is None:
            return self.prompt_tokens
        return self.prompt_tokens - self.seen_prompt_tokens
    
    @property
    def total_tokens(self) -> int:
        return self.completion_tokens + self.prompt_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            seen_prompt_tokens=self.seen_prompt_tokens + other.seen_prompt_tokens,
        )

    def get(self, key, default=None) -> Any:
        return self.to_dict().get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "seen_prompt_tokens": self.seen_prompt_tokens,
            "new_prompt_tokens": self.new_prompt_tokens,
        }




def num_tokens_from_messages_openai(
    messages: List[Dict[str, str]], 
    encoding: tiktoken.Encoding,
    include_reply_prompt: bool = False,
):
    """Return the number of tokens used by a list of messages.
    Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    # NOTE: this may change in the future
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    if include_reply_prompt:
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens