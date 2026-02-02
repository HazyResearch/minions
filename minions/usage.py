from dataclasses import dataclass, field
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


@dataclass
class PromptUsage:
    """Track token usage separately for each of the 6 remote prompts that constitute μ_remote.
    
    The 6 prompts are:
    - advice: ADVICE_PROMPT - Initial guidance generation
    - decompose_r1: DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC - Round 1 job creation
    - decompose_r2plus: DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND - Later round job creation
    - synth_cot: REMOTE_SYNTHESIS_COT - Chain-of-thought reasoning
    - synth_json: REMOTE_SYNTHESIS_JSON - Structured JSON decision
    - synth_final: REMOTE_SYNTHESIS_FINAL - Forced final answer (when max rounds exhausted)
    """
    advice: Usage = field(default_factory=Usage)
    decompose_r1: Usage = field(default_factory=Usage)
    decompose_r2plus: Usage = field(default_factory=Usage)
    synth_cot: Usage = field(default_factory=Usage)
    synth_json: Usage = field(default_factory=Usage)
    synth_final: Usage = field(default_factory=Usage)
    
    @property
    def total(self) -> Usage:
        """Return the combined usage across all prompts."""
        return (self.advice + self.decompose_r1 + self.decompose_r2plus +
                self.synth_cot + self.synth_json + self.synth_final)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with per-prompt breakdown and total.
        
        The 'total' field ensures backward compatibility with code that reads
        usage.remote.total_tokens.
        """
        return {
            "advice": self.advice.to_dict(),
            "decompose_r1": self.decompose_r1.to_dict(),
            "decompose_r2plus": self.decompose_r2plus.to_dict(),
            "synth_cot": self.synth_cot.to_dict(),
            "synth_json": self.synth_json.to_dict(),
            "synth_final": self.synth_final.to_dict(),
            # Include total for backward compatibility
            **self.total.to_dict(),
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