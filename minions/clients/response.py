"""Standardized response types for all Minions clients."""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from minions.usage import Usage


@dataclass(frozen=True)
class ChatResponse:
    """
    Standardized return type for all client chat() methods.

    This class provides backward compatibility with tuple unpacking
    while offering type-safe attribute access for new code.

    Attributes:
        responses: List of generated response texts
        usage: Token usage information
        done_reasons: Optional list of finish reasons (one per response)
        tool_calls: Optional list of tool call objects
        audio: Optional audio data as bytes (for multimodal models)
        metadata: Optional additional metadata dictionary

    Examples:
        # New code (type-safe):
        response = client.chat(messages)
        print(response.responses[0])
        print(response.usage.total_tokens)

        # Old code (backward compatible):
        responses, usage = client.chat(messages)
        print(responses[0])
        print(usage.total_tokens)

        # 3-tuple unpacking:
        responses, usage, done_reasons = client.chat(messages)
    """

    responses: List[str]
    usage: Usage
    done_reasons: Optional[List[str]] = None
    tool_calls: Optional[List[Any]] = None
    audio: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None

    def __iter__(self) -> Iterator:
        """
        Allow unpacking for backward compatibility.

        Yields elements based on which optional fields are populated:
        - Always yields: responses, usage
        - If done_reasons is not None: yields done_reasons
        - If tool_calls is not None: yields tool_calls

        This enables:
            responses, usage = chat_response  # 2-tuple
            responses, usage, done_reasons = chat_response  # 3-tuple
            responses, usage, done_reasons, tools = chat_response  # 4-tuple
        """
        yield self.responses
        yield self.usage
        if self.done_reasons is not None:
            yield self.done_reasons
        if self.tool_calls is not None:
            yield self.tool_calls

    def __getitem__(self, index):
        """
        Allow indexing and slicing for backward compatibility.

        Supports: chat_response[0] = responses, chat_response[1] = usage, etc.
        Also supports slicing: chat_response[:2], chat_response[1:], etc.
        """
        # Handle slice objects
        if isinstance(index, slice):
            return self.to_tuple()[index]

        # Handle integer indices
        if index == 0:
            return self.responses
        elif index == 1:
            return self.usage
        elif index == 2:
            if self.done_reasons is not None:
                return self.done_reasons
            raise IndexError(f"ChatResponse has no element at index {index}")
        elif index == 3:
            if self.tool_calls is not None:
                return self.tool_calls
            raise IndexError(f"ChatResponse has no element at index {index}")
        else:
            raise IndexError(f"ChatResponse index out of range: {index}")

    def to_tuple(self) -> Tuple:
        """
        Convert to tuple format for maximum compatibility.

        Returns variable-length tuple based on populated fields:
        - (responses, usage) if only required fields
        - (responses, usage, done_reasons) if done_reasons populated
        - (responses, usage, done_reasons, tool_calls) if tool_calls populated
        """
        if self.tool_calls is not None:
            return (self.responses, self.usage, self.done_reasons, self.tool_calls)
        elif self.done_reasons is not None:
            return (self.responses, self.usage, self.done_reasons)
        else:
            return (self.responses, self.usage)

    def __len__(self) -> int:
        """Return the effective tuple length for this response."""
        if self.tool_calls is not None:
            return 4
        elif self.done_reasons is not None:
            return 3
        else:
            return 2
