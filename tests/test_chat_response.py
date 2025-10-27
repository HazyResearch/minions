"""Tests for ChatResponse dataclass and backward compatibility."""

import unittest
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.response import ChatResponse
from minions.usage import Usage


class TestChatResponse(unittest.TestCase):
    """Test ChatResponse dataclass and backward compatibility."""

    def test_create_basic_response(self):
        """Test creating a basic ChatResponse with required fields."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hello world"],
            usage=usage
        )

        self.assertEqual(response.responses, ["Hello world"])
        self.assertEqual(response.usage, usage)
        self.assertIsNone(response.done_reasons)
        self.assertIsNone(response.tool_calls)
        self.assertIsNone(response.audio)

    def test_create_response_with_done_reasons(self):
        """Test ChatResponse with finish reasons."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hello"],
            usage=usage,
            done_reasons=["stop"]
        )

        self.assertEqual(response.done_reasons, ["stop"])

    def test_create_response_with_tool_calls(self):
        """Test ChatResponse with tool calls."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        tool_call = {"name": "search", "args": {}}
        response = ChatResponse(
            responses=["Result"],
            usage=usage,
            done_reasons=["tool_calls"],
            tool_calls=[tool_call]
        )

        self.assertEqual(response.tool_calls, [tool_call])

    def test_backward_compat_2_tuple_unpacking(self):
        """Test backward compatibility: unpack as 2-tuple."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(responses=["Hi"], usage=usage)

        # Should work like old code
        responses, usage_out = response

        self.assertEqual(responses, ["Hi"])
        self.assertEqual(usage_out, usage)

    def test_backward_compat_3_tuple_unpacking(self):
        """Test backward compatibility: unpack as 3-tuple."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["stop"]
        )

        # Should work like old code
        responses, usage_out, done_reasons = response

        self.assertEqual(responses, ["Hi"])
        self.assertEqual(usage_out, usage)
        self.assertEqual(done_reasons, ["stop"])

    def test_backward_compat_4_tuple_unpacking(self):
        """Test backward compatibility: unpack as 4-tuple."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        tool_call = {"name": "test"}
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["tool_calls"],
            tool_calls=[tool_call]
        )

        # Should work like old code
        responses, usage_out, done_reasons, tool_calls = response

        self.assertEqual(responses, ["Hi"])
        self.assertEqual(usage_out, usage)
        self.assertEqual(done_reasons, ["tool_calls"])
        self.assertEqual(tool_calls, [tool_call])

    def test_indexing_backward_compat(self):
        """Test backward compatibility: access by index."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["stop"]
        )

        self.assertEqual(response[0], ["Hi"])
        self.assertEqual(response[1], usage)
        self.assertEqual(response[2], ["stop"])

    def test_indexing_out_of_range(self):
        """Test that indexing beyond valid range raises IndexError."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(responses=["Hi"], usage=usage)

        with self.assertRaises(IndexError):
            _ = response[5]

    def test_to_tuple_2_elements(self):
        """Test to_tuple() with 2-element response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(responses=["Hi"], usage=usage)

        result = response.to_tuple()
        self.assertEqual(result, (["Hi"], usage))
        self.assertEqual(len(result), 2)

    def test_to_tuple_3_elements(self):
        """Test to_tuple() with 3-element response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["stop"]
        )

        result = response.to_tuple()
        self.assertEqual(result, (["Hi"], usage, ["stop"]))
        self.assertEqual(len(result), 3)

    def test_to_tuple_4_elements(self):
        """Test to_tuple() with 4-element response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        tool_call = {"name": "test"}
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["tool_calls"],
            tool_calls=[tool_call]
        )

        result = response.to_tuple()
        self.assertEqual(result, (["Hi"], usage, ["tool_calls"], [tool_call]))
        self.assertEqual(len(result), 4)

    def test_immutable_after_creation(self):
        """Test that ChatResponse is immutable (frozen dataclass)."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(responses=["Hi"], usage=usage)

        # Try to modify - should raise FrozenInstanceError
        with self.assertRaises(Exception):  # dataclasses.FrozenInstanceError
            response.responses = ["Changed"]

    def test_len_2_tuple(self):
        """Test __len__ for 2-tuple response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(responses=["Hi"], usage=usage)

        self.assertEqual(len(response), 2)

    def test_len_3_tuple(self):
        """Test __len__ for 3-tuple response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["stop"]
        )

        self.assertEqual(len(response), 3)

    def test_len_4_tuple(self):
        """Test __len__ for 4-tuple response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        tool_call = {"name": "test"}
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            done_reasons=["tool_calls"],
            tool_calls=[tool_call]
        )

        self.assertEqual(len(response), 4)

    def test_metadata_field(self):
        """Test that metadata field can store additional information."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            metadata={"reasoning": "Some reasoning content"}
        )

        self.assertEqual(response.metadata, {"reasoning": "Some reasoning content"})

    def test_audio_field(self):
        """Test that audio field can store bytes."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        audio_data = b"fake audio bytes"
        response = ChatResponse(
            responses=["Hi"],
            usage=usage,
            audio=audio_data
        )

        self.assertEqual(response.audio, audio_data)

    def test_attribute_access(self):
        """Test type-safe attribute access."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response = ChatResponse(
            responses=["Hello world"],
            usage=usage,
            done_reasons=["stop"]
        )

        # New code should use attributes for type safety
        self.assertEqual(response.responses[0], "Hello world")
        self.assertEqual(response.usage.total_tokens, 30)
        self.assertEqual(response.done_reasons[0], "stop")


if __name__ == '__main__':
    unittest.main()
