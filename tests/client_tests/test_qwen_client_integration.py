import pytest
import os
from minions.clients.qwen import QwenClient
from minions.usage import Usage


def test_qwen_client_initialization():
    """Test that QwenClient can be initialized with default values."""
    client = QwenClient()
    
    assert client.model_name == "qwen-plus"
    assert client.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert client.temperature == 0.0
    assert client.max_tokens == 4096
    assert client.local is False


def test_qwen_client_initialization_with_custom_values():
    """Test that QwenClient can be initialized with custom values."""
    client = QwenClient(
        model_name="qwen-turbo",
        api_key="test_key",
        temperature=0.5,
        max_tokens=2048,
        base_url="https://custom-url.com/v1"
    )
    
    assert client.model_name == "qwen-turbo"
    assert client.temperature == 0.5
    assert client.max_tokens == 2048
    assert client.base_url == "https://custom-url.com/v1"


@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY not set"
)
def test_qwen_client_chat():
    """Test that QwenClient can perform chat completion."""
    client = QwenClient()
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    response, usage = client.chat(messages)
    
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens >= 0
    assert usage.completion_tokens >= 0


@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY not set"
)
def test_qwen_client_chat_with_system_message():
    """Test that QwenClient can handle system messages."""
    client = QwenClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    response, usage = client.chat(messages)
    
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)
    assert isinstance(usage, Usage)


def test_qwen_client_chat_empty_messages():
    """Test that QwenClient raises assertion error for empty messages."""
    client = QwenClient()
    
    with pytest.raises(AssertionError):
        client.chat([])


if __name__ == "__main__":
    # Run a simple test if the API key is available
    if os.getenv("DASHSCOPE_API_KEY"):
        client = QwenClient()
        messages = [{"role": "user", "content": "Hello!"}]
        try:
            response, usage = client.chat(messages)
            print(f"Response: {response}")
            print(f"Usage: {usage}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("DASHSCOPE_API_KEY not set, skipping API test") 