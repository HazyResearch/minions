import os
import pytest
from unittest.mock import patch, MagicMock

from minions.clients.notdiamond import NotDiamondAIClient


@pytest.fixture
def mock_openai_response():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20
    )
    return mock_response


def test_get_supported_providers():
    providers_str = NotDiamondAIClient.get_supported_providers()
    assert "Supported AI Providers:" in providers_str
    assert "OpenAI" in providers_str
    assert "Anthropic" in providers_str
    assert "Google" in providers_str
    assert "Mistral AI" in providers_str
    assert "Together AI" in providers_str
    assert "Perplexity AI" in providers_str
    
    # Verify format
    lines = providers_str.split("\n")
    assert len(lines) == 7  # Header + 6 providers
    for line in lines[1:]:  # Skip header
        assert line.startswith("- ")


def test_notdiamond_client_initialization():
    client = NotDiamondAIClient(
        api_key="test_key",
        models=["gpt-4", "claude-3"],
        tradeoff="cost",
        preference_id="test_pref"
    )
    
    assert client.api_key == "test_key"
    assert client.models == ["gpt-4", "claude-3"]
    assert client.tradeoff == "cost"
    assert client.preference_id == "test_pref"


def test_notdiamond_client_initialization_with_env_vars():
    with patch.dict(os.environ, {"NOTDIAMOND_API_KEY": "env_test_key"}):
        client = NotDiamondAIClient()
        assert client.api_key == "env_test_key"
        assert client.models == ["gpt-4", "claude-3-5-sonnet"]
        assert client.tradeoff == "cost"
        assert client.preference_id is None


@patch("openai.OpenAI")
def test_notdiamond_chat_completion(mock_openai, mock_openai_response):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client

    client = NotDiamondAIClient(api_key="test_key")
    messages = [{"role": "user", "content": "Hello"}]
    
    response, usage = client.chat(messages)
    
    # Verify the response
    assert response == ["Test response"]
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    
    # Verify the API call
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    
    assert call_kwargs["model"] == "notdiamond"
    assert call_kwargs["messages"] == messages
    assert "extra_body" in call_kwargs
    assert call_kwargs["extra_body"]["models"] == ["gpt-4", "claude-3-5-sonnet"]
    assert call_kwargs["extra_body"]["tradeoff"] == "cost"


@patch("openai.OpenAI")
def test_notdiamond_chat_completion_with_override_params(mock_openai, mock_openai_response):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client

    client = NotDiamondAIClient(
        api_key="test_key",
        models=["model1", "model2"],
        tradeoff="cost",
        preference_id="default_pref"
    )
    
    messages = [{"role": "user", "content": "Hello"}]
    override_models = ["gpt-4", "claude-3"]
    override_tradeoff = "latency"
    override_pref = "override_pref"
    
    response, usage = client.chat(
        messages,
        models=override_models,
        tradeoff=override_tradeoff,
        preference_id=override_pref
    )
    
    # Verify the API call parameters
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["extra_body"]["models"] == override_models
    assert call_kwargs["extra_body"]["tradeoff"] == override_tradeoff
    assert call_kwargs["extra_body"]["preference_id"] == override_pref


def test_notdiamond_empty_messages():
    client = NotDiamondAIClient(api_key="test_key")
    with pytest.raises(AssertionError, match="Messages cannot be empty."):
        client.chat([]) 