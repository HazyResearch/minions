"""
Tests for Moonshot AI client integration.
"""

import os
import pytest
from unittest.mock import Mock, patch
from minions.clients.moonshot import MoonshotClient
from minions.usage import Usage


class TestMoonshotClientIntegration:
    """Test suite for Moonshot AI client integration."""

    def test_moonshot_client_initialization(self):
        """Test that MoonshotClient initializes correctly."""
        client = MoonshotClient(
            model_name="moonshot-v1-8k",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1024,
        )
        
        assert client.model_name == "moonshot-v1-8k"
        assert client.api_key == "test-key"
        assert client.temperature == 0.7
        assert client.max_tokens == 1024
        assert client.base_url == "https://api.moonshot.cn/v1"

    def test_moonshot_client_env_var_api_key(self):
        """Test that MoonshotClient uses environment variable for API key."""
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "env-api-key"}):
            client = MoonshotClient()
            assert client.api_key == "env-api-key"

    def test_moonshot_client_custom_base_url(self):
        """Test that MoonshotClient accepts custom base URL."""
        custom_url = "https://custom.moonshot.ai/v1"
        client = MoonshotClient(base_url=custom_url)
        assert client.base_url == custom_url

    @patch("openai.OpenAI")
    def test_moonshot_client_chat_success(self, mock_openai):
        """Test successful chat completion with Moonshot API."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Hello! How can I help you today?"), finish_reason="stop")
        ]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test client
        client = MoonshotClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        responses, usage = client.chat(messages)
        
        assert len(responses) == 1
        assert responses[0] == "Hello! How can I help you today?"
        assert isinstance(usage, Usage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 15

    @patch("openai.OpenAI")
    def test_moonshot_client_chat_with_local_flag(self, mock_openai):
        """Test chat completion with local flag returns finish reasons."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Hello! How can I help you today?"), finish_reason="stop")
        ]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test client with local=True
        client = MoonshotClient(api_key="test-key", local=True)
        messages = [{"role": "user", "content": "Hello"}]
        
        responses, usage, finish_reasons = client.chat(messages)
        
        assert len(responses) == 1
        assert responses[0] == "Hello! How can I help you today?"
        assert isinstance(usage, Usage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 15
        assert finish_reasons == ["stop"]

    @patch("openai.OpenAI")
    def test_moonshot_client_chat_with_kwargs(self, mock_openai):
        """Test chat completion with additional kwargs."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Response"), finish_reason="stop")
        ]
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=10)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test client
        client = MoonshotClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        # Call with additional kwargs
        responses, usage = client.chat(messages, stream=False, top_p=0.9)
        
        # Verify the OpenAI client was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        
        assert call_args["model"] == "moonshot-v1-8k"
        assert call_args["messages"] == messages
        assert call_args["max_tokens"] == 4096
        assert call_args["temperature"] == 0.0
        assert call_args["stream"] == False
        assert call_args["top_p"] == 0.9

    def test_moonshot_client_chat_empty_messages(self):
        """Test that chat raises assertion error for empty messages."""
        client = MoonshotClient(api_key="test-key")
        
        with pytest.raises(AssertionError, match="Messages cannot be empty"):
            client.chat([])

    @patch("openai.OpenAI")
    def test_moonshot_client_chat_api_error(self, mock_openai):
        """Test chat completion handles API errors gracefully."""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Test client
        client = MoonshotClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="API Error"):
            client.chat(messages)

    def test_moonshot_client_default_model_options(self):
        """Test that client supports different Moonshot model options."""
        models = ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
        
        for model in models:
            client = MoonshotClient(model_name=model)
            assert client.model_name == model


if __name__ == "__main__":
    pytest.main([__file__]) 