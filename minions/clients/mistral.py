import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from mistralai import Mistral

from minions.usage import Usage


class MistralClient:
    def __init__(
        self,
        model_name: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        websearch_agent: bool = False,
        websearch_premium: bool = False,
    ):
        """
        Initialize the Mistral client.

        Args:
            model_name: The name of the model to use (default: "mistral-large-latest")
            api_key: Mistral API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            enable_websearch: Whether to enable websearch capabilities (default: False)
            websearch_premium: Whether to use premium websearch with news agencies (default: False)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.logger = logging.getLogger("MistralClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_websearch = websearch_agent
        self.websearch_premium = websearch_premium
        self.client = Mistral(api_key=self.api_key)
        self.websearch_agent = None
        
        # Create websearch agent if enabled
        if self.enable_websearch:
            self._create_websearch_agent()

    def _create_websearch_agent(self):
        """
        Create a websearch agent with the specified configuration.
        """
        try:
            # Choose websearch tool type based on premium setting
            websearch_tool_type = "web_search_premium" if self.websearch_premium else "web_search"
            
            self.websearch_agent = self.client.beta.agents.create(
                model=self.model_name,
                description="Agent able to search information over the web, such as news, weather, sport results...",
                name="Websearch Agent",
                instructions="You have the ability to perform web searches to find up-to-date information. Use web search when you need current information that may not be in your training data.",
                tools=[{"type": websearch_tool_type}],
                completion_args={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "max_tokens": self.max_tokens,
                }
            )
            self.logger.info(f"Created websearch agent with ID: {self.websearch_agent.id}")
        except Exception as e:
            self.logger.error(f"Error creating websearch agent: {e}")
            raise

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Mistral API.
        If websearch is enabled, uses the websearch agent for conversations.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.chat.complete

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # If websearch is enabled and we have an agent, use conversation API
        if self.enable_websearch and self.websearch_agent:
            return self._chat_with_websearch_agent(messages, **kwargs)
        
        # Otherwise, use regular chat completion
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }

            response = self.client.chat.complete(**params)
        except Exception as e:
            self.logger.error(f"Error during Mistral API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
        # Extract done reasons (finish_reason in Mistral API)
        done_reasons = [choice.finish_reason for choice in response.choices]

        return [choice.message.content for choice in response.choices], usage, done_reasons

    def _chat_with_websearch_agent(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat using the websearch agent through the conversations API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        try:
            # Extract the user's input from the last message
            user_input = messages[-1]["content"] if messages else ""
            
            # Start a conversation with the websearch agent
            response = self.client.beta.conversations.start(
                agent_id=self.websearch_agent.id,
                inputs=user_input
            )
            
            # Extract the response content
            response_texts = []
            usage_info = response.usage
            
            for output in response.outputs:
                if output.type == "message.output":
                    # Combine text chunks and references
                    content_parts = []
                    for chunk in output.content:
                        if type(chunk) == str:
                            content_parts.append(chunk)
                        elif chunk.type == "text":
                            content_parts.append(chunk.text)
                        elif chunk.type == "tool_reference":
                            # Include reference information in the response
                            content_parts.append(f"\n\n**Source:** [{chunk.title}]({chunk.url})")
                    
                    response_texts.append("".join(content_parts))
            
            # Create usage object
            usage = Usage(
                prompt_tokens=usage_info.prompt_tokens,
                completion_tokens=usage_info.completion_tokens
            )
            
            # For websearch agents, we'll assume completion
            done_reasons = ["stop"] * len(response_texts) if response_texts else ["stop"]
            
            return response_texts, usage, done_reasons
            
        except Exception as e:
            self.logger.error(f"Error during websearch agent conversation: {e}")
            raise

    def chat_with_websearch(self, query: str, **kwargs) -> Tuple[str, Usage, List[Dict[str, Any]]]:
        """
        Convenience method to perform a websearch-enabled chat with a single query.

        Args:
            query: The search query or question
            **kwargs: Additional arguments

        Returns:
            Tuple of (response_text, Usage, references) where references contains source information
        """
        if not self.enable_websearch or not self.websearch_agent:
            raise ValueError("Websearch is not enabled. Set enable_websearch=True when initializing the client.")
        
        try:
            response = self.client.beta.conversations.start(
                agent_id=self.websearch_agent.id,
                inputs=query
            )
            
            response_text = ""
            references = []
            usage_info = response.usage
            
            for output in response.outputs:
                if output.type == "message.output":
                    for chunk in output.content:
                        if chunk.type == "text":
                            response_text += chunk.text
                        elif chunk.type == "tool_reference":
                            references.append({
                                "title": chunk.title,
                                "url": chunk.url,
                                "source": chunk.source
                            })
            
            usage = Usage(
                prompt_tokens=usage_info.prompt_tokens,
                completion_tokens=usage_info.completion_tokens
            )
            
            return response_text, usage, references
            
        except Exception as e:
            self.logger.error(f"Error during websearch query: {e}")
            raise

    def embed(self, inputs: List[str], **kwargs) -> Any:
        """
        Generate embeddings using the Mistral embeddings API.

        Args:
            inputs: List of strings to embed
            **kwargs: Additional arguments to pass to client.embeddings.create

        Returns:
            The embeddings response from the Mistral API
        """
        assert len(inputs) > 0, "Inputs cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "inputs": inputs,
                **kwargs,
            }

            response = self.client.embeddings.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Mistral embeddings API call: {e}")
            raise

        return response 
    
    