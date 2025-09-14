import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import anthropic

from minions.usage import Usage
from minions.clients.base import MinionsClient


class AnthropicClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "claude-opus-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        use_web_search: bool = False,
        include_search_queries: bool = False,
        use_caching: bool = False,
        use_code_interpreter: bool = False,
        use_thinking: bool = False,
        thinking_budget_tokens: int = 10000,
        local: bool = False,
        **kwargs
    ):
        """
        Initialize the Anthropic client.

        Args:
            model_name: The name of the model to use (default: "claude-opus-4-20250514")
            api_key: Anthropic API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            use_web_search: Whether to enable web search functionality (default: False)
            include_search_queries: Whether to include search queries in the response (default: False)
            use_caching: Whether to use caching for the client (default: False)
            use_code_interpreter: Whether to use the code interpreter (default: False)
            use_thinking: Whether to enable thinking mode (default: False)
            thinking_budget_tokens: Token budget for thinking when enabled (default: 10000)
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
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.use_web_search = use_web_search
        self.include_search_queries = include_search_queries
        self.use_code_interpreter = use_code_interpreter
        self.use_caching = use_caching
        self.use_thinking = use_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        
        # Initialize client with appropriate headers
        beta_headers = []
        if self.use_code_interpreter:
            beta_headers.append("code-execution-2025-05-22")
            
        if beta_headers:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers={
                    "anthropic-beta": ",".join(beta_headers)
                }
            )
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)

        self.system_prompt = "You are a helpful assistant that can answer questions and help with tasks. Your outputs should be structured JSON objects. Follow the instructions in the user's message to generate the JSON object."
          

    def _detect_urls_in_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if there are any URLs in the messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if URLs are found, False otherwise
        """
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                if url_pattern.search(content):
                    return True
            elif isinstance(content, list):
                # Handle structured content (like with caching)
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_content = item.get('text', '')
                        if url_pattern.search(text_content):
                            return True
        return False

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Anthropic API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to client.messages.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Detect if URLs are present in messages
            has_urls = self._detect_urls_in_messages(messages)
            
            # Create a new client with web fetch support if URLs detected
            if has_urls:
                beta_headers = []
                if self.use_code_interpreter:
                    beta_headers.append("code-execution-2025-05-22")
                beta_headers.append("web-fetch-2025-09-10")
                
                client_for_request = anthropic.Anthropic(
                    api_key=self.api_key,
                    default_headers={
                        "anthropic-beta": ",".join(beta_headers)
                    }
                )
            else:
                client_for_request = self.client
            
            if self.use_caching:
                final_message = messages[-1]
                final_message["content"] =[ {
                    "type": "text",
                    "text": final_message["content"],
                    "cache_control": {"type": "ephemeral"},
                }]
                messages[-1] = final_message


            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": self.system_prompt,
                **kwargs,
            }

            # Add thinking parameter if enabled
            if self.use_thinking:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens
                }

            # Add web search tool if enabled
            if self.use_web_search:
                web_search_tool = {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": kwargs.get("max_web_search_uses", 5),
                }
                params["tools"] = params.get("tools", []) + [web_search_tool]

            # Add web fetch tool if URLs detected
            if has_urls:
                web_fetch_tool = {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_uses": kwargs.get("max_web_fetch_uses", 5),
                    "citations": {"enabled": True}
                }
                params["tools"] = params.get("tools", []) + [web_fetch_tool]

            if self.use_code_interpreter:
                code_interpreter_tool = {
                    "type": "code_execution_20250522",
                    "name": "code_execution",
                }
                if "tools" in params:
                    params["tools"].append(code_interpreter_tool)
                else:
                    params["tools"] = [code_interpreter_tool]

            response = client_for_request.messages.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Anthropic API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

        # Process response content
        if hasattr(response, "content") and isinstance(response.content, list):
            # Handle structured response with potential web search results and thinking blocks
            full_text_parts = []
            citations_parts = []
            search_queries = []
            thinking_parts = []

            for content_item in response.content:
                # Handle text content
                if content_item.type == "text":
                    text = content_item.text
                    full_text_parts.append(text)

                    # Process citations if present
                    if hasattr(content_item, "citations") and content_item.citations:
                        for citation in content_item.citations:
                            if citation.type == "web_search_result_location":
                                citation_text = (
                                    f'Source: {citation.url} - "{citation.cited_text}"'
                                )
                                if (
                                    citation_text not in citations_parts
                                ):  # Avoid duplicates
                                    citations_parts.append(citation_text)

                # Handle thinking content
                elif content_item.type == "thinking":
                    if hasattr(content_item, "thinking"):
                        thinking_summary = f"Thinking summary: {content_item.thinking}"
                        thinking_parts.append(thinking_summary)

                # Capture search queries (only if web search is enabled)
                elif (
                    self.use_web_search
                    and content_item.type == "server_tool_use"
                    and content_item.name == "web_search"
                ):
                    search_query = (
                        f"Search query: \"{content_item.input.get('query', '')}\""
                    )
                    search_queries.append(search_query)

                # Capture web fetch queries (when URLs are detected)
                elif (
                    has_urls
                    and content_item.type == "server_tool_use"
                    and content_item.name == "web_fetch"
                ):
                    fetch_url = content_item.input.get('url', '')
                    if fetch_url:
                        search_queries.append(f"Fetched content from: {fetch_url}")

                # Handle web fetch tool results
                elif content_item.type == "web_fetch_tool_result":
                    if hasattr(content_item, "content") and content_item.content:
                        fetch_content = content_item.content
                        if hasattr(fetch_content, "url"):
                            # Add URL info to citations if not already present
                            url_citation = f"Source: {fetch_content.url}"
                            if url_citation not in citations_parts:
                                citations_parts.append(url_citation)

                # We skip web_search_tool_result as the relevant information will be in citations

            # Combine all text parts
            full_text = " ".join(full_text_parts).strip()

            # Build result text with thinking, main content, search queries, and citations
            result_parts = []
            
            # Add thinking summaries if present
            if thinking_parts:
                result_parts.extend(thinking_parts)
            
            # Add main response text
            if full_text:
                result_parts.append(full_text)

            # Add search queries if enabled and present
            if (self.use_web_search or has_urls) and self.include_search_queries and search_queries:
                result_parts.extend(search_queries)

            # Add citations if present
            if citations_parts:
                result_parts.extend(citations_parts)

            result_text = "\n\n".join(result_parts) if result_parts else ""

            if self.local:
                return [result_text], usage, ["stop"]
            else:
                return [result_text], usage
                
        else:
            # Standard response handling for non-web-search or simple responses
            if (
                hasattr(response, "content")
                and isinstance(response.content, list)
                and len(response.content) > 0
            ):
                if hasattr(response.content[0], "text"):
                    print(response.content[-1].text)
                    if self.local:
                        return [response.content[-1].text], usage, ["stop"]
                    else:
                        return [response.content[-1].text], usage
                else:
                    self.logger.warning(
                        "Unexpected response format - missing text attribute"
                    )
                    if self.local:
                        return [str(response.content)], usage, ["stop"]
                    else:
                        return [str(response.content)], usage
            else:
                self.logger.warning("Unexpected response format - missing content list")
                if self.local:
                    return [str(response)], usage, ["stop"]
                else:
                    return [str(response)], usage
