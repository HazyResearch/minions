import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import anthropic

from minions.usage import Usage


class AnthropicClient:
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        use_web_search: bool = False,
        include_search_queries: bool = False,
        use_caching: bool = False,
        use_code_interpreter: bool = False,
    ):
        """
        Initialize the Anthropic client.

        Args:
            model_name: The name of the model to use (default: "claude-3-sonnet-20240229")
            api_key: Anthropic API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            use_web_search: Whether to enable web search functionality (default: False)
            include_search_queries: Whether to include search queries in the response (default: False)
            use_caching: Whether to use caching for the client (default: False)
            use_code_interpreter: Whether to use the code interpreter (default: False)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.logger = logging.getLogger("AnthropicClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_web_search = use_web_search
        self.include_search_queries = include_search_queries
        self.use_code_interpreter = use_code_interpreter
        self.use_caching = use_caching
        
        # Initialize client with appropriate headers
        if self.use_code_interpreter:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers={
                    "anthropic-beta": "code-execution-2025-05-22"
                }
            )
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)

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
                **kwargs,
            }

            # Add web search tool if enabled
            if self.use_web_search:
                web_search_tool = {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": kwargs.get("max_web_search_uses", 5),
                }
                params["tools"] = params.get("tools", []) + [web_search_tool]

            if self.use_code_interpreter:
                code_interpreter_tool = {
                    "type": "code_execution_20250522",
                    "name": "code_execution",
                }
                if "tools" in params:
                    params["tools"].append(code_interpreter_tool)
                else:
                    params["tools"] = [code_interpreter_tool]

            response = self.client.messages.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Anthropic API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

        # Process response content
        if (
            self.use_web_search
            and hasattr(response, "content")
            and isinstance(response.content, list)
        ):
            # Handle structured response with potential web search results
            full_text_parts = []
            citations_parts = []
            search_queries = []

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

                # Capture search queries
                elif (
                    content_item.type == "server_tool_use"
                    and content_item.name == "web_search"
                ):
                    search_query = (
                        f"Search query: \"{content_item.input.get('query', '')}\""
                    )
                    search_queries.append(search_query)

                # We skip web_search_tool_result as the relevant information will be in citations

            # Combine all text parts
            full_text = " ".join(full_text_parts).strip()

            # Add search queries and citations if present
            result_text = full_text

            if self.include_search_queries and search_queries:
                result_text += "\n\n" + "\n".join(search_queries)

            if citations_parts:
                result_text += "\n\n" + "\n".join(citations_parts)

            return [result_text], usage
        
        else:
            # Standard response handling for non-web-search or simple responses
            if (
                hasattr(response, "content")
                and isinstance(response.content, list)
                and len(response.content) > 0
            ):
                if hasattr(response.content[0], "text"):
                    return [response.content[-1].text], usage
                else:
                    self.logger.warning(
                        "Unexpected response format - missing text attribute"
                    )
                    return [str(response.content)], usage
            else:
                self.logger.warning("Unexpected response format - missing content list")
                return [str(response)], usage
