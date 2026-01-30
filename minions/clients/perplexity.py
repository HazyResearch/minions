import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient

try:
    from perplexity import Perplexity
except ImportError:
    Perplexity = None




class PerplexityAIClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "sonar-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        search_receny_filter: Optional[str] = None,
        search_after_date_filter: Optional[str] = None,
        search_before_date_filter: Optional[str] = None,
        search_last_updated_filter: Optional[str] = None,
        search_max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Perplexity client.

        Args:
            model_name: The name of the model to use (default: "sonar-pro")
            api_key: Perplexity API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            search_receny_filter: Filter search results by recency (optional)
            search_after_date_filter: Filter search results after a specific date (optional)
            search_before_date_filter: Filter search results before a specific date (optional)
            search_last_updated_filter: Filter search results by when content was last updated (optional)
            search_max_tokens: Maximum tokens extracted per page in search results (optional)
            base_url: Base URL for the Perplexity API (optional, falls back to PERPLEXITY_BASE_URL environment variable or default URL)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        # Client-specific configuration
        openai.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.api_key = openai.api_key
        
        # Get base URL from parameter, environment variable, or use default
        base_url = base_url or os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
        
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=base_url
        )
        self.search_receny_filter = search_receny_filter
        self.search_after_date_filter = search_after_date_filter
        self.search_before_date_filter = search_before_date_filter
        self.search_last_updated_filter = search_last_updated_filter
        self.search_max_tokens = search_max_tokens

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI  client, but route to perplexity

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # add a system prompt to the top of the messages
        messages.insert(
            0,
            {
                "role": "system",
                "content": "You are language model that has access to the internet if you need it.",
            },
        )

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                **kwargs,
            }

            params["temperature"] = self.temperature
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.logger.error(f"Error during Sonar API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        
        # Extract cost information if available (new in July 2025)
        try:
            if hasattr(response.usage, 'cost') and response.usage.cost:
                cost = response.usage.cost
                usage.input_tokens_cost = getattr(cost, 'input_tokens_cost', None)
                usage.output_tokens_cost = getattr(cost, 'output_tokens_cost', None)
                usage.request_cost = getattr(cost, 'request_cost', None)
                usage.total_cost = getattr(cost, 'total_cost', None)
                
                # Log cost information if available
                if usage.total_cost is not None:
                    self.logger.info(f"Perplexity API cost: ${usage.total_cost:.6f} (input: ${usage.input_tokens_cost:.6f}, output: ${usage.output_tokens_cost:.6f}, request: ${usage.request_cost:.6f})")
        except AttributeError:
            # Cost information not available in this response
            pass
        
        # Extract search context size if available (Perplexity-specific)
        try:
            if hasattr(response.usage, 'search_context_size'):
                usage.search_context_size = response.usage.search_context_size
                self.logger.info(f"Search context size: {usage.search_context_size}")
        except AttributeError:
            # Search context size not available
            pass

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage

    def search(self, query: Union[str, List[str]], **kwargs):
        """
        Perform a search using Perplexity's search capabilities.
        
        Args:
            query: A single query string or list of query strings to search for
            **kwargs: Additional arguments to pass to the search API
            
        Returns:
            SearchResponse: Object containing search results with title and URL for each result
        """
        # Ensure query is a list
        search_client = Perplexity(api_key=self.api_key)

        if isinstance(query, str):
            queries = [query]
        else:
            queries = query
            
        try:
            search_params = {
                "query": queries,
                "search_after_date_filter": self.search_after_date_filter,
                "search_before_date_filter": self.search_before_date_filter,
                "search_recent_filter": self.search_receny_filter,
            }
            
            # Add last_updated_filter if specified (December 2025 enhancement)
            if self.search_last_updated_filter is not None:
                search_params["last_updated_filter"] = self.search_last_updated_filter
            
            # Add max_tokens if specified (December 2025 enhancement)
            if self.search_max_tokens is not None:
                search_params["max_tokens"] = self.search_max_tokens
            
            search = search_client.search(**search_params, **kwargs)
            results = search.results
            return results
            
            
        except Exception as e:
            self.logger.error(f"Error during Perplexity search: {e}")
            # Return empty results on error
            return []

    def responses(
        self,
        input_text: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        preset: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Usage]:
        """
        Use Perplexity's Agentic Research API for multi-provider access with web search.
        
        See: https://docs.perplexity.ai/docs/grounded-llm/responses/quickstart
        
        Args:
            input_text: Query string
            model: Model to use (e.g., "openai/gpt-5.2"). Required if preset not specified.
            tools: List of tools to enable (e.g., [{"type": "web_search"}])
            instructions: System instructions for the model
            preset: Use a preset (e.g., "pro-search"). If specified, model is optional.
            **kwargs: Additional parameters passed to the API
            
        Returns:
            Tuple of (output_text, Usage)
                
        Example:
            # Using a third-party model with web search
            response, usage = client.responses(
                input_text="What are the latest developments in AI?",
                model="openai/gpt-5.2",
                tools=[{"type": "web_search"}],
                instructions="Use web_search for current events.",
            )
            
            # Using a preset
            response, usage = client.responses(
                input_text="Explain quantum computing",
                preset="pro-search",
            )
        """
        if Perplexity is None:
            raise ImportError(
                "Perplexity SDK is required for the Agentic Research API. "
                "Install with: pip install perplexityai"
            )
        
        pplx_client = Perplexity(api_key=self.api_key)
        
        try:
            params = {"input": input_text}
            
            if preset:
                params["preset"] = preset
            elif model:
                params["model"] = model
            else:
                params["model"] = "openai/gpt-5.2"
            
            if tools:
                params["tools"] = tools
            
            if instructions:
                params["instructions"] = instructions
            
            params.update(kwargs)
            
            response = pplx_client.responses.create(**params)
            
            # Extract output text using convenience property
            output_text = response.output_text if hasattr(response, 'output_text') else ""
            
            # Extract usage
            usage = Usage()
            if hasattr(response, 'usage') and response.usage:
                usage = Usage(
                    prompt_tokens=getattr(response.usage, 'input_tokens', 0),
                    completion_tokens=getattr(response.usage, 'output_tokens', 0),
                )
            
            return output_text, usage
            
        except Exception as e:
            self.logger.error(f"Error during Perplexity Agentic Research API call: {e}")
            raise

    @staticmethod
    def get_available_models():
        """
        Get a list of available models from Perplexity AI.
        
        Returns:
            List[str]: List of model names available through Perplexity API
        """
        return [
            # Search models - lightweight, cost-effective information retrieval
            "sonar",
            "sonar-pro", 
            
            # Reasoning models - complex, multi-step tasks with step-by-step thinking
            # Note: sonar-reasoning was deprecated and removed as of December 15, 2025
            "sonar-reasoning-pro",
            
            # Research models - in-depth analysis and comprehensive reports
            "sonar-deep-research",
        ]
