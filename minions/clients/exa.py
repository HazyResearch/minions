import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class ExaClient(MinionsClient):
    """
    Exa AI client for search-powered LLM responses.
    
    Exa provides OpenAI-compatible endpoints with built-in web search capabilities.
    See: https://exa.ai/docs/reference/openai-sdk
    
    Models:
        - "exa": Default model for /answer endpoint - quick search-augmented responses
        - "exa-research": Research model for deeper analysis
        - "exa-research-pro": Pro research model for comprehensive research tasks
    """
    
    def __init__(
        self,
        model_name: str = "exa",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        include_text: bool = True,
        **kwargs
    ):
        """
        Initialize the Exa client.

        Args:
            model_name: The name of the model to use. Options:
                       - "exa" (default): Quick search-augmented answers
                       - "exa-research": Deeper research analysis
                       - "exa-research-pro": Comprehensive research
            api_key: Exa API key (optional, falls back to EXA_API_KEY env var)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Base URL for the Exa API (default: https://api.exa.ai)
            include_text: Whether to include full text from sources (default: True)
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
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.include_text = include_text
        self.last_citations = None  # Store citations from last response
        
        # Get base URL from parameter, environment variable, or use default
        self.base_url = base_url or os.getenv("EXA_BASE_URL", "https://api.exa.ai")
        
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=self.base_url
        )

    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        output_schema: Optional[Dict[str, Any]] = None,
        include_text: Optional[bool] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using Exa's OpenAI-compatible API.
        
        This routes to Exa's /chat/completions endpoint which provides
        search-augmented responses with citations.

        Example:
            client = ExaClient(model_name="exa")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are the latest developments in quantum computing?"}
            ]
            responses, usage = client.chat(messages)
            print(responses[0])
            print(client.get_citations())  # Get citations from last response
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # Determine whether to include text
        text_setting = include_text if include_text is not None else self.include_text

        try:
            # Build extra_body with Exa-specific parameters
            extra_body = kwargs.pop("extra_body", {})
            extra_body["text"] = text_setting
            
            # Add output_schema if provided
            if output_schema:
                extra_body["output_schema"] = output_schema

            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                "extra_body": extra_body,
                "stream": stream,
                **kwargs,
            }

            response = self.client.chat.completions.create(**params)
            
            # Handle streaming response
            if stream:
                return self._handle_stream_response(response)
                
        except Exception as e:
            self.logger.error(f"Error during Exa API call: {e}")
            raise

        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        
        # Extract cost information if available
        self._extract_cost_info(response, usage)
        
        # Extract citations from the response
        self._extract_citations(response)

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage

    def responses(
        self,
        input_text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Usage]:
        """
        Use Exa's Responses API for single-turn research tasks.
            
        Example:
            client = ExaClient(model_name="exa-research")
            response, usage = client.responses(
                "Summarize the impact of CRISPR on gene therapy with recent developments"
            )
            print(response)
        """
        model_to_use = model or self.model_name
        
        # Validate model for responses API
        if model_to_use == "exa":
            self.logger.warning(
                "The 'exa' model may not fully support the Responses API. "
                "Consider using 'exa-research' or 'exa-research-pro' for better results."
            )

        try:
            # Use the responses endpoint
            response = self.client.responses.create(
                model=model_to_use,
                input=input_text,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error during Exa Responses API call: {e}")
            raise

        # Extract usage information if available
        usage = Usage()
        if hasattr(response, 'usage') and response.usage:
            usage = Usage(
                prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
                completion_tokens=getattr(response.usage, 'completion_tokens', 0),
            )
            self._extract_cost_info(response, usage)

        # Extract the output text
        output_text = ""
        if hasattr(response, 'output'):
            output_text = response.output
        elif hasattr(response, 'choices') and response.choices:
            output_text = response.choices[0].message.content

        return output_text, usage

    def _handle_stream_response(self, stream_response) -> Tuple[List[str], Usage]:
        """Handle streaming response and collect full content."""
        full_content = ""
        usage = Usage()
        
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                
            # Try to extract usage from final chunk
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens if chunk.usage else 0,
                    completion_tokens=chunk.usage.completion_tokens if chunk.usage else 0,
                )
        
        return [full_content], usage

    def _extract_cost_info(self, response, usage: Usage):
        """Extract cost information from response if available."""
        try:
            if hasattr(response, 'usage') and hasattr(response.usage, 'cost') and response.usage.cost:
                cost = response.usage.cost
                usage.input_tokens_cost = getattr(cost, 'input_tokens_cost', None)
                usage.output_tokens_cost = getattr(cost, 'output_tokens_cost', None)
                usage.request_cost = getattr(cost, 'request_cost', None)
                usage.total_cost = getattr(cost, 'total_cost', None)
                
                # Log cost information if available
                if usage.total_cost is not None:
                    self.logger.info(
                        f"Exa API cost: ${usage.total_cost:.6f} "
                        f"(input: ${usage.input_tokens_cost:.6f}, "
                        f"output: ${usage.output_tokens_cost:.6f}, "
                        f"request: ${usage.request_cost:.6f})"
                    )
        except AttributeError:
            # Cost information not available in this response
            pass

    def _extract_citations(self, response):
        """Extract citations from response if available."""
        self.last_citations = None
        try:
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'citations'):
                    self.last_citations = message.citations
        except AttributeError:
            pass

    def get_citations(self) -> Optional[List[Any]]:
        """
        Get citations from the last chat response.
        
        Returns:
            List of citation objects from the last response, or None if no citations
        """
        return self.last_citations

    @staticmethod
    def get_available_models():
        """
        Get a list of available models from Exa.
        
        Returns:
            List[str]: List of model names available through Exa API
        """
        return [
            "exa",              # Default model for quick search-augmented answers
            "exa-research",     # Research model for deeper analysis
            "exa-research-pro", # Pro research model for comprehensive research
        ]

