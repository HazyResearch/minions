import asyncio
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import re

from minions.usage import Usage
from minions.clients.base import MinionsClient


class OllamaClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "llama-3.2",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        num_ctx: int = 48000,
        structured_output_schema: Optional[BaseModel] = None,
        use_async: bool = False,
        tool_calling: bool = False,
        thinking: bool = False,
        mcp_client=None,
        max_tool_iterations: int = 5,
        **kwargs
    ):
        """Initialize Ollama Client.
        
        Args:
            model_name: The Ollama model to use (default: "llama-3.2")
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            num_ctx: Context window size (default: 48000)
            structured_output_schema: Optional Pydantic model for structured output
            use_async: Whether to use async API calls (default: False)
            tool_calling: Whether to support tool calling (default: False)
            thinking: Whether to enable thinking mode (default: False)
            mcp_client: Optional MCP client for tool calling (SyncMCPClient)
            max_tool_iterations: Maximum number of tool calling iterations (default: 5)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Client-specific configuration
        self.num_ctx = num_ctx

        if self.model_name == "granite3.2-vision":
            self.num_ctx = 131072
            self.max_tokens = 131072

        self.use_async = use_async
        self.return_tools = tool_calling
        self.thinking = thinking
        
        # MCP tooling configuration
        self.mcp_client = mcp_client
        self.max_tool_iterations = max_tool_iterations
        self.mcp_tools_enabled = mcp_client is not None

        # If we want structured schema output:
        self.format_structured_output = None
        if structured_output_schema:
            self.format_structured_output = structured_output_schema.model_json_schema()

        # For async calls
        from ollama import AsyncClient

        self.client = AsyncClient() if use_async else None

        # Ensure model is pulled
        self._ensure_model_available()

        # Generate MCP tools in Ollama format
        self.ollama_tools = self._convert_mcp_tools_to_ollama_format() if self.mcp_tools_enabled else []

    @staticmethod
    def get_available_models():
        """
        Get a list of available Ollama models

        Returns:
            List[str]: List of model names
        """
        try:
            import ollama

            models = ollama.list()

            # Extract model names from the list
            model_names = [model.model for model in models["models"]]
            return model_names
        except Exception as e:
            logging.error(f"Failed to get Ollama model list: {e}")
            return []

    def _ensure_model_available(self):
        import ollama

        try:
            ollama.chat(
                model=self.model_name, messages=[{"role": "system", "content": "test"}]
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                self.logger.info(
                    f"Model {self.model_name} not found locally. Pulling..."
                )
                ollama.pull(self.model_name)
                self.logger.info(f"Successfully pulled model {self.model_name}")
            else:
                raise

    def _convert_mcp_tools_to_ollama_format(self) -> List[Dict]:
        """Convert MCP tools to Ollama tools format."""
        if not self.mcp_client:
            return []
            
        ollama_tools = []
        for tool in self.mcp_client.available_tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            ollama_tools.append(ollama_tool)
        
        return ollama_tools

    def _process_ollama_tool_calls(self, ollama_tool_calls: List[Dict]) -> str:
        """Process Ollama tool calls and execute MCP tools."""
        if not ollama_tool_calls:
            return ""
            
        
        results = []
        
        for i, tool_call in enumerate(ollama_tool_calls, 1):
            if "function" in tool_call:
                function_info = tool_call["function"]
                tool_name = function_info.get("name")
                arguments = function_info.get("arguments", {})
                
                if not tool_name:
                    results.append(f"Tool call {i}: Missing function name")
                    continue
                    
                try:
                    
                    result = self.mcp_client.execute_tool(tool_name, **arguments)
                    formatted_result = self.mcp_client.format_output(result)
                    results.append(f"Tool call {i} ({tool_name}):\n{formatted_result}")
                except Exception as e:
                    results.append(f"Tool call {i} ({tool_name}) failed: {str(e)}")
        
        return "\n\n".join(results)



    def _prepare_options(self):
        """Common chat options for both sync and async calls."""
        opts = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        chat_kwargs = {"options": opts}
        if self.format_structured_output:
            chat_kwargs["format"] = self.format_structured_output
        return chat_kwargs

    #
    #  ASYNC
    #
    def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], List[Usage], List[str]]:
        """
        Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a running event loop (e.g., in Streamlit)
                # Create a new loop in a separate thread to avoid conflicts
                import threading
                import concurrent.futures

                # Use a thread to run our async code
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_in_new_loop, messages, **kwargs)
                    return future.result()
            else:
                # We have a loop but it's not running
                return loop.run_until_complete(self._achat_internal(messages, **kwargs))
        except RuntimeError:
            # No event loop exists, create one (the normal case)
            try:
                return asyncio.run(self._achat_internal(messages, **kwargs))
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Create a new event loop and set it as the current one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._achat_internal(messages, **kwargs)
                        )
                    finally:
                        loop.close()
                raise

    def _run_in_new_loop(self, messages, **kwargs):
        """Run the async chat in a new event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._achat_internal(messages, **kwargs))
        finally:
            loop.close()

    async def _achat_internal(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle async chat with MCP tool calling support.
        """
        # If the user provided a single dictionary, wrap it in a list.
        if isinstance(messages, dict):
            messages = [messages]

        # Create a copy of messages to avoid modifying the original
        working_messages = messages.copy()
        
        chat_kwargs = self._prepare_options()
        
        # Add tools to chat kwargs if MCP is enabled
        if self.mcp_tools_enabled and self.ollama_tools:
            chat_kwargs["tools"] = self.ollama_tools

        if self.thinking:
            kwargs["think"] = True

        texts = []
        usage_total = Usage()
        done_reasons = []

        # Tool calling loop
        iteration = 0
        while iteration < self.max_tool_iterations:
            try:
                resp = await self.client.chat(
                    model=self.model_name,
                    messages=working_messages,
                    **chat_kwargs,
                    **kwargs,
                )
                
                response_content = resp["message"]["content"]
                
                # Track usage
                usage_total += Usage(
                    prompt_tokens=resp["prompt_eval_count"], 
                    completion_tokens=resp["eval_count"]
                )
                done_reasons.append(resp["done_reason"])

                # Check if MCP tools are enabled and handle tool calls from Ollama response
                if self.mcp_tools_enabled and "tool_calls" in resp["message"]:
                    ollama_tool_calls = resp["message"]["tool_calls"]
                    
                    if ollama_tool_calls:
                        # Execute tools
                        tool_results = self._process_ollama_tool_calls(ollama_tool_calls)
                        
                        # Add the model's response to conversation
                        working_messages.append({
                            "role": "assistant",
                            "content": response_content,
                            "tool_calls": ollama_tool_calls
                        })
                        
                        # Add tool results as tool message
                        for i, tool_call in enumerate(ollama_tool_calls):
                            tool_name = tool_call.get("function", {}).get("name", f"tool_{i}")
                            working_messages.append({
                                "role": "tool",
                                "content": tool_results,
                                "tool_call_id": f"call_{i}"
                            })
                        
                        iteration += 1
                        continue  # Continue the loop for another iteration
                
                # No tool calls or tools disabled - this is the final response
                texts.append(response_content)
                break

            except Exception as e:
                self.logger.error(f"Error during async Ollama API call: {e}")
                raise

        # If we completed max iterations without a final response, use fallback
        if not texts and iteration >= self.max_tool_iterations:
            texts.append("Maximum tool iterations reached. Unable to provide final response.")
            done_reasons = ["max_iterations"]

        return texts, usage_total, done_reasons

    def schat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions with optional MCP tool calling.
        """
        import ollama

        # If the user provided a single dictionary, wrap it
        if isinstance(messages, dict):
            messages = [messages]

        # Create a copy of messages to avoid modifying the original
        working_messages = messages.copy()
        
        chat_kwargs = self._prepare_options()
        
        # Add tools to chat kwargs if MCP is enabled
        if self.mcp_tools_enabled and self.ollama_tools:
            chat_kwargs["tools"] = self.ollama_tools
        responses = []
        usage_total = Usage()
        done_reasons = []
        tools = []

        # Tool calling loop
        iteration = 0
        while iteration < self.max_tool_iterations:

            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=working_messages,
                    **chat_kwargs,
                    **kwargs,
                )
                
                response_content = response["message"]["content"]
                
                # Track usage
                try:
                    usage_total += Usage(
                        prompt_tokens=response["prompt_eval_count"],
                        completion_tokens=response["eval_count"],
                    )
                except Exception:
                    usage_total += Usage(prompt_tokens=0, completion_tokens=0)
                
                try:
                    done_reasons.append(response["done_reason"])
                except Exception:
                    done_reasons.append("stop")

                # Check if MCP tools are enabled and handle tool calls from Ollama response  
                if self.mcp_tools_enabled and "tool_calls" in response["message"]:
                    ollama_tool_calls = response["message"]["tool_calls"]
                    
                    if ollama_tool_calls:
                        # Execute tools
                        tool_results = self._process_ollama_tool_calls(ollama_tool_calls)
                        
                        # Add the model's response to conversation
                        working_messages.append({
                            "role": "assistant", 
                            "content": response_content,
                            "tool_calls": ollama_tool_calls
                        })
                        
                        # Add tool results as tool message
                        for i, tool_call in enumerate(ollama_tool_calls):
                            tool_name = tool_call.get("function", {}).get("name", f"tool_{i}")
                            working_messages.append({
                                "role": "tool",
                                "content": tool_results,
                                "tool_call_id": f"call_{i}"
                            })
                        
                        iteration += 1
                        continue  # Continue the loop for another iteration
                
                # No tool calls or tools disabled - this is the final response
                responses.append(response_content)
                
                if "tool_calls" in response["message"]:
                    tools.append(response["message"]["tool_calls"])
                    
                break

            except Exception as e:
                self.logger.error(f"Error during Ollama API call: {e}")
                raise

        # If we completed max iterations without a final response, use the last response
        if not responses and iteration >= self.max_tool_iterations:
            responses.append("Maximum tool iterations reached. Unable to provide final response.")
            done_reasons = ["max_iterations"]

        if self.return_tools:
            return responses, usage_total, done_reasons, tools
        else:
            return responses, usage_total, done_reasons

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions. If you pass a list of message dicts,
        we do one call for that entire conversation. If you pass a single dict,
        we wrap it in a list so there's no error.
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)

    def embed(
        self,
        content,
        **kwargs,
    ):
        """Embed content using model (must support embeddings)."""
        import ollama

        response = ollama.embed(model=self.model_name, input=content, **kwargs)
        return response["embeddings"]
