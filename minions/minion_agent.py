"""
Minion Agent: Executes the Minion protocol using an Agno Agent as the local LM.

This module wraps an Agno Agent with tools to be compatible with the Minion protocol,
allowing the local model to leverage search, computation, and file handling tools.
"""

from typing import List, Dict, Any, Optional
import os

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.models.ollama import Ollama
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools
from agno.tools.website import WebsiteTools
from agno.tools.calculator import CalculatorTools
from agno.tools.python import PythonTools
from agno.tools.file import FileTools

from minions.usage import Usage
from minions.minion import Minion


class AgnoLocalClient:
    """
    Wrapper that adapts an Agno Agent to work as a local client for the Minion protocol.
    
    The Minion protocol expects a local_client with a chat() method that returns:
        (response: List[str], usage: Usage, done_reason: str)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        instructions: Optional[List[str]] = None,
        debug_mode: bool = False,
        debug_level: int = 0,
    ):
        """
        Initialize the Agno-based local client.
        
        Args:
            model: Agno model instance (HuggingFace, Ollama, OpenRouter, etc.)
            tools: List of Agno tool instances
            instructions: List of instruction strings for the agent
            debug_mode: Enable debug output
            debug_level: Debug verbosity level
        """
        # Default tools if none provided
        if tools is None:
            tools = [
                DuckDuckGoTools(),
                WikipediaTools(),
                ArxivTools(),
                WebsiteTools(),
                CalculatorTools(),
                PythonTools(),
                FileTools(),
            ]
        
        # Default instructions if none provided
        if instructions is None:
            instructions = [
                "Break down complex questions into steps",
                "Use the relevant tools to gather information, read files or perform calculations when appropriate",
                "Verify answers before responding",
                "Always cite sources when using external information",
            ]
        
        # Default model if none provided
        if model is None:
            model = OpenRouter(id="nvidia/nemotron-3-nano-30b-a3b")
        
        self.agent = Agent(
            model=model,
            tools=tools,
            markdown=True,
            instructions=instructions,
            debug_mode=debug_mode,
            debug_level=debug_level,
        )
        
        # Track token usage (Agno may not provide this, so we estimate)
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> tuple:
        """
        Process messages using the Agno Agent.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Tuple of (response_list, usage, done_reason)
        """
        # Extract the last user message or combine context from system + user messages
        prompt = self._messages_to_prompt(messages)
        
        try:
            # Run the agent and get response
            response = self.agent.run(prompt)
            
            # Extract the response content
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'messages') and response.messages:
                # Get the last assistant message
                response_text = response.messages[-1].content if response.messages else ""
            else:
                response_text = str(response)
            
            # Try to extract usage metrics if available
            usage = self._extract_usage(response)
            
            return ([response_text], usage, "stop")
            
        except Exception as e:
            print(f"Error in AgnoLocalClient.chat: {e}")
            # Return error message as response
            error_response = f"Error processing request: {str(e)}"
            return ([error_response], Usage(), "error")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a single prompt string for the Agno Agent.
        
        The Agno Agent expects a single prompt, so we need to combine the
        system context and user messages appropriately.
        """
        system_content = ""
        user_content = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_content = content
            elif role == "user":
                # Take the last user message as the main prompt
                user_content = content
            elif role == "assistant":
                # Include assistant responses in context for multi-turn
                if user_content:
                    user_content += f"\n\nPrevious response: {content}"
        
        # Combine system context with user query
        if system_content and user_content:
            # Include system context as background information
            prompt = f"""Background Context:
{system_content}

Current Request:
{user_content}"""
        else:
            prompt = user_content or system_content
        
        return prompt
    
    def _extract_usage(self, response: Any) -> Usage:
        """
        Extract token usage from Agno response if available.
        """
        usage = Usage()
        
        try:
            # Try to get usage metrics from response
            if hasattr(response, 'metrics'):
                metrics = response.metrics
                if hasattr(metrics, 'prompt_tokens'):
                    usage.prompt_tokens = metrics.prompt_tokens
                if hasattr(metrics, 'completion_tokens'):
                    usage.completion_tokens = metrics.completion_tokens
            elif hasattr(response, 'usage'):
                if hasattr(response.usage, 'prompt_tokens'):
                    usage.prompt_tokens = response.usage.prompt_tokens
                if hasattr(response.usage, 'completion_tokens'):
                    usage.completion_tokens = response.usage.completion_tokens
        except Exception:
            pass
        
        return usage


class MinionAgent:
    """
    High-level interface for running the Minion protocol with an Agno Agent as the local model.
    """
    
    def __init__(
        self,
        remote_client: Any,
        local_model: Optional[Any] = None,
        local_tools: Optional[List[Any]] = None,
        local_instructions: Optional[List[str]] = None,
        max_rounds: int = 3,
        callback: Optional[callable] = None,
        log_dir: str = "minion_agent_logs",
        debug_mode: bool = False,
    ):
        """
        Initialize the MinionAgent.
        
        Args:
            remote_client: Client for the remote/supervisor model (e.g., OpenAIClient)
            local_model: Agno model instance for the local worker
            local_tools: List of Agno tools for the local worker
            local_instructions: Instructions for the local worker agent
            max_rounds: Maximum conversation rounds in the protocol
            callback: Optional callback for message updates
            log_dir: Directory for logging
            debug_mode: Enable debug output for the local agent
        """
        # Create the Agno-wrapped local client
        self.local_client = AgnoLocalClient(
            model=local_model,
            tools=local_tools,
            instructions=local_instructions,
            debug_mode=debug_mode,
        )
        
        # Initialize the Minion protocol handler
        self.minion = Minion(
            local_client=self.local_client,
            remote_client=remote_client,
            max_rounds=max_rounds,
            callback=callback,
            log_dir=log_dir,
        )
    
    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds: Optional[int] = None,
        doc_metadata: Optional[Any] = None,
        logging_id: Optional[str] = None,
        is_privacy: bool = False,
        images: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the Minion protocol to answer a task.
        
        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds
            doc_metadata: Optional metadata about documents
            logging_id: Optional identifier for logging
            is_privacy: Enable privacy protection
            images: Optional images for multimodal tasks
        
        Returns:
            Dict with final_answer, conversation histories, and usage stats
        """
        return self.minion(
            task=task,
            context=context,
            max_rounds=max_rounds,
            doc_metadata=doc_metadata,
            logging_id=logging_id,
            is_privacy=is_privacy,
            images=images,
        )


# -- Sample Code --
# from minions.minion_agent import MinionAgent
# from minions.clients import OpenAIClient
# from agno.models.huggingface import HuggingFace

# # Remote supervisor (cloud)
# remote_client = OpenAIClient(model_name="gpt-4o-mini")

# # Local worker with tools (Agno Agent)
# local_model = HuggingFace(id="meta-llama/Llama-3.2-3B-Instruct")

# # Create MinionAgent
# minion_agent = MinionAgent(
#     remote_client=remote_client,
#     local_model=local_model,
#     max_rounds=3,
# )

# # Run a task
# result = minion_agent(
#     task="What is the population of Tokyo?",
#     context=["Provide accurate information."],
# )

# print(result['final_answer'])