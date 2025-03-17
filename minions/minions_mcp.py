import json
import os
import subprocess
import tempfile
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
import requests
import uuid
import time

from dataclasses import dataclass
from typing import Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment
import asyncio
from contextlib import AsyncExitStack

from minions.minions import Minions, USEFUL_IMPORTS, JobManifest, JobOutput, Job

from minions.prompts.minions_mcp import (
    DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC,
    DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""

    command: str
    args: list[str]
    env: Optional[Dict[str, str]] = None


class MCPConfigManager:
    """Manages MCP server configurations"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP config manager

        Args:
            config_path: Path to MCP config file. If None, will look in default locations
        """
        self.config_path = config_path
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load MCP configuration from file"""
        paths_to_try = [
            self.config_path,
            os.path.join(os.getcwd(), "mcp.json"),
            os.path.join(os.getcwd(), ".mcp.json"),
            os.path.expanduser("~/.mcp.json"),
        ]

        config_file = None
        for path in paths_to_try:
            if path and os.path.exists(path):
                config_file = path
                break

        if not config_file:
            return

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            if "mcpServers" in config:
                for server_name, server_config in config["mcpServers"].items():
                    self.servers[server_name] = MCPServerConfig(
                        command=server_config["command"],
                        args=server_config["args"],
                        env=server_config.get("env"),
                    )
        except Exception as e:
            raise ValueError(f"Failed to load MCP config from {config_file}: {str(e)}")

    def get_server_config(self, server_name: str) -> MCPServerConfig:
        """Get configuration for a specific MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"MCP server '{server_name}' not found in config")
        return self.servers[server_name]

    def list_servers(self) -> list[str]:
        """Get list of configured server names"""
        return list(self.servers.keys())


class SyncMCPClient:
    """A synchronous wrapper around the async MCP client API"""

    def __init__(self, server_name: str, config_manager: MCPConfigManager):
        """Initialize the synchronous MCP client"""
        self.server_name = server_name
        self.config_manager = config_manager
        self.server_config = self.config_manager.get_server_config(server_name)
        self._available_tools = []
        self._asyncio_thread = None
        self._client_initialized = threading.Event()
        self._result_ready = threading.Event()
        self._request_queue = []
        self._result = None
        self._error = None
        self._initialize()

    def _run_async_loop(self):
        """Run the asyncio event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def init_client():
            try:
                # Set up environment
                env = self.server_config.env
                if env:
                    default_envs = get_default_environment()
                    env = {**default_envs, **env}

                # Create server parameters
                server_params = StdioServerParameters(
                    command=self.server_config.command,
                    args=self.server_config.args,
                    env=env,
                )

                # Create stdio client and session
                async with stdio_client(server_params) as (stdio, write):
                    async with ClientSession(stdio, write) as session:
                        await session.initialize()

                        # Get available tools
                        response = await session.list_tools()
                        self._available_tools = [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            }
                            for tool in response.tools
                        ]

                        # Signal that the client is initialized
                        self._client_initialized.set()

                        # Process requests
                        while True:
                            if not self._request_queue:
                                await asyncio.sleep(0.1)  # Don't busy-wait
                                continue

                            # Get the next request
                            tool_name, kwargs = self._request_queue.pop(0)

                            try:
                                # Execute the tool
                                result = await session.call_tool(tool_name, kwargs)
                                self._result = result
                                self._error = None
                            except Exception as e:
                                self._result = None
                                self._error = e

                            # Signal that the result is ready
                            self._result_ready.set()

            except Exception as e:
                print(f"Error initializing MCP client: {e}")
                self._error = e
                self._client_initialized.set()  # Signal even on error

        # Start the asyncio task
        loop.run_until_complete(init_client())

    def _initialize(self):
        """Initialize the client in a separate thread"""
        self._asyncio_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._asyncio_thread.start()

        # Wait for the client to be initialized
        if not self._client_initialized.wait(timeout=30):
            raise TimeoutError("Timed out waiting for MCP client to initialize")

        # Check if there was an error during initialization
        if self._error:
            raise self._error

    @property
    def available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools"""
        return self._available_tools

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with the given parameters synchronously"""
        # Reset flags
        self._result_ready.clear()

        # Expand home directory if needed
        if (
            "path" in kwargs
            and isinstance(kwargs["path"], str)
            and "~" in kwargs["path"]
        ):
            kwargs["path"] = os.path.expanduser(kwargs["path"])

        print(f"Executing tool {tool_name} with args: {kwargs}")

        # Add request to queue
        self._request_queue.append((tool_name, kwargs))

        # Wait for the result
        if not self._result_ready.wait(timeout=240):
            raise TimeoutError(f"Timed out waiting for tool {tool_name} to execute")

        # Check if there was an error
        if self._error:
            raise self._error

        return self._result

    def format_output(self, output: Any) -> str:
        """Format the output of a tool"""
        return output.content[0].text


class SyncMCPToolExecutor:
    """A class to execute MCP tools synchronously"""

    def __init__(self, mcp_client: SyncMCPClient):
        """Initialize with a SyncMCPClient"""
        self.mcp_client = mcp_client

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute an MCP tool synchronously"""
        output = self.mcp_client.execute_tool(tool_name, **kwargs)
        return self.mcp_client.format_output(output)

    # def format_output(self, output: Any) -> str:
    #     """Format the output of a tool"""
    #     return self.mcp_client.format_output(output)


class SyncMinionsMCP(Minions):
    """Minions with synchronous MCP tool integration"""

    def __init__(
        self,
        local_client=None,
        remote_client=None,
        mcp_config_path=None,
        mcp_server_name="filesystem",
        max_rounds=5,
        callback=None,
        **kwargs,
    ):
        """Initialize SyncMinionsMCP with local, remote LLM clients and MCP."""
        # Modify the decompose task prompts to include MCP tools info

        decompose_task_prompt = (
            kwargs.get("decompose_task_prompt", None)
            or DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC
        )
        decompose_task_prompt_abbreviated = (
            kwargs.get("decompose_task_prompt_abbreviated", None)
            or DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC
        )

        kwargs["decompose_task_prompt"] = decompose_task_prompt
        kwargs["decompose_task_prompt_abbreviated"] = decompose_task_prompt_abbreviated

        # Set up the parent class
        super().__init__(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=max_rounds,
            callback=callback,
            **kwargs,
        )

        # Initialize MCP config and client
        self.mcp_config_manager = MCPConfigManager(config_path=mcp_config_path)
        self.mcp_client = SyncMCPClient(
            server_name=mcp_server_name, config_manager=self.mcp_config_manager
        )
        self.mcp_tool_executor = SyncMCPToolExecutor(self.mcp_client)

    def _execute_code(
        self,
        code: str,
        starting_globals: Dict[str, Any] = {},
        fn_name: str = "prepare_jobs",
        **kwargs,
    ) -> Tuple[Any, str]:
        """Execute code with MCP tools available"""
        # Add MCP tools to the execution globals
        exec_globals = {
            **starting_globals,
            "mcp_tools": self.mcp_tool_executor,
        }

        print("About to execute code...")
        # Compile and execute the code
        try:
            compile(code, "<string>", "exec")
            print("Code compiled successfully")
            exec(code, exec_globals)
            print("Code executed successfully")

            if fn_name not in exec_globals:
                raise ValueError(f"Function {fn_name} not found in the code block.")

            print(f"About to call {fn_name}...")
            function = exec_globals[fn_name]
            output = function(**kwargs)
            print("Function call completed")

            return output, code

        except Exception as e:
            print(f"Error executing code: {e}")
            raise

    def __call__(
        self,
        task: str,
        doc_metadata: str,
        context: List[str],
        max_rounds=None,
        num_tasks_per_round=3,
        num_samples_per_task=1,
        use_bm25=False,
    ):
        """Run the minions protocol with MCP tools available"""
        # Generate MCP tools info
        mcp_tools_info = "# Available MCP Tools\n\n"
        for tool in self.mcp_client.available_tools:
            mcp_tools_info += f"## {tool['name']}\n\n"
            mcp_tools_info += f"**Description**: {tool['description']}\n\n"

            # Create parameter list from schema
            params = []
            if "properties" in tool["input_schema"]:
                for param_name in tool["input_schema"]["properties"].keys():
                    params.append(param_name)
            # Add return parameter information if available

            mcp_tools_info += f"**Usage**: mcp_tools.execute_tool(\"{tool['name']}\", {', '.join([f'{p}={p}' for p in params])})\n\n"

        # Run the parent class call with MCP tools info
        result = super().__call__(
            task=task,
            doc_metadata=doc_metadata,
            context=context,
            max_rounds=max_rounds,
            num_tasks_per_round=num_tasks_per_round,
            num_samples_per_task=num_samples_per_task,
            mcp_tools_info=mcp_tools_info,
            use_bm25=use_bm25,
        )

        return result


# Example usage
if __name__ == "__main__":
    # Example usage of SyncMinionsMCP
    from minions.clients.ollama import OllamaClient
    from minions.clients.openai import OpenAIClient
    from pydantic import BaseModel

    # Initialize clients
    class StructuredLocalOutput(BaseModel):
        explanation: str
        citation: str | None
        answer: str | None

    # Option 1: Ollama
    local_client = OllamaClient(
        model_name="llama3.1:8b",
        temperature=0.0,
        structured_output_schema=StructuredLocalOutput,
    )
    remote_client = OpenAIClient(model_name="gpt-4o", temperature=0.0)

    # Get MCP config path from environment or use default
    mcp_config_path = os.environ.get("MCP_CONFIG_PATH", "~/.mcp.json")

    try:
        # Create SyncMinionsMCP instance
        minions = SyncMinionsMCP(
            local_client=local_client,
            remote_client=remote_client,
            mcp_config_path=mcp_config_path,
            mcp_server_name="filesystem",
        )

        # Run the minions protocol with MCP tools available
        result = minions(
            task="Get me the paths to doordash food order reciepts in /Users/avanikanarayan/Downloads/mcp_test/",
            doc_metadata="File system analysis task",
            context=[],
            max_rounds=2,
        )

        print(result["final_answer"])
    except Exception as e:
        print(f"Error running SyncMinionsMCP: {e}")
