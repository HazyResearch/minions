import os
import pathlib
import subprocess
import sys

from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion
from minions.minions_mcp import SyncMCPClient, MCPConfigManager

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_DIR / "minion_logs"
MCP_CONFIG_PATH = PROJECT_DIR / "mcp.json"


def _start_mcp_server(server_name: str, server_config_manager: MCPConfigManager):
    """Starts the `server_name` MCP server via the command specified in the config manager"""
    command = server_config_manager.servers[server_name].command
    args = server_config_manager.servers[server_name].args
    env = os.environ.copy() | (server_config_manager.servers[server_name].env or {})
    subprocess.Popen([command, *args], stderr=sys.stderr, stdout=sys.stdout, env=env)


def _start_clients() -> tuple[OllamaClient, OpenAIClient]:
    print("Connecting to Ollama and OpenAI... ")
    local_client = OllamaClient(model_name="phi4-mini")
    remote_client = OpenAIClient(model_name="gpt-4o")
    print("done.")
    return local_client, remote_client


def _make_mcp_minion(mcp_server_name: str) -> Minion:
    """Start MCP server (e.g. 'github' or 'filesystem') and make a Minion with access to it"""
    local_client, remote_client = _start_clients()

    config_manager = MCPConfigManager(MCP_CONFIG_PATH)
    _start_mcp_server(mcp_server_name, config_manager)
    mcp_client = SyncMCPClient(mcp_server_name, config_manager)

    # Instantiate the Minion object with both clients
    return Minion(local_client, remote_client, mcp_client=mcp_client, log_dir=LOG_DIR)
