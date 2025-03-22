import os
import pathlib
import subprocess
import sys

from minions.clients.ollama import OllamaClient
import time
from minions.clients.openai import OpenAIClient
from minions.minion import Minion
from minions.minions_mcp import SyncMCPClient, MCPConfigManager

MCP_CONFIG_PATH = "/Users/thomasbreydo/dev/minions/mcp.json"

LOG_DIR = pathlib.Path(__file__).parent.parent / "minions_logs"


def _start_github_mcp_server(server_config_manager: MCPConfigManager):
    command = server_config_manager.servers["github"].command
    args = server_config_manager.servers["github"].args
    env = os.environ.copy() | server_config_manager.servers["github"].env
    subprocess.Popen([command, *args], stderr=sys.stderr, stdout=sys.stdout, env=env)


def _start_filesystem_mcp_server(server_config_manager: MCPConfigManager):
    command = server_config_manager.servers["filesystem"].command
    args = server_config_manager.servers["filesystem"].args
    subprocess.Popen([command, *args], stderr=sys.stderr, stdout=sys.stdout)


def example_mcp_unnecessary():
    # context = """
    # Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
    # Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
    # """

    # task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."
    pass


def _start_clients() -> tuple[OllamaClient, OpenAIClient]:
    print("Connecting to Ollama client... ")
    local_client = OllamaClient(
        model_name="phi4-mini",
    )
    print("done.")

    print("Connecting to OpenAI client... ")
    remote_client = OpenAIClient(
        model_name="gpt-4o",
    )
    print("done.")

    return local_client, remote_client


def _make_filesystem_minion() -> Minion:
    """Start filesystem MCP server and make a Minion with access to it"""
    local_client, remote_client = _start_clients()

    config_manager = MCPConfigManager(MCP_CONFIG_PATH)
    _start_filesystem_mcp_server(config_manager)
    mcp_client = SyncMCPClient("filesystem", config_manager)

    # Instantiate the Minion object with both clients
    return Minion(local_client, remote_client, mcp_client=mcp_client, log_dir=LOG_DIR)


def _make_github_minion() -> Minion:
    """Start GitHub MCP server and make a Minion with access to it"""
    local_client, remote_client = _start_clients()

    config_manager = MCPConfigManager(MCP_CONFIG_PATH)
    _start_github_mcp_server(config_manager)
    mcp_client = SyncMCPClient("github", config_manager)

    # Instantiate the Minion object with both clients
    return Minion(local_client, remote_client, mcp_client=mcp_client, log_dir=LOG_DIR)


def example_local_codebase():
    minion = _make_filesystem_minion()
    context = ""
    task = "Identify all files in my minions repository that contain ollama imports."

    # Execute the minion protocol for up to two communication rounds
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(output)


def example_github():
    minion = _make_github_minion()
    context = "The MCP tools give access to GitHub."
    task = "Summarize the coding profile of the user, thomasbreydo, based on their repos."

    # Execute the minion protocol for up to two communication rounds
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(output)


def example_local_doordash():
    minion = _make_filesystem_minion()
    context = ""
    task = "Summarize the contents of my latest doordash receipts."

    # Execute the minion protocol for up to two communication rounds
    output = minion(
        task=task,
        context=[context],
        max_rounds=5,
        logging_id=int(time.time()),
    )

    print(output)


if __name__ == '__main__':
    example_github()
