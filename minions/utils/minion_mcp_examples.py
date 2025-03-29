import os
import pathlib
import subprocess
import sys

from minions.clients.ollama import OllamaClient
import time
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


def example_local_codebase():
    minion = _make_mcp_minion("filesystem")
    context = ""
    task = "Identify all files in my minions repository that contain ollama imports."

    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_github():
    minion = _make_mcp_minion("github")
    context = "The MCP tools give access to GitHub."
    task = "Summarize the coding profile of the user, thomasbreydo, based on their repos."

    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_github_read_files():
    minion = _make_mcp_minion("github")
    context = "The MCP tools give access to the repo."
    task = "Identify how many .py files are in my thomasbreydo/minions repo, including .py files in subdirectories."

    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_local():
    minion = _make_mcp_minion("filesystem")
    context = ""
    task = "Summarize the contents of my mcp_test directory"

    output = minion(
        task=task,
        context=[context],
        max_rounds=5,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_mcp_unnecessary():
    minion = _make_mcp_minion("filesystem")

    context = """
    Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
    Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
    """

    task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

    output = minion(
        task=task,
        context=[context],
        max_rounds=5,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


if __name__ == '__main__':
    example_github()
