# Example usage of SyncMinionsMCP using a simple Calculator class
from minions.clients.lemonade import LemonadeClient
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from pydantic import BaseModel
from minions.minions_mcp import SyncMinionsMCP

# Initialize clients
class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None


test_lemonade = False

if test_lemonade:
    # Option 1: Lemonade
    local_client = LemonadeClient(
        model_name="Qwen3-8B-GGUF",
        temperature=0.0,
        structured_output_schema=StructuredLocalOutput,
        use_async=True
    )
else:
    # Option 2: Ollama
    local_client = OllamaClient(
        model_name="llama3.2:1b",
        temperature=0.0,
        structured_output_schema=StructuredLocalOutput,
    )

remote_client = OpenAIClient(model_name="gpt-4o", temperature=0.0)

# Get MCP config path from environment or use default
mcp_config_path = "C:\\Users\\Edan\\minions\\mcp.json"

try:
    # Create SyncMinionsMCP instance
    minions = SyncMinionsMCP(
        local_client=local_client,
        remote_client=remote_client,
        mcp_config_path=mcp_config_path,
        mcp_server_name="calculator"
    )

    # Run the minions protocol with MCP tools available
    result = minions(
        task="Use the calculator addition to add 2 + 2 and then also give me the answer to 6 * 7",
        doc_metadata="Calculator",
        context=[],
        max_rounds=2,
    )

    print(result["final_answer"])
except Exception as e:
    print(f"Error running SyncMinionsMCP: {e}")