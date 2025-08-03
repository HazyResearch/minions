from minions.clients.lemonade import LemonadeClient
from minions.clients.openai import OpenAIClient
from minions.minion_code import DevMinion   # DevMinion lives here

from typing import Dict, List
from pydantic import BaseModel

class StructuredLocalOutput(BaseModel):
    files: Dict[str, str]  # filename
    documentation: str
    setup_instructions: List[str]
    completion_notes: str

# 1️⃣  pick your models
local_client  = LemonadeClient(model_name="Qwen3-8B-GGUF", structured_output_schema=StructuredLocalOutput)
remote_client = OpenAIClient(model_name="gpt-4o")

# 2️⃣  create a workspace (it’s auto‑made if missing)
dev = DevMinion(
        local_client,
        remote_client,
        workspace_dir="dev_workspace",   # or any path you like
        max_edit_rounds=3                # iterations per sub‑task
)

# 3️⃣  launch a job
result = dev(
    task="Build a function that finds the next prime number after a passed in value.",
    requirements="""
        * Use Python
    """
)

print("Runbook generated at:", result["runbook"]["project_overview"])
print("Final assessment:\n", result["final_assessment"])