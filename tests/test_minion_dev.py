from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion_code import DevMinion   # DevMinion lives here

# 1️⃣  pick your models
local_client  = OllamaClient(model_name="llama3.2")
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
    task="Build a tiny Flask API that echoes JSON",
    requirements="""
        * Use Flask 3.x
        * Expose POST /echo
        * Include Dockerfile & unit tests
    """
)

print("Runbook generated at:", result["runbook"]["project_overview"])
print("Final assessment:\n", result["final_assessment"])