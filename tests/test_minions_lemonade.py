from minions.clients.lemonade import LemonadeClient
from minions.clients.openai import OpenAIClient
from minions.minions import Minions
from pydantic import BaseModel

# Define the structured output schema for local client
class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

# Instantiate the local Lemonade client (make sure Lemonade server is running with GGUF model)
local_client = LemonadeClient(
    model_name="Qwen3-8B-GGUF",  # Must be a GGUF model
    temperature=0.0,
    max_tokens=2048,
    structured_output_schema=StructuredLocalOutput,
    use_async=True,
)

# Instantiate the remote OpenAI client
remote_client = OpenAIClient(
    model_name="gpt-4o",
)

# Instantiate the Minions protocol with both clients
minions = Minions(local_client, remote_client)

context = """
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

# Execute the minions protocol for up to two communication rounds
output = minions(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
