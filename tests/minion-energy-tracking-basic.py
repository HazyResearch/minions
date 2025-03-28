from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

from minions.utils.energy_tracking import PowerMonitor, cloud_inference_energy_estimate

# Setup power monitor
monitor = PowerMonitor(mode="auto", interval=1.0)

local_client = OllamaClient(
        model_name="llama3.2",
    )

remote_client = OpenAIClient(
        model_name="gpt-4o",
    )

# Instantiate the Minion object with both clients
minion = Minion(local_client, remote_client)


context = """
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

# Execute the minion protocol for up to two communication rounds
monitor.start()
output = minion(
    task=task,
    context=[context],
    max_rounds=2
)
monitor.stop()

final_estimates = monitor.get_final_estimates()

# Display energy consumption
print("\nMeasured Energy and Power Metrics --- (Minions) Local Client")
for key, value in final_estimates.items():
    print(f"{key}: {value}")

# Estimate energy consumption of workload done fully in cloud
local_tokens = output["local_usage"].completion_tokens + output["local_usage"].prompt_tokens
power_estimate, energy_estimate, _ = cloud_inference_energy_estimate(tokens=local_tokens)

print(f"(Baseline) Remote Only Energy Estimate: {energy_estimate} J")

for x in output.keys():
    print(f"\n--- {x} ---")
    print(output[x])