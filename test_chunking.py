from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minions import Minions
from pydantic import BaseModel
import time
import os 
import json

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None


local_client = OllamaClient(
    model_name="llama3.2",
    structured_output_schema=StructuredLocalOutput,
    temperature=0.0
)

remote_client = OpenAIClient(
    model_name="gpt-4o",
)


# Instantiate the Minion object with both clients
minion = Minions(local_client, remote_client, use_bm25=False)

def run_experiment(doc: str, task: str, metadata: str) -> dict:
    start_time = time.perf_counter()
    output = minion(
        task=task,
        doc_metadata=metadata,
        context=[doc],
        max_rounds=2
    )
    end_time = time.perf_counter()

    results = {
        "output": output,
        "processing_time": end_time - start_time,
    }
    return results

def main():
    root_dir = "minions/examples"
    subfolders = {"code": "Python code document", "finance": "Finance report document", "health": "Health report document"}

    all_results = {}
    
    for folder, metadata in subfolders.items():
        folder_path = os.path.join(root_dir, folder)
        doc_path = os.path.join(folder_path, "sample.txt")
        task_path = os.path.join(folder_path, "task.json")
        
        with open(doc_path, "r", encoding="utf-8") as f:
            doc = f.read()
        
        with open(task_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        
        question = task_data.get("question", "")
        known_answer = task_data.get("answer", [])
        
        print(f"Processing folder: {folder}")
        result = run_experiment(doc, question, metadata)
        
        all_results[folder] = {
            "question": question,
            "known_answer": known_answer,
            "result": result
        }
        
        print(f"Finished processing {folder}.")
        print("-" * 60)
    
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, default=str)

if __name__ == "__main__":
    main()