import json
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

def load_test_cases(filepath):
    """Load test cases from a JSON file."""
    with open(filepath, "r") as f:
        test_cases = json.load(f)
    return test_cases

def run_tests(test_cases, minion, max_rounds=2):
    """Run each test case using the minion protocol."""
    results = []
    for idx, test in enumerate(test_cases):
        context = test["Context"]
        # Use the "Query" field as the task for testing
        task = test["Query"]

        print(f"Running test case {idx+1}: {test['One sentence summary of the context']}")
        # Execute the minion protocol
        output = minion(
            task=task,
            context=[context],
            max_rounds=max_rounds
        )

        # Save output along with test case details
        result = {
            "test_case": idx + 1,
            "context_summary": test["One sentence summary of the context"],
            "task": task,
            "output": output
        }
        results.append(result)

        # Print a summary of the final answer for quick review
        if "final_answer" in output:
            print(f"Final Answer: {output['final_answer']}\n")
        else:
            print(f"No final answer provided.\n")
    return results

if __name__ == '__main__':
    # Instantiate the clients
    local_client = OllamaClient(model_name="llama3.2")
    remote_client = OpenAIClient(model_name="gpt-4o")

    # Instantiate the Minion object with both clients
    minion = Minion(local_client, remote_client)

    # Load test cases from a JSON file (e.g., 'test_cases.json')
    test_cases = load_test_cases("test_cases.json")

    # Run tests (you can adjust max_rounds if needed)
    results = run_tests(test_cases, minion, max_rounds=2)

    # Write results to a JSON file
    import jsonpickle

    with open("test_results_post.json", "w") as f:
        f.write(jsonpickle.encode(results, indent=2))