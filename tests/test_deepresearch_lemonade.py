from minions.clients.lemonade import LemonadeClient
from minions.clients.openai import OpenAIClient
from minions.minions_deep_research import DeepResearchMinions, JobOutput

# Initialize Lemonade as the local client (sync mode, GGUF model recommended for best results)
local_client = LemonadeClient(
    model_name="Qwen3-8B-GGUF",
    temperature=0.2,
    max_tokens=2048,
    structured_output_schema=JobOutput,
    use_async=False  # DeepResearchMinions uses sync by default
)

# Initialize OpenAI as the remote client
remote_client = OpenAIClient(
    model_name="gpt-4o"
)

# Instantiate DeepResearchMinions with both clients
minion = DeepResearchMinions(
    local_client=local_client,
    remote_client=remote_client,
    max_rounds=3,
    max_sources_per_round=5
)

# Example research query
query = (
    "Explain how Anthropic's MCP works?"
)

# Run the DeepResearch protocol
if __name__ == "__main__":
    output, visited_urls = minion(
        query=query,
    )
    print("=== DeepResearch Output ===")
    print(output)
    print("\nVisited URLs:")
    print(visited_urls)
