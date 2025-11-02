"""
Smart Router Usage Example

This example demonstrates how to use the Minions protocol with smart routing.
The smart router automatically decides whether to use local-only processing
or escalate to the full distributed protocol.
"""

# Example usage with Minions protocol
"""
from minions.minions import Minions
from minions.clients import OllamaClient, OpenAIClient

# Initialize clients
local_client = OllamaClient(model="llama3.1:8b")
remote_client = OpenAIClient(model="gpt-4")

# Initialize Minions (smart routing is automatic)
minion = Minions(
    local_client=local_client,
    remote_client=remote_client,
    max_rounds=5
)

# Example 1: Simple lookup query (should use local-only)
print("\\n" + "="*60)
print("Example 1: Lookup Query (Expected: Local-only)")
print("="*60)

result = minion(
    task="What is a cartridge",
    doc_metadata="General knowledge query",
    context=[]
)

print(f"Final Answer: {result['final_answer']}")
print(f"Routing: {result['meta'][0].get('routing', 'Full protocol')}")
print(f"Time: {result['timing']['total_time']:.2f}s")


# Example 2: Complex reasoning query (should escalate)
print("\\n" + "="*60)
print("Example 2: Multi-hop Query (Expected: Full protocol)")
print("="*60)

result = minion(
    task="Analyze the economic implications of widespread AI adoption and recommend policy interventions",
    doc_metadata="Policy analysis",
    context=["...relevant context documents..."]
)

print(f"Final Answer: {result['final_answer'][:100]}...")
print(f"Routing: {result['meta'][0].get('routing', 'Full protocol')}")
print(f"Time: {result['timing']['total_time']:.2f}s")


# Example 3: Math query (should try local first)
print("\\n" + "="*60)
print("Example 3: Math Query (Expected: Local-only if confident)")
print("="*60)

result = minion(
    task="Calculate the average of 15, 27, and 33",
    doc_metadata="Math calculation",
    context=[]
)

print(f"Final Answer: {result['final_answer']}")
print(f"Routing: {result['meta'][0].get('routing', 'Full protocol')}")
print(f"Time: {result['timing']['total_time']:.2f}s")
"""


# Direct Smart Router usage (without full Minions protocol)
from minions.utils.smart_router import SmartRouter

print("\n" + "="*60)
print("Direct Smart Router Usage")
print("="*60)

router = SmartRouter()

# Test different query types
queries = [
    "What is a cartridge",
    "Analyze the implications of quantum computing",
    "Calculate 15 + 27",
    "Extract all names from the document",
]

for query in queries:
    query_type = router.classify_query_type(query)
    threshold = router.query_type_thresholds.get(query_type, 0.5)

    print(f"\nQuery: '{query}'")
    print(f"  Type: {query_type}")
    print(f"  Threshold: {threshold}")
    print(f"  Strategy: {'Try local first' if threshold > 0.5 else 'Prefer remote'}")


print("\n" + "="*60)
print("Key Benefits of Smart Routing:")
print("="*60)
print("""
1. Performance: ~99% faster for simple queries (1-2s vs 120s+)
2. Cost: Zero remote API costs for local-only responses
3. Automatic: No manual configuration required
4. Fallback: Automatically escalates when local model is uncertain
5. Transparent: Routing decisions are logged and visible

Query Type Routing Strategy:
  - Lookup/Math: Conservative, tries local first (thresholds: 0.7-0.8)
  - Extract/Code: Moderate approach (thresholds: 0.5-0.6)
  - Multi-hop/Analysis: Quick escalation to remote (thresholds: 0.3-0.4)
""")
