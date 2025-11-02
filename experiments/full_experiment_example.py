"""
Full Smart Router Experiment with Real LLM Clients

This is a complete example showing how to run experiments with actual
local and remote LLM clients.

Before running:
1. Install required clients: pip install openai ollama
2. Ensure Ollama is running locally (for local client)
3. Set OPENAI_API_KEY environment variable (for remote client)

Usage:
    python experiments/full_experiment_example.py
"""

import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.minions import Minions
from minions.utils.smart_router import SmartRouter
from experiments.test_data import TestDataset


def run_with_minions_client():
    """
    Example using actual Minions with local and remote clients.

    This demonstrates the smart routing in action with real LLM calls.
    """
    print("\n" + "="*80)
    print("FULL EXPERIMENT WITH REAL CLIENTS")
    print("="*80)

    # ===== STEP 1: Initialize Clients =====
    print("\nStep 1: Initializing clients...")

    try:
        # Local client (using Ollama)
        from minions.clients import OllamaClient
        local_client = OllamaClient(
            model="llama3.1:8b",
            base_url="http://localhost:11434"
        )
        print("✓ Local client (Ollama) initialized")
    except Exception as e:
        print(f"✗ Could not initialize local client: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return

    try:
        # Remote client (using OpenAI) - optional for this test
        # from minions.clients import OpenAIClient
        # remote_client = OpenAIClient(model="gpt-4o-mini")
        remote_client = None
        print("  Remote client: Not configured (optional for this demo)")
    except Exception as e:
        print(f"  Remote client: {e}")
        remote_client = None

    # ===== STEP 2: Initialize Minions with Smart Routing =====
    print("\nStep 2: Initializing Minions with smart routing...")

    minion = Minions(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=3
    )
    print("✓ Minions initialized (smart routing automatic)")

    # ===== STEP 3: Load Test Data =====
    print("\nStep 3: Loading test queries...")

    dataset = TestDataset()

    # Select a subset for quick testing
    test_queries = [
        *dataset.get_by_type("lookup")[:2],  # 2 lookup queries
        *dataset.get_by_type("math")[:1],    # 1 math query
        *dataset.get_by_type("multi-hop")[:1] # 1 complex query
    ]

    print(f"✓ Loaded {len(test_queries)} test queries")

    # ===== STEP 4: Run Experiments =====
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)

    results = []
    total_time = 0
    local_routed = 0

    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"Type: {test_query.query_type}")
        print(f"Query: {test_query.query}")
        print(f"Expected route: {test_query.expected_route}")

        # Run query through Minions (smart routing happens automatically)
        start_time = time.time()

        try:
            result = minion(
                task=test_query.query,
                doc_metadata="Test query - no document",
                context=test_query.context if test_query.context else []
            )

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # Check if it was routed locally
            routing_info = result.get('meta', [{}])[0].get('routing', None)
            if routing_info:
                actual_route = "local"
                local_routed += 1
                print(f"\n✓ Routed locally!")
                print(f"  Decision: {routing_info['decision']}")
                print(f"  Time saved: {routing_info.get('time_saved', 'N/A')}")
            else:
                actual_route = "remote"
                print(f"\n→ Used full protocol")

            print(f"\nTime: {elapsed_time:.2f}s")
            print(f"Answer: {result['final_answer'][:200]}...")

            # Save results
            results.append({
                "query": test_query.query,
                "query_type": test_query.query_type,
                "expected_route": test_query.expected_route,
                "actual_route": actual_route,
                "time_seconds": elapsed_time,
                "answer": result['final_answer'],
                "routing_correct": actual_route == test_query.expected_route
            })

        except Exception as e:
            print(f"\n✗ Error: {e}")
            results.append({
                "query": test_query.query,
                "error": str(e)
            })

    # ===== STEP 5: Print Summary =====
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    successful_results = [r for r in results if "error" not in r]

    print(f"\nTotal queries: {len(test_queries)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Routed locally: {local_routed}/{len(successful_results)} "
          f"({local_routed/len(successful_results)*100:.0f}%)")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average time per query: {total_time/len(successful_results):.2f}s")

    # Routing accuracy
    correct_routing = sum(1 for r in successful_results if r.get("routing_correct", False))
    print(f"\nRouting accuracy: {correct_routing}/{len(successful_results)} "
          f"({correct_routing/len(successful_results)*100:.0f}%)")

    # Detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    for i, result in enumerate(successful_results, 1):
        print(f"\n{i}. {result['query_type'].upper()}")
        print(f"   Query: {result['query'][:60]}...")
        print(f"   Routed: {result['actual_route']} (expected: {result['expected_route']})")
        print(f"   Time: {result['time_seconds']:.2f}s")
        status = "✓" if result['routing_correct'] else "✗"
        print(f"   Routing: {status}")

    return results


def run_direct_smart_router():
    """
    Example using SmartRouter directly (without full Minions protocol).

    This is faster and simpler for testing just the routing logic.
    """
    print("\n" + "="*80)
    print("DIRECT SMART ROUTER TEST")
    print("="*80)

    try:
        from minions.clients import OllamaClient
        local_client = OllamaClient(
            model="llama3.1:8b",
            base_url="http://localhost:11434"
        )
        print("✓ Local client initialized")
    except Exception as e:
        print(f"✗ Could not initialize local client: {e}")
        return

    router = SmartRouter()
    dataset = TestDataset()

    # Test a few simple queries
    test_queries = dataset.get_by_type("lookup")[:2]

    print(f"\nTesting {len(test_queries)} queries...\n")

    for query in test_queries:
        print(f"\nQuery: {query.query}")

        # Classify
        query_type = router.classify_query_type(query.query)
        print(f"Classification: {query_type}")

        # Try to get local-only response
        start_time = time.time()
        result = router.get_local_only_response(
            query.query,
            local_client,
            query_type
        )
        elapsed_time = time.time() - start_time

        if result:
            print(f"✓ Handled locally in {elapsed_time:.2f}s")
            print(f"Response: {result['final_answer'][:150]}...")
        else:
            print(f"✗ Would escalate to remote")


def main():
    """
    Run the experiments.

    Uncomment the version you want to run:
    1. run_direct_smart_router() - Quick test of smart router only
    2. run_with_minions_client() - Full Minions protocol with smart routing
    """
    print("\n" + "="*80)
    print("SMART ROUTER LIVE EXPERIMENT")
    print("="*80)
    print("\nChoose experiment mode:")
    print("1. Direct Smart Router (fast, local only)")
    print("2. Full Minions Protocol (slow, complete system)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        run_direct_smart_router()
    elif choice == "2":
        run_with_minions_client()
    else:
        print("Invalid choice. Running direct smart router test...")
        run_direct_smart_router()


if __name__ == "__main__":
    main()
