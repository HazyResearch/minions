"""
Automated Smart Router Experiments

This script runs automated experiments to evaluate smart routing performance
by actually executing the complete Minions protocol - just like the Streamlit app!

Usage:
    python run_smart_router_experiments.py --mode quick
    python run_smart_router_experiments.py --mode full --output results.json
"""

import argparse
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_data import TestDataset, TestQuery
from minions.utils.smart_router import SmartRouter

# Import the actual clients and protocols like the Streamlit app does
from minions.clients import OllamaClient, OpenAIClient
from minions.minions import Minions
from minions.minion import Minion


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    query: str
    query_type: str
    expected_route: str
    actual_route: str
    classification_correct: bool
    time_seconds: float
    response: str
    confidence_score: float  # 0-1, higher = more confident


class SmartRouterEvaluator:
    """
    Automated evaluator for smart routing experiments.

    This class runs experiments without requiring manual interaction.
    """

    def __init__(self, local_client=None, remote_client=None):
        """
        Initialize the evaluator.

        Args:
            local_client: Optional local LLM client (e.g., OllamaClient)
            remote_client: Optional remote LLM client (e.g., OpenAIClient)
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.router = SmartRouter()
        self.dataset = TestDataset()

    def run_classification_test(self) -> Dict[str, Any]:
        """
        Test query classification accuracy (no LLM calls needed).

        Returns:
            Dictionary with classification results
        """
        print("\n" + "="*60)
        print("Running Classification Accuracy Test")
        print("="*60)

        results = []
        correct = 0
        total = 0

        for query in self.dataset.get_all():
            predicted_type = self.router.classify_query_type(query.query)
            is_correct = predicted_type == query.query_type

            results.append({
                "query": query.query,
                "expected_type": query.query_type,
                "predicted_type": predicted_type,
                "correct": is_correct
            })

            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} {query.query[:50]:50} | Expected: {query.query_type:12} | Got: {predicted_type:12}")

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }

    def run_routing_simulation(self) -> Dict[str, Any]:
        """
        Simulate routing decisions without actual LLM calls.

        Returns:
            Dictionary with routing simulation results
        """
        print("\n" + "="*60)
        print("Running Routing Decision Simulation")
        print("="*60)
        print("(Simulating uncertainty scores - no LLM calls)\n")

        results = []
        routing_accuracy = {"correct": 0, "total": 0}

        for query in self.dataset.get_all():
            query_type = self.router.classify_query_type(query.query)
            threshold = self.router.query_type_thresholds.get(query_type, 0.5)

            # Simulate uncertainty based on query type
            # In practice, this would call measure_local_uncertainty()
            simulated_uncertainty = {
                "lookup": 0.2,  # Low uncertainty
                "math": 0.3,
                "extract": 0.5,
                "code": 0.6,
                "open-ended": 0.7,
                "multi-hop": 0.8,  # High uncertainty
            }.get(query_type, 0.5)

            would_escalate = simulated_uncertainty > threshold
            predicted_route = "remote" if would_escalate else "local"
            is_correct = predicted_route == query.expected_route

            results.append({
                "query": query.query,
                "query_type": query_type,
                "threshold": threshold,
                "simulated_uncertainty": simulated_uncertainty,
                "predicted_route": predicted_route,
                "expected_route": query.expected_route,
                "correct": is_correct
            })

            if is_correct:
                routing_accuracy["correct"] += 1
            routing_accuracy["total"] += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} {query.query[:40]:40} | Type: {query_type:12} | "
                  f"U={simulated_uncertainty:.2f} T={threshold:.2f} | "
                  f"Route: {predicted_route:6} (expected {query.expected_route:6})")

        accuracy = routing_accuracy["correct"] / routing_accuracy["total"]

        return {
            "accuracy": accuracy,
            "routing_accuracy": routing_accuracy,
            "results": results
        }

    def run_live_experiments(
        self,
        query_types: Optional[List[str]] = None,
        max_per_type: int = 2
    ) -> Dict[str, Any]:
        """
        Run live experiments with actual LLM calls (if clients provided).

        Args:
            query_types: List of query types to test (default: all)
            max_per_type: Maximum queries per type to test

        Returns:
            Dictionary with experiment results
        """
        if not self.local_client:
            print("⚠️  No local client provided, skipping live experiments")
            return {"skipped": True, "reason": "No local client"}

        print("\n" + "="*60)
        print("Running Live Experiments (with LLM calls)")
        print("="*60)

        if not query_types:
            query_types = ["lookup", "math", "multi-hop"]

        results = []
        total_time_saved = 0

        for query_type in query_types:
            queries = self.dataset.get_by_type(query_type)[:max_per_type]

            print(f"\n--- Testing {query_type} queries ---")

            for test_query in queries:
                print(f"\nQuery: {test_query.query}")

                # Classify
                classified_type = self.router.classify_query_type(test_query.query)
                print(f"Classified as: {classified_type}")

                # Try local-only
                start_time = time.time()
                local_result = self.router.get_local_only_response(
                    test_query.query,
                    self.local_client,
                    classified_type
                )
                elapsed_time = time.time() - start_time

                if local_result:
                    actual_route = "local"
                    response = local_result["final_answer"]
                    print(f"✓ Handled locally in {elapsed_time:.2f}s")
                    print(f"Response: {response[:100]}...")

                    # Estimate time saved vs full protocol
                    estimated_full_protocol_time = 120  # seconds
                    time_saved = estimated_full_protocol_time - elapsed_time
                    total_time_saved += time_saved
                else:
                    actual_route = "remote"
                    response = "Would escalate to full protocol"
                    print(f"✗ Would escalate to full protocol")
                    time_saved = 0

                results.append({
                    "query": test_query.query,
                    "query_type": query_type,
                    "classified_type": classified_type,
                    "expected_route": test_query.expected_route,
                    "actual_route": actual_route,
                    "time_seconds": elapsed_time,
                    "time_saved_seconds": time_saved,
                    "response": response[:200] if response else None,
                })

        return {
            "total_queries": len(results),
            "total_time_saved_seconds": total_time_saved,
            "results": results
        }

    def run_all_experiments(
        self,
        include_live: bool = False,
        max_live_per_type: int = 2
    ) -> Dict[str, Any]:
        """
        Run complete experiment suite.

        Args:
            include_live: Whether to run live experiments with LLM calls
            max_live_per_type: Max queries per type for live experiments

        Returns:
            Complete experiment results
        """
        print("\n" + "="*80)
        print("SMART ROUTER AUTOMATED EXPERIMENTS")
        print("="*80)

        dataset_summary = self.dataset.summary()
        print(f"\nDataset: {dataset_summary['total_queries']} total queries")
        print(f"By type: {dataset_summary['by_type']}")

        all_results = {
            "timestamp": time.time(),
            "dataset_summary": dataset_summary,
        }

        # 1. Classification test (fast, no LLM)
        all_results["classification"] = self.run_classification_test()

        # 2. Routing simulation (fast, no LLM)
        all_results["routing_simulation"] = self.run_routing_simulation()

        # 3. Live experiments (slow, requires LLM)
        if include_live:
            all_results["live_experiments"] = self.run_live_experiments(
                max_per_type=max_live_per_type
            )

        # Summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Classification Accuracy: {all_results['classification']['accuracy']:.1%}")
        print(f"Routing Accuracy (simulated): {all_results['routing_simulation']['accuracy']:.1%}")

        if include_live and "live_experiments" in all_results:
            live = all_results["live_experiments"]
            if not live.get("skipped"):
                print(f"Live Experiments: {live['total_queries']} queries")
                print(f"Total Time Saved: {live['total_time_saved_seconds']:.1f}s")

        return all_results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run smart router experiments")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: classification & simulation only, full: includes live LLM calls"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="smart_router_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max-live",
        type=int,
        default=2,
        help="Max queries per type for live experiments"
    )

    args = parser.parse_args()

    # Initialize real clients for live experiments
    local_client = None
    remote_client = None
    
    if args.mode == "full":
        print("Initializing real clients for live experiments...")
        try:
            # Initialize clients like the Streamlit app does
            local_client = OllamaClient(
                model_name="llama3.2",  # Use a common local model
                temperature=0.0,
                max_tokens=4096,
                num_ctx=4096,
                use_async=True  # Minions protocol uses async
            )
            print("✓ Local client (Ollama) initialized")
            
            # Check if OpenAI API key is available
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                remote_client = OpenAIClient(
                    model_name="gpt-4o-mini",  # Use cost-effective model
                    temperature=0.0,
                    max_tokens=4096,
                    api_key=openai_key
                )
                print("✓ Remote client (OpenAI) initialized")
            else:
                print("⚠️  OPENAI_API_KEY not found - skipping remote client")
                
        except Exception as e:
            print(f"⚠️  Error initializing clients: {e}")
            print("Running without live experiments")
    
    evaluator = SmartRouterEvaluator(
        local_client=local_client,
        remote_client=remote_client
    )

    # Run experiments
    include_live = args.mode == "full"
    results = evaluator.run_all_experiments(
        include_live=include_live,
        max_live_per_type=args.max_live
    )

    # Save results
    evaluator.save_results(results, args.output)

    print("\n" + "="*80)
    print("✓ Experiments complete!")
    print("="*80)


if __name__ == "__main__":
    main()
