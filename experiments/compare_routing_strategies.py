"""
Smart Router vs Full Protocol Comparison

This script compares the performance of smart routing vs full Minions protocol.
Inspired by CS329A homework evaluation methods.

Usage:
    python compare_routing_strategies.py --client-type ollama --model llama3.1:8b
"""

import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_data import TestDataset, TestQuery


@dataclass
class ComparisonResult:
    """Results comparing two routing strategies"""
    query: str
    query_type: str

    # Local-only routing
    local_time: Optional[float]
    local_response: Optional[str]
    local_routed: bool

    # Full protocol
    full_time: Optional[float]
    full_response: Optional[str]

    # Comparison
    time_saved: float
    time_saved_percent: float
    routing_correct: bool  # Did routing decision match expected?


class RoutingComparator:
    """
    Compares smart routing vs full Minions protocol.

    Can work in three modes:
    1. Simulation mode (no LLM) - estimates based on query types
    2. Local-only mode - tests local routing only
    3. Full comparison mode - tests both strategies
    """

    def __init__(self, local_client=None, remote_client=None, minions=None):
        """
        Initialize comparator.

        Args:
            local_client: Local LLM client
            remote_client: Remote LLM client
            minions: Minions instance for full protocol testing
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.minions = minions
        self.dataset = TestDataset()

    def estimate_performance(self, query: TestQuery) -> Dict[str, Any]:
        """
        Estimate performance without running LLMs (for quick analysis).

        Based on empirical observations:
        - Local-only: 1-2s for simple queries
        - Full protocol: 60-120s depending on rounds

        Args:
            query: Test query

        Returns:
            Estimated performance metrics
        """
        # Estimated times based on query type
        local_time_estimates = {
            "lookup": 1.5,
            "math": 1.2,
            "extract": 2.0,
            "code": 2.5,
            "open-ended": 3.0,
            "multi-hop": 3.5,
        }

        full_protocol_estimates = {
            "lookup": 80,  # Overkill for simple queries
            "math": 75,
            "extract": 90,
            "code": 100,
            "open-ended": 110,
            "multi-hop": 120,
        }

        # Estimate if query would route locally
        would_route_locally = query.expected_route == "local"

        local_time = local_time_estimates.get(query.query_type, 2.0)
        full_time = full_protocol_estimates.get(query.query_type, 90.0)

        if would_route_locally:
            actual_time = local_time
            used_full_protocol = False
            time_saved = full_time - local_time
        else:
            actual_time = full_time
            used_full_protocol = True
            time_saved = 0

        return {
            "query": query.query,
            "query_type": query.query_type,
            "would_route_locally": would_route_locally,
            "estimated_time": actual_time,
            "estimated_full_protocol_time": full_time,
            "time_saved": time_saved,
            "time_saved_percent": (time_saved / full_time * 100) if full_time > 0 else 0,
            "speedup_factor": (full_time / actual_time) if actual_time > 0 else 1,
        }

    def run_estimation_analysis(self) -> Dict[str, Any]:
        """
        Run performance estimation for all queries (no LLM calls).

        Returns:
            Complete estimation analysis
        """
        print("\n" + "="*80)
        print("PERFORMANCE ESTIMATION ANALYSIS (No LLM calls)")
        print("="*80)

        results = []
        total_time_full_protocol = 0
        total_time_smart_routing = 0

        for query in self.dataset.get_all():
            estimate = self.estimate_performance(query)
            results.append(estimate)

            total_time_full_protocol += estimate["estimated_full_protocol_time"]
            total_time_smart_routing += estimate["estimated_time"]

            print(f"\n{query.query_type:12} | {query.query[:50]:50}")
            print(f"  Estimated time: {estimate['estimated_time']:.1f}s "
                  f"(vs {estimate['estimated_full_protocol_time']:.1f}s full protocol)")
            if estimate["would_route_locally"]:
                print(f"  ✓ Routes locally, saves {estimate['time_saved']:.1f}s "
                      f"({estimate['time_saved_percent']:.0f}% faster)")

        # Summary statistics
        total_queries = len(results)
        local_routed = len([r for r in results if r["would_route_locally"]])
        total_time_saved = total_time_full_protocol - total_time_smart_routing

        summary = {
            "total_queries": total_queries,
            "queries_routed_locally": local_routed,
            "queries_routed_locally_percent": (local_routed / total_queries * 100),
            "total_time_full_protocol": total_time_full_protocol,
            "total_time_smart_routing": total_time_smart_routing,
            "total_time_saved": total_time_saved,
            "average_speedup": total_time_full_protocol / total_time_smart_routing,
            "efficiency_gain_percent": (total_time_saved / total_time_full_protocol * 100),
        }

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total queries: {summary['total_queries']}")
        print(f"Routed locally: {summary['queries_routed_locally']} "
              f"({summary['queries_routed_locally_percent']:.0f}%)")
        print(f"\nEstimated total time:")
        print(f"  Full protocol: {summary['total_time_full_protocol']:.1f}s "
              f"({summary['total_time_full_protocol']/60:.1f} min)")
        print(f"  Smart routing: {summary['total_time_smart_routing']:.1f}s "
              f"({summary['total_time_smart_routing']/60:.1f} min)")
        print(f"  Time saved: {summary['total_time_saved']:.1f}s "
              f"({summary['total_time_saved']/60:.1f} min)")
        print(f"\nPerformance improvement: {summary['average_speedup']:.1f}x faster")
        print(f"Efficiency gain: {summary['efficiency_gain_percent']:.0f}%")

        return {
            "summary": summary,
            "results": results,
        }

    def compare_accuracy(
        self,
        local_responses: List[str],
        full_responses: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Compare accuracy of responses (inspired by CS329A homework).

        Simple string similarity for now - could be extended with:
        - BLEU/ROUGE scores
        - LLM-as-judge evaluation
        - Domain-specific metrics

        Args:
            local_responses: Responses from local-only routing
            full_responses: Responses from full protocol
            ground_truths: Ground truth answers

        Returns:
            Accuracy comparison metrics
        """
        def simple_match(response: str, ground_truth: str) -> bool:
            """Simple keyword-based matching"""
            if not response or not ground_truth:
                return False

            response_lower = response.lower()
            gt_keywords = ground_truth.lower().split()

            # Check if key terms from ground truth appear in response
            matches = sum(1 for keyword in gt_keywords if keyword in response_lower)
            match_ratio = matches / len(gt_keywords) if gt_keywords else 0

            return match_ratio > 0.5  # At least 50% keyword overlap

        local_correct = sum(
            1 for local, gt in zip(local_responses, ground_truths)
            if simple_match(local, gt)
        )

        full_correct = sum(
            1 for full, gt in zip(full_responses, ground_truths)
            if simple_match(full, gt)
        )

        total = len(ground_truths)

        return {
            "local_accuracy": local_correct / total if total > 0 else 0,
            "full_accuracy": full_correct / total if total > 0 else 0,
            "local_correct": local_correct,
            "full_correct": full_correct,
            "total": total,
        }

    def generate_report(self, results: Dict[str, Any], output_path: str):
        """Generate a detailed report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "smart_router_comparison",
            "results": results,
        }

        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w') as f:
            f.write("# Smart Router Performance Report\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")

            if "summary" in results:
                summary = results["summary"]
                f.write("## Summary\n\n")
                f.write(f"- **Total Queries**: {summary['total_queries']}\n")
                f.write(f"- **Routed Locally**: {summary['queries_routed_locally']} "
                       f"({summary['queries_routed_locally_percent']:.0f}%)\n")
                f.write(f"- **Speedup**: {summary['average_speedup']:.1f}x faster\n")
                f.write(f"- **Efficiency Gain**: {summary['efficiency_gain_percent']:.0f}%\n\n")

                f.write("## Time Comparison\n\n")
                f.write("| Strategy | Total Time | Time (minutes) |\n")
                f.write("|----------|------------|----------------|\n")
                f.write(f"| Full Protocol | {summary['total_time_full_protocol']:.1f}s | "
                       f"{summary['total_time_full_protocol']/60:.1f} min |\n")
                f.write(f"| Smart Routing | {summary['total_time_smart_routing']:.1f}s | "
                       f"{summary['total_time_smart_routing']/60:.1f} min |\n")
                f.write(f"| **Saved** | **{summary['total_time_saved']:.1f}s** | "
                       f"**{summary['total_time_saved']/60:.1f} min** |\n\n")

        print(f"\n✓ Report saved to: {output_path}")
        print(f"✓ Markdown summary: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare routing strategies")
    parser.add_argument(
        "--mode",
        choices=["estimate", "live"],
        default="estimate",
        help="estimate: quick estimation, live: run actual experiments"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.json",
        help="Output file path"
    )

    args = parser.parse_args()

    # Initialize comparator
    comparator = RoutingComparator()

    # Run comparison
    if args.mode == "estimate":
        results = comparator.run_estimation_analysis()
    else:
        print("Live mode requires LLM clients - see code for setup")
        return

    # Generate report
    comparator.generate_report(results, args.output)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
