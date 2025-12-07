"""
Enhanced Smart Routing Evaluation with Formal Complexity Analysis

This script evaluates routing decisions using:
1. Complexity Scorer (Erotetic logic, Bloom's taxonomy, 6 dimensions)
2. Logprobs Uncertainty (if available from API)
3. Self-Consistency Uncertainty (sampling-based)

Usage:
    python experiments/enhanced_routing_eval.py --max-queries 5
"""

import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_data import TestDataset, TestQuery
from minions.minions import Minions
from minions.utils.complexity_scorer import ComplexityScorer, ComplexityScore
from minions.utils.logprobs_uncertainty import LogprobsUncertaintyEstimator


@dataclass
class RoutingMetrics:
    """Metrics for a single routing decision"""
    query: str
    query_type: str

    # Complexity analysis
    complexity_score: float
    erotetic_type: str
    bloom_level: str
    reasoning_depth: int

    # Uncertainty measures
    self_consistency_uncertainty: Optional[float] = None
    logprobs_uncertainty: Optional[float] = None

    # Routing decision
    recommended_route: str = ""
    confidence: float = 0.0

    # Timing
    decision_time: float = 0.0


@dataclass
class ExperimentResult:
    """Result for one query"""
    query: str
    query_type: str

    # Routing analysis
    routing_metrics: RoutingMetrics

    # Local response
    local_answer: str
    local_time: float
    local_used_route: bool

    # Full protocol response
    full_answer: str
    full_time: float
    full_error: Optional[str] = None

    # Judge evaluation
    judge_choice: Optional[int] = None
    judge_reason: str = ""
    winner: str = ""

    # Ground truth
    ground_truth: str = ""

    # Performance
    time_saved: float = 0.0
    speedup_factor: float = 0.0


class LLMJudge:
    """LLM-as-a-judge for comparing answers"""

    def __init__(self, judge_client):
        self.judge_client = judge_client

    def judge(self, query: str, answer_a: str, answer_b: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Compare two answers"""
        gt_context = f"\n\nGround Truth Reference:\n{ground_truth}\n" if ground_truth else ""

        prompt = f"""You are an expert evaluator comparing two AI-generated answers.

Question:
{query}
{gt_context}

Candidate 0 (Smart Routing):
{answer_a}

Candidate 1 (Full Protocol):
{answer_b}

Instructions:
- Carefully analyze both answers for correctness, completeness, and accuracy
- Consider which answer better addresses the question
- If both are equally good or both wrong, return null
- If one has errors and the other doesn't, prefer the correct one

Response Format (JSON only, no other text):
{{"choice": <0, 1, or null>, "reason": "<one sentence explanation>"}}

Your response:"""

        try:
            result = self.judge_client.chat([{"role": "user", "content": prompt}])
            if len(result) == 3:
                response, usage, _ = result
            else:
                response, usage = result

            raw = response[0]
            choice, reason = self._parse_json(raw)

            return {"choice": choice, "reason": reason, "raw": raw}
        except Exception as e:
            print(f"Judge error: {e}")
            return {"choice": None, "reason": f"Error: {str(e)}", "raw": ""}

    def _parse_json(self, text: str) -> Tuple[Optional[int], str]:
        """Parse JSON response"""
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                choice = data.get("choice")
                if choice == "null" or choice == "None":
                    choice = None
                elif choice is not None:
                    choice = int(choice)
                reason = data.get("reason", "")
                return choice, reason
        except:
            pass
        return None, "Could not parse response"


class EnhancedRoutingEvaluator:
    """Evaluator using enhanced routing with formal complexity analysis"""

    def __init__(
        self,
        local_client,
        remote_client,
        judge_client,
        use_logprobs: bool = True,
        self_consistency_k: int = 3
    ):
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge = LLMJudge(judge_client)

        # Create separate client for self-consistency with temperature > 0
        from minions.clients import OllamaClient
        self.sc_client = OllamaClient(model_name="llama3.2", temperature=0.7, max_tokens=200, use_async=False)

        # Initialize enhanced components
        self.complexity_scorer = ComplexityScorer()
        self.logprobs_estimator = LogprobsUncertaintyEstimator() if use_logprobs else None
        self.self_consistency_k = self_consistency_k

        # Adaptive thresholds (from enhanced router)
        self.thresholds = {
            'factual': 0.7,
            'conceptual': 0.6,
            'procedural': 0.55,
            'causal': 0.5,
            'comparative': 0.45,
            'evaluative': 0.4,
            'synthetic': 0.3,
            'metacognitive': 0.4
        }

    def analyze_query(self, query: str, context: Optional[List[str]] = None) -> RoutingMetrics:
        """Analyze query using formal complexity scoring"""
        start_time = time.time()

        # 1. Complexity scoring
        complexity_score = self.complexity_scorer.score(query, context)

        # 2. Self-consistency uncertainty
        sc_uncertainty = self._measure_self_consistency(query, context)

        # 3. Logprobs uncertainty (placeholder - would need API support)
        logprobs_unc = None  # TODO: Extract from API if available

        # 4. Make routing decision
        threshold = self.thresholds.get(complexity_score.erotetic_type.value, 0.5)

        # Combined score (complexity + uncertainty)
        combined_score = (
            0.6 * complexity_score.overall +
            0.4 * (sc_uncertainty if sc_uncertainty else 0)
        )

        if combined_score < threshold:
            route = "local"
            confidence = (threshold - combined_score) / threshold
        else:
            route = "remote"
            confidence = (combined_score - threshold) / (1.0 - threshold)

        decision_time = time.time() - start_time

        return RoutingMetrics(
            query=query,
            query_type=complexity_score.erotetic_type.value,
            complexity_score=complexity_score.overall,
            erotetic_type=complexity_score.erotetic_type.value,
            bloom_level=f"{complexity_score.bloom_level.name} (L{complexity_score.bloom_level.value})",
            reasoning_depth=complexity_score.reasoning_depth,
            self_consistency_uncertainty=sc_uncertainty,
            logprobs_uncertainty=logprobs_unc,
            recommended_route=route,
            confidence=confidence,
            decision_time=decision_time
        )

    def _measure_self_consistency(self, query: str, context: Optional[List[str]] = None) -> float:
        """Measure uncertainty via self-consistency"""
        if self.self_consistency_k < 2:
            return 0.0

        try:
            responses = []
            messages = [{"role": "user", "content": query}]
            if context:
                context_str = "\n\n".join(context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

            for _ in range(self.self_consistency_k):
                result = self.sc_client.chat(messages=messages)
                if isinstance(result, tuple):
                    response = result[0][0] if isinstance(result[0], list) else result[0]
                else:
                    response = result[0] if isinstance(result, list) else result
                responses.append(response.strip().lower())

            unique = len(set(responses))
            uncertainty = unique / len(responses)
            return uncertainty
        except:
            return 0.5  # Default moderate uncertainty

    def run_experiment(self, test_query: TestQuery) -> ExperimentResult:
        """Run experiment on one query"""
        print(f"\nQuery: {test_query.query}")
        print(f"Type: {test_query.query_type}")
        print(f"Expected route: {test_query.expected_route}")
        print("-" * 80)

        # 1. Analyze query with complexity scorer
        print("\n📊 Complexity Analysis:")
        routing_metrics = self.analyze_query(test_query.query, test_query.context)

        print(f"  Erotetic Type: {routing_metrics.erotetic_type}")
        print(f"  Bloom Level: {routing_metrics.bloom_level}")
        print(f"  Reasoning Depth: {routing_metrics.reasoning_depth} steps")
        print(f"  Complexity Score: {routing_metrics.complexity_score:.3f}")
        print(f"  SC Uncertainty: {routing_metrics.self_consistency_uncertainty:.3f}")
        print(f"  → Recommended: {routing_metrics.recommended_route.upper()} (confidence: {routing_metrics.confidence:.2%})")

        # 2. Get local response
        print("\n--- Local Response ---")
        local_start = time.time()
        try:
            messages = [{"role": "user", "content": test_query.query}]
            if test_query.context:
                context_str = "\n\n".join(test_query.context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {test_query.query}"

            result = self.local_client.chat(messages=messages)
            if isinstance(result, tuple):
                local_answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                local_answer = result[0] if isinstance(result, list) else result

            local_time = time.time() - local_start
            print(f"✓ Local answer in {local_time:.2f}s")
            print(f"  {local_answer[:200]}...")
        except Exception as e:
            local_answer = f"Error: {e}"
            local_time = time.time() - local_start
            print(f"✗ Local error: {e}")

        # 3. Get full protocol response (for comparison)
        print("\n--- Full Minions Protocol ---")
        full_start = time.time()
        try:
            minions = Minions(
                local_client=self.local_client,
                remote_client=self.remote_client,
                mode="smart_routing"
            )
            # Use __call__ instead of run
            result = minions(
                task=test_query.query,
                doc_metadata="query",  # Required parameter
                context=test_query.context if test_query.context else [],
                max_rounds=3
            )
            # Extract final_answer from result dict
            full_answer = result.get("final_answer", "") if isinstance(result, dict) else str(result)
            full_time = time.time() - full_start
            full_error = None
            print(f"✓ Full protocol in {full_time:.2f}s")
            print(f"  {full_answer[:200]}...")
        except Exception as e:
            full_answer = f"Error: {e}"
            full_time = time.time() - full_start
            full_error = str(e)
            print(f"✗ Full protocol error: {e}")

        # 4. Judge evaluation
        print("\n--- Judge Evaluation ---")
        judge_result = self.judge.judge(
            test_query.query,
            local_answer,
            full_answer,
            test_query.ground_truth
        )

        choice = judge_result["choice"]
        if choice == 0:
            winner = "local"
        elif choice == 1:
            winner = "full"
        else:
            winner = "tie"

        print(f"Judge: {winner.upper()}")
        print(f"Reason: {judge_result['reason']}")

        # 5. Calculate metrics
        time_saved = full_time - local_time
        speedup = full_time / local_time if local_time > 0 else 0

        return ExperimentResult(
            query=test_query.query,
            query_type=test_query.query_type,
            routing_metrics=routing_metrics,
            local_answer=local_answer,
            local_time=local_time,
            local_used_route=(routing_metrics.recommended_route == "local"),
            full_answer=full_answer,
            full_time=full_time,
            full_error=full_error,
            judge_choice=choice,
            judge_reason=judge_result["reason"],
            winner=winner,
            ground_truth=test_query.ground_truth,
            time_saved=time_saved,
            speedup_factor=speedup
        )


def main():
    parser = argparse.ArgumentParser(description="Enhanced smart routing evaluation")
    parser.add_argument("--max-queries", type=int, default=5, help="Max queries to test")
    parser.add_argument("--output", type=str, default="enhanced_results.json", help="Output file")
    parser.add_argument("--query-types", type=str, nargs="+", default=None, help="Specific query types")

    args = parser.parse_args()

    print("="*80)
    print("ENHANCED ROUTING EVALUATION")
    print("Using: Complexity Scorer + Self-Consistency + Formal Reasoning")
    print("="*80)

    # Initialize clients
    print("\nInitializing clients...")
    from minions.clients import OllamaClient, TogetherClient

    local_client = OllamaClient(model_name="llama3.2", temperature=0.0, max_tokens=4096, use_async=False)
    remote_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
    judge_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")

    print("✓ Clients ready")

    # Load test data
    dataset = TestDataset()
    if args.query_types:
        queries = []
        for qt in args.query_types:
            queries.extend(dataset.get_by_type(qt))
    else:
        queries = dataset.get_all()

    queries = queries[:args.max_queries]
    print(f"✓ Loaded {len(queries)} queries\n")

    # Run experiments
    evaluator = EnhancedRoutingEvaluator(local_client, remote_client, judge_client)

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'='*80}")

        result = evaluator.run_experiment(query)
        results.append(result)

    # Analysis
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    local_wins = sum(1 for r in results if r.winner == "local")
    full_wins = sum(1 for r in results if r.winner == "full")
    ties = sum(1 for r in results if r.winner == "tie")

    print(f"\nTotal queries: {len(results)}")
    print(f"Local wins: {local_wins} ({local_wins/len(results)*100:.1f}%)")
    print(f"Full wins: {full_wins} ({full_wins/len(results)*100:.1f}%)")
    print(f"Ties: {ties} ({ties/len(results)*100:.1f}%)")

    avg_speedup = np.mean([r.speedup_factor for r in results if r.speedup_factor > 0])
    total_time_saved = sum(r.time_saved for r in results)

    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Total time saved: {total_time_saved:.2f}s")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "total_queries": len(results),
            "local_wins": local_wins,
            "full_wins": full_wins,
            "ties": ties,
            "avg_speedup": float(avg_speedup),
            "total_time_saved": float(total_time_saved)
        },
        "results": [asdict(r) for r in results]
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
