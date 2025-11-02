"""
Head-to-Head Comparison: Local-Only vs Full Protocol

This script runs BOTH strategies on the same queries and uses an LLM judge
(inspired by CS329A homework) to evaluate which approach gives better answers.

Usage:
    python experiments/head_to_head_comparison.py --max-queries 5
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
from minions.minions import Minions
from minions.utils.smart_router import SmartRouter


@dataclass
class JudgeConfig:
    temperature: float = 0.7
    max_choices: int = 2


class LLMJudge:
    """
    LLM-as-a-judge to compare local-only vs full protocol answers.
    Inspired by CS329A Homework 2 LLM judge implementation.
    """

    def __init__(self, judge_client, cfg: JudgeConfig = JudgeConfig()):
        """
        Initialize judge with a client.

        Args:
            judge_client: LLM client for judging (e.g., TogetherAIClient)
            cfg: Judge configuration
        """
        self.judge_client = judge_client
        self.cfg = cfg

    def judge(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two answers and select the better one.

        Args:
            query: The original query
            answer_a: Local-only answer (Candidate 0)
            answer_b: Full protocol answer (Candidate 1)
            ground_truth: Optional ground truth for context

        Returns:
            Dictionary with:
            - choice: 0 for answer_a, 1 for answer_b, None if can't decide
            - reason: Explanation for choice
            - raw_response: Full judge response
        """
        # Build judge prompt (similar to homework's _build_messages)
        prompt = self._build_prompt(query, answer_a, answer_b, ground_truth)

        # Get judge's response
        try:
            # Together client returns (response, usage) when local=False
            result = self.judge_client.chat([{
                "role": "user",
                "content": prompt
            }])

            # Handle both 2-value and 3-value returns
            if len(result) == 3:
                response, usage, _ = result
            else:
                response, usage = result

            raw_response = response[0]

            # Parse JSON response (similar to homework's _parse_json_choice)
            choice, reason = self._parse_json_choice(raw_response)

            return {
                "choice": choice,
                "reason": reason,
                "raw_response": raw_response,
                "usage": usage.to_dict() if hasattr(usage, 'to_dict') else {}
            }
        except Exception as e:
            print(f"Judge error: {e}")
            return {
                "choice": None,
                "reason": f"Error: {str(e)}",
                "raw_response": "",
                "usage": {}
            }

    def _build_prompt(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        ground_truth: Optional[str]
    ) -> str:
        """Build judge prompt (inspired by homework's _build_messages)"""

        gt_context = ""
        if ground_truth:
            gt_context = f"""
Ground Truth Reference (for context):
{ground_truth}
"""

        prompt = f"""You are an expert evaluator comparing two AI-generated answers.

Question:
{query}

{gt_context}

Candidate 0 (Local-only model):
{answer_a}

Candidate 1 (Full protocol):
{answer_b}

Instructions:
- Carefully analyze both answers for correctness, completeness, and accuracy
- Consider which answer better addresses the question
- If there's a ground truth, use it as a reference but don't require exact matches
- If both answers are equally good or both are wrong, you may return null

Response Format:
You MUST respond with a single-line JSON object and nothing else:
{{"choice": <0, 1, or null>, "reason": "<one short sentence explaining your choice>"}}

Examples:
{{"choice": 0, "reason": "More accurate and complete answer"}}
{{"choice": 1, "reason": "Better handles edge cases and provides correct analysis"}}
{{"choice": null, "reason": "Both answers are equally valid"}}

Your response:"""

        return prompt

    def _parse_json_choice(self, raw: str) -> Tuple[Optional[int], str]:
        """Parse judge's JSON response (similar to homework's method)"""
        import json
        import re

        if not raw or not raw.strip():
            return None, "Empty response"

        # Try to parse first line as JSON
        first = raw.strip().splitlines()[0].strip()
        obj = None

        try:
            obj = json.loads(first)
        except:
            # Try to find JSON object in response
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except:
                    obj = None

        if not isinstance(obj, dict):
            return None, "Unparseable response"

        choice = obj.get("choice", None)
        reason = obj.get("reason", "")

        if choice is None:
            return None, reason or "No choice made"

        try:
            idx = int(choice)
            return (idx if idx in [0, 1] else None), reason
        except:
            return None, reason or "Invalid choice format"


@dataclass
class ComparisonResult:
    """Results from comparing local-only vs full protocol"""
    query: str
    query_type: str

    # Local-only results
    local_answer: str
    local_time: float
    local_used: bool  # Did smart routing use local-only?

    # Full protocol results
    full_answer: str
    full_time: float

    # Judge evaluation
    judge_choice: Optional[int]  # 0=local, 1=full, None=tie
    judge_reason: str
    winner: str  # "local", "full", or "tie"

    # Ground truth
    ground_truth: Optional[str]

    # Performance
    time_saved: float
    speedup_factor: float


class HeadToHeadComparator:
    """
    Compares local-only vs full protocol head-to-head on same queries.
    Uses LLM judge to determine which answer is better.
    """

    def __init__(
        self,
        local_client,
        remote_client,
        judge_client,
        dataset: TestDataset
    ):
        """
        Initialize comparator.

        Args:
            local_client: Local LLM client (e.g., Ollama)
            remote_client: Remote LLM client (e.g., Together AI)
            judge_client: Judge LLM client (e.g., Together AI)
            dataset: Test dataset
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge = LLMJudge(judge_client)
        self.dataset = dataset

        # Initialize both strategies
        self.smart_router = SmartRouter()
        self.minions_full = Minions(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=3
        )

    def run_comparison(
        self,
        test_queries: List[TestQuery],
        verbose: bool = True
    ) -> List[ComparisonResult]:
        """
        Run head-to-head comparison on test queries.

        For each query:
        1. Get local-only answer (if smart router allows)
        2. Get full protocol answer
        3. Use LLM judge to compare
        4. Record results

        Args:
            test_queries: List of test queries
            verbose: Print progress

        Returns:
            List of comparison results
        """
        results = []

        for i, query in enumerate(test_queries, 1):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Query {i}/{len(test_queries)}")
                print(f"{'='*80}")
                print(f"Type: {query.query_type}")
                print(f"Query: {query.query}")
                print(f"Expected route: {query.expected_route}")

            # === 1. Try local-only approach ===
            if verbose:
                print(f"\n--- Method 1: Local-Only ---")

            local_answer = None
            local_time = 0
            local_used = False

            try:
                query_type = self.smart_router.classify_query_type(query.query)

                start_time = time.time()
                local_result = self.smart_router.get_local_only_response(
                    query.query,
                    self.local_client,
                    query_type
                )
                local_time = time.time() - start_time

                if local_result:
                    local_answer = local_result["final_answer"]
                    local_used = True
                    if verbose:
                        print(f"✓ Local-only succeeded in {local_time:.2f}s")
                        print(f"Answer: {local_answer[:150]}...")
                else:
                    # If smart routing says no, still get a local answer for comparison
                    if verbose:
                        print(f"✗ Smart router would escalate")
                        print(f"Getting local answer anyway for comparison...")

                    response, _, _ = self.local_client.chat([{
                        "role": "user",
                        "content": query.query
                    }])
                    local_answer = response[0]
                    local_time = time.time() - start_time

                    if verbose:
                        print(f"Local answer (not confident): {local_answer[:150]}...")

            except Exception as e:
                print(f"✗ Local error: {e}")
                local_answer = f"Error: {e}"

            # === 2. Run full protocol ===
            if verbose:
                print(f"\n--- Method 2: Full Protocol ---")

            full_answer = None
            full_time = 0

            try:
                start_time = time.time()

                # Run actual full Minions protocol
                minions_result = self.minions_full(
                    task=query.query,
                    doc_metadata="Comparison test query",
                    context=query.context if query.context else []
                )

                full_time = time.time() - start_time
                full_answer = minions_result.get("final_answer", "No answer generated")

                if verbose:
                    print(f"✓ Full protocol completed in {full_time:.2f}s")
                    print(f"Answer: {full_answer[:150]}...")

            except Exception as e:
                print(f"✗ Full protocol error: {e}")
                full_answer = f"Error: {e}"
                full_time = 0

            # === 3. Judge comparison ===
            if verbose:
                print(f"\n--- Judge Evaluation ---")

            judge_result = self.judge.judge(
                query=query.query,
                answer_a=local_answer,
                answer_b=full_answer,
                ground_truth=query.ground_truth
            )

            choice = judge_result["choice"]
            reason = judge_result["reason"]

            winner = "tie"
            if choice == 0:
                winner = "local"
            elif choice == 1:
                winner = "full"

            if verbose:
                print(f"Judge choice: {choice} ({winner})")
                print(f"Reason: {reason}")

            # === 4. Record results ===
            time_saved = full_time - local_time if local_time > 0 else 0
            speedup = full_time / local_time if local_time > 0 else 1

            result = ComparisonResult(
                query=query.query,
                query_type=query.query_type,
                local_answer=local_answer,
                local_time=local_time,
                local_used=local_used,
                full_answer=full_answer,
                full_time=full_time,
                judge_choice=choice,
                judge_reason=reason,
                winner=winner,
                ground_truth=query.ground_truth,
                time_saved=time_saved,
                speedup_factor=speedup
            )

            results.append(result)

        return results

    def analyze_results(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze and summarize comparison results"""

        total = len(results)
        local_wins = sum(1 for r in results if r.winner == "local")
        full_wins = sum(1 for r in results if r.winner == "full")
        ties = sum(1 for r in results if r.winner == "tie")

        total_local_time = sum(r.local_time for r in results)
        total_full_time = sum(r.full_time for r in results)
        total_time_saved = sum(r.time_saved for r in results)

        avg_speedup = sum(r.speedup_factor for r in results) / total if total > 0 else 1

        # Accuracy comparison
        local_accuracy = local_wins / (local_wins + full_wins) if (local_wins + full_wins) > 0 else 0
        full_accuracy = full_wins / (local_wins + full_wins) if (local_wins + full_wins) > 0 else 0

        return {
            "total_queries": total,
            "local_wins": local_wins,
            "full_wins": full_wins,
            "ties": ties,
            "local_win_rate": local_wins / total if total > 0 else 0,
            "full_win_rate": full_wins / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
            "total_local_time": total_local_time,
            "total_full_time": total_full_time,
            "total_time_saved": total_time_saved,
            "avg_speedup": avg_speedup,
            "efficiency_gain": (total_time_saved / total_full_time * 100) if total_full_time > 0 else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Head-to-head comparison")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help="Maximum queries to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="head_to_head_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--query-types",
        type=str,
        nargs="+",
        default=["lookup", "math", "multi-hop"],
        help="Query types to test"
    )

    args = parser.parse_args()

    # ===== Initialize Clients =====
    print("\n" + "="*80)
    print("INITIALIZING CLIENTS")
    print("="*80)

    try:
        # Local client (Ollama)
        from minions.clients import OllamaClient
        local_client = OllamaClient(
            model_name="llama3.2",
            temperature=0.0,
            max_tokens=4096,
            num_ctx=4096,
            use_async=True
        )
        print("✓ Local client (Ollama) initialized")
    except Exception as e:
        print(f"✗ Could not initialize local client: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return

    # Remote client (Together AI)
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("✗ TOGETHER_API_KEY not found in environment")
        print("  Set it with: export TOGETHER_API_KEY=your_key")
        return

    try:
        from minions.clients import TogetherClient
        remote_client = TogetherClient(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0.0,
            max_tokens=4096,
            api_key=together_key
        )
        print("✓ Remote client (Together AI) initialized")

        # Use same client for judge
        judge_client = TogetherClient(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=2048,
            api_key=together_key
        )
        print("✓ Judge client (Together AI) initialized")

    except Exception as e:
        print(f"✗ Could not initialize Together AI client: {e}")
        return

    # ===== Load Dataset =====
    dataset = TestDataset()

    # Select test queries
    test_queries = []
    for qtype in args.query_types:
        queries = dataset.get_by_type(qtype)
        test_queries.extend(queries[:args.max_queries // len(args.query_types)])

    test_queries = test_queries[:args.max_queries]

    print(f"\n✓ Loaded {len(test_queries)} test queries")

    # ===== Run Comparison =====
    comparator = HeadToHeadComparator(
        local_client=local_client,
        remote_client=remote_client,
        judge_client=judge_client,
        dataset=dataset
    )

    results = comparator.run_comparison(test_queries, verbose=True)

    # ===== Analyze Results =====
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    analysis = comparator.analyze_results(results)

    print(f"\nTotal queries: {analysis['total_queries']}")
    print(f"\nWins:")
    print(f"  Local-only: {analysis['local_wins']} ({analysis['local_win_rate']:.1%})")
    print(f"  Full protocol: {analysis['full_wins']} ({analysis['full_win_rate']:.1%})")
    print(f"  Ties: {analysis['ties']} ({analysis['tie_rate']:.1%})")

    print(f"\nPerformance:")
    print(f"  Total local time: {analysis['total_local_time']:.2f}s")
    print(f"  Total full time: {analysis['total_full_time']:.2f}s")
    print(f"  Time saved: {analysis['total_time_saved']:.2f}s")
    print(f"  Average speedup: {analysis['avg_speedup']:.1f}x")
    print(f"  Efficiency gain: {analysis['efficiency_gain']:.1f}%")

    # ===== Save Results =====
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "results": [asdict(r) for r in results]
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
