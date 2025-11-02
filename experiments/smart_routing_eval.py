"""
Smart Routing Evaluation: Local-Only vs Full Protocol

This script runs BOTH strategies on the same queries and uses an LLM judge
(inspired by CS329A homework) to evaluate which approach gives better answers.

Usage:
    python experiments/smart_routing_eval.py --max-queries 5
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


class SmartRoutingEvaluator:
    """
    Evaluates smart routing by comparing local-only vs full protocol on same queries.
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
        Initialize evaluator.

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
        Run smart routing evaluation on test queries.

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

            # === 2. Run FULL Minions Protocol ===
            if verbose:
                print(f"\n--- Method 2: Full Minions Protocol ---")

            full_answer = None
            full_time = 0

            try:
                start_time = time.time()

                # Run actual full Minions protocol with context
                minions_result = self.minions_full(
                    task=query.query,
                    doc_metadata=f"{query.query_type} query with {len(query.context)} context documents",
                    context=query.context  # Now has real context!
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

    def plot_results(self, results: List[ComparisonResult], output_prefix: str = "comparison"):
        """Generate visualization plots from comparison results"""

        if not results:
            print("No results to plot")
            return

        # Group by query type
        by_type = defaultdict(list)
        for r in results:
            by_type[r.query_type].append(r)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Time Comparison by Query Type
        ax1 = plt.subplot(2, 3, 1)
        query_types = list(by_type.keys())
        local_times = [np.mean([r.local_time for r in by_type[qt]]) for qt in query_types]
        full_times = [np.mean([r.full_time for r in by_type[qt] if r.full_time > 0]) for qt in query_types]

        x = np.arange(len(query_types))
        width = 0.35
        ax1.bar(x - width/2, local_times, width, label='Local-only', color='#2ecc71')
        ax1.bar(x + width/2, full_times, width, label='Full Protocol', color='#e74c3c')
        ax1.set_xlabel('Query Type')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Average Response Time by Query Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(query_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Speedup Factor by Query Type
        ax2 = plt.subplot(2, 3, 2)
        speedups = [np.mean([r.speedup_factor for r in by_type[qt] if r.speedup_factor > 0]) for qt in query_types]
        bars = ax2.bar(query_types, speedups, color='#3498db')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        ax2.set_xlabel('Query Type')
        ax2.set_ylabel('Speedup Factor (x)')
        ax2.set_title('Speedup: Local vs Full Protocol')
        ax2.set_xticklabels(query_types, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{speedup:.1f}x',
                        ha='center', va='bottom', fontsize=9)

        # 3. Win/Loss/Tie Distribution
        ax3 = plt.subplot(2, 3, 3)
        wins = [
            sum(1 for r in results if r.winner == "local"),
            sum(1 for r in results if r.winner == "full"),
            sum(1 for r in results if r.winner == "tie")
        ]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        labels = [f'Local Wins\n({wins[0]})', f'Full Wins\n({wins[1]})', f'Ties\n({wins[2]})']
        wedges, texts, autotexts = ax3.pie(wins, labels=labels, colors=colors,
                                            autopct='%1.0f%%', startangle=90)
        ax3.set_title('Judge Evaluation Results')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # 4. Time Savings by Query Type
        ax4 = plt.subplot(2, 3, 4)
        time_saved = [np.sum([r.time_saved for r in by_type[qt]]) for qt in query_types]
        bars = ax4.bar(query_types, time_saved, color='#9b59b6')
        ax4.set_xlabel('Query Type')
        ax4.set_ylabel('Time Saved (seconds)')
        ax4.set_title('Total Time Saved by Query Type')
        ax4.set_xticklabels(query_types, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar, saved in zip(bars, time_saved):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{saved:.1f}s',
                    ha='center', va='bottom', fontsize=9)

        # 5. Query Type Distribution
        ax5 = plt.subplot(2, 3, 5)
        type_counts = [len(by_type[qt]) for qt in query_types]
        ax5.bar(query_types, type_counts, color='#f39c12')
        ax5.set_xlabel('Query Type')
        ax5.set_ylabel('Number of Queries')
        ax5.set_title('Query Type Distribution')
        ax5.set_xticklabels(query_types, rotation=45, ha='right')
        ax5.grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, (qt, count) in enumerate(zip(query_types, type_counts)):
            ax5.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

        # 6. Cost Savings Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        total_time_saved = sum(r.time_saved for r in results)
        total_local_time = sum(r.local_time for r in results)
        total_full_time = sum(r.full_time for r in results if r.full_time > 0)
        avg_speedup = np.mean([r.speedup_factor for r in results if r.speedup_factor > 0])

        summary_text = f"""
        Performance Summary
        {'='*30}

        Total Queries: {len(results)}

        Time Metrics:
        • Local Total: {total_local_time:.1f}s
        • Full Total: {total_full_time:.1f}s
        • Time Saved: {total_time_saved:.1f}s

        Efficiency:
        • Avg Speedup: {avg_speedup:.1f}x
        • Time Reduction: {(total_time_saved/total_full_time*100) if total_full_time > 0 else 0:.1f}%

        Quality (Judge Eval):
        • Local Wins: {wins[0]} ({wins[0]/len(results)*100:.0f}%)
        • Full Wins: {wins[1]} ({wins[1]/len(results)*100:.0f}%)
        • Ties: {wins[2]} ({wins[2]/len(results)*100:.0f}%)
        """

        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save plot
        plot_file = f"{output_prefix}_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plots saved to: {plot_file}")

        # Also save individual plots for better resolution
        self._save_individual_plots(results, by_type, output_prefix)

        plt.close()

    def _save_individual_plots(self, results: List[ComparisonResult],
                                by_type: Dict, output_prefix: str):
        """Save high-resolution individual plots"""

        query_types = list(by_type.keys())

        # Individual plot 1: Time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        local_times = [np.mean([r.local_time for r in by_type[qt]]) for qt in query_types]
        full_times = [np.mean([r.full_time for r in by_type[qt] if r.full_time > 0]) for qt in query_types]

        x = np.arange(len(query_types))
        width = 0.35
        ax.bar(x - width/2, local_times, width, label='Local-only', color='#2ecc71')
        ax.bar(x + width/2, full_times, width, label='Full Protocol', color='#e74c3c')
        ax.set_xlabel('Query Type', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Response Time Comparison: Local vs Full Protocol', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_types, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_time_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Time comparison plot: {output_prefix}_time_comparison.png")

        # Individual plot 2: Speedup
        fig, ax = plt.subplots(figsize=(10, 6))
        speedups = [np.mean([r.speedup_factor for r in by_type[qt] if r.speedup_factor > 0]) for qt in query_types]
        bars = ax.bar(query_types, speedups, color='#3498db', edgecolor='black', linewidth=1.5)
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=2, label='No speedup (1x)')
        ax.set_xlabel('Query Type', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('Local-only Speedup vs Full Protocol', fontsize=14, fontweight='bold')
        ax.set_xticklabels(query_types, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{speedup:.1f}x',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_speedup.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Speedup plot: {output_prefix}_speedup.png")


def main():
    parser = argparse.ArgumentParser(description="Smart routing evaluation")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help="Maximum queries to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="smart_routing_eval_results.json",
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
        # Use Qwen 2.5 72B - large model with high token limits (32k context)
        # Better choice than 405B which has only 2048 token limit
        remote_client = TogetherClient(
            model_name="Qwen/Qwen2.5-72B-Instruct-Turbo",
            temperature=0.0,
            max_tokens=4096,  # Can use higher limits with Qwen
            api_key=together_key
        )
        print("✓ Remote client (Together AI - Qwen 2.5 72B) initialized")

        # Use same model for judge
        judge_client = TogetherClient(
            model_name="Qwen/Qwen2.5-72B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=2048,
            api_key=together_key
        )
        print("✓ Judge client (Together AI - Qwen 2.5 72B) initialized")

    except Exception as e:
        print(f"✗ Could not initialize Together AI client: {e}")
        return

    # ===== Load Dataset =====
    dataset = TestDataset()

    # Select test queries - ensure we get at least some queries
    test_queries = []
    if args.query_types:
        # Get queries from specified types
        for qtype in args.query_types:
            queries = dataset.get_by_type(qtype)
            # Ensure at least 1 query per type if max_queries allows
            per_type = max(1, args.max_queries // len(args.query_types))
            test_queries.extend(queries[:per_type])
    else:
        # Get all queries
        test_queries = dataset.get_all()

    # Limit to max_queries
    test_queries = test_queries[:args.max_queries]

    print(f"\n✓ Loaded {len(test_queries)} test queries")

    # ===== Run Comparison =====
    evaluator = SmartRoutingEvaluator(
        local_client=local_client,
        remote_client=remote_client,
        judge_client=judge_client,
        dataset=dataset
    )

    results = evaluator.run_comparison(test_queries, verbose=True)

    # ===== Analyze Results =====
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    analysis = evaluator.analyze_results(results)

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

    # ===== Generate Plots =====
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    output_prefix = args.output.replace('.json', '')
    evaluator.plot_results(results, output_prefix=output_prefix)

    print("\n" + "="*80)
    print("✓ All plots generated!")
    print("="*80)


if __name__ == "__main__":
    main()
