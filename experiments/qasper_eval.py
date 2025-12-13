"""
QASPER Evaluation Script

This script evaluates three approaches on the QASPER dataset:
1. Local Client (Gemma 3 12B via Ollama)
2. Minions Protocol (Local + Remote) - Placeholder
3. Remote Client (Qwen 2.5 72B via Together)

It uses an LLM Judge to compare the quality of answers.
"""

import argparse
import json
import time
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.minions import Minions
from minions.clients import OllamaClient, TogetherClient
from minions.qasper_dataset import QasperDatasetLoader, create_prompts_for_example
from experiments.enhanced_routing_eval import LLMJudge  # Reusing LLMJudge
from experiments.financebench_eval import GeminiJudgeClient

@dataclass
class EvalResult:
    """Result for one query evaluation"""
    query: str
    paper_title: str
    ground_truth: str
    paper_id: str
    
    # Answers
    local_answer: str
    local_time: float
    
    minions_answer: str
    minions_time: float
    
    remote_answer: str
    remote_time: float
    
    # Judgments
    minions_vs_local_choice: Optional[int]
    minions_vs_local_reason: str
    minions_vs_local_winner: str
    
    minions_vs_remote_choice: Optional[int]
    minions_vs_remote_reason: str
    minions_vs_remote_winner: str

    local_vs_remote_choice: Optional[int]
    local_vs_remote_reason: str
    local_vs_remote_winner: str

    # Accuracy (Exact match or simple string check for Unanswerable)
    # Note: Judge-based correctness check can be added if needed, sticking to comparative for now.


class QasperEvaluator:
    def __init__(self, local_client, remote_client, judge_client, output_file: str):
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge = LLMJudge(judge_client)
        self.output_file = output_file

    def evaluate_query(self, example) -> EvalResult:
        """Evaluate a single Qasper example"""
        query = example.question
        ground_truth = example.ground_truth
        
        # Prepare content/prompt
        prompt_content = create_prompts_for_example(example)
        
        print(f"\nQuery: {query}")
        print(f"Ground Truth: {ground_truth[:100]}...")

        # 1. Local Response
        print("\n--- Local Response ---")
        local_start = time.time()
        try:
            # Gemma/Local prompt structure
            messages = [{"role": "user", "content": prompt_content}]
            result = self.local_client.chat(messages=messages)
            if isinstance(result, tuple):
                local_answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                local_answer = result[0] if isinstance(result, list) else result
            print(f"✓ Local answer received: {local_answer[:100]}...")
        except Exception as e:
            local_answer = f"Error: {str(e)}"
            print(f"✗ Local error: {e}")
        local_time = time.time() - local_start

        # 2. Minions Protocol (Placeholder)
        print("\n--- Minions Protocol ---")
        minions_start = time.time()
        try:
            minions = Minions(local_client=self.local_client, remote_client=self.remote_client, max_rounds=3)
            result = minions(task=query, doc_metadata=f"Paper: {example.title}", context=[example.context])
            minions_answer = result.get("final_answer", "") if isinstance(result, dict) else str(result)
            print(f"✓ Minions answer received")
        except Exception as e:
            minions_answer = f"Error: {str(e)}"
            print(f"✗ Minions error: {e}")
        minions_time = time.time() - minions_start

        # 3. Remote Response (Baseline)
        print("\n--- Remote Response ---")
        remote_start = time.time()
        try:
            messages = [{"role": "user", "content": prompt_content}]
            result = self.remote_client.chat(messages=messages)
            if isinstance(result, tuple):
                remote_answer = result[0][0]
            else:
                remote_answer = str(result)
            print(f"✓ Remote answer received: {remote_answer[:100]}...")
        except Exception as e:
            remote_answer = f"Error: {str(e)}"
            print(f"✗ Remote error: {e}")
        remote_time = time.time() - remote_start
        
        # 4. Judging
        print("\n--- Judging ---")
        
        # Comparison Logic
        def get_winner(choice):
            if choice == 0: return "minions" # or Candidate A
            if choice == 1: return "other"   # or Candidate B
            return "tie"

        # Minions (A) vs Local (B)
        judgment_local = self.judge.judge(
            query=query, 
            answer_a=minions_answer, 
            answer_b=local_answer, # Candidate 1
            ground_truth=ground_truth
        )
        if judgment_local['choice'] == 0: m_vs_l = 'minions'
        elif judgment_local['choice'] == 1: m_vs_l = 'local'
        else: m_vs_l = 'tie'
            
        # Minions (A) vs Remote (B)
        judgment_remote = self.judge.judge(
            query=query, 
            answer_a=minions_answer, 
            answer_b=remote_answer, 
            ground_truth=ground_truth
        )
        if judgment_remote['choice'] == 0: m_vs_r = 'minions'
        elif judgment_remote['choice'] == 1: m_vs_r = 'remote'
        else: m_vs_r = 'tie'

        # Local (A) vs Remote (B)
        judgment_l_vs_r = self.judge.judge(
            query=query, 
            answer_a=local_answer, 
            answer_b=remote_answer, 
            ground_truth=ground_truth
        )
        if judgment_l_vs_r['choice'] == 0: l_vs_r = 'local'
        elif judgment_l_vs_r['choice'] == 1: l_vs_r = 'remote'
        else: l_vs_r = 'tie'
            
        print(f"Minions vs Local: {m_vs_l.upper()}")
        print(f"Minions vs Remote: {m_vs_r.upper()}")
        print(f"Local vs Remote: {l_vs_r.upper()}")

        return EvalResult(
            query=query,
            paper_title=example.title,
            ground_truth=ground_truth,
            paper_id=example.metadata.get('paper_id', ''),
            local_answer=local_answer,
            local_time=local_time,
            minions_answer=minions_answer,
            minions_time=minions_time,
            remote_answer=remote_answer,
            remote_time=remote_time,
            minions_vs_local_choice=judgment_local['choice'],
            minions_vs_local_reason=judgment_local['reason'],
            minions_vs_local_winner=m_vs_l,
            minions_vs_remote_choice=judgment_remote['choice'],
            minions_vs_remote_reason=judgment_remote['reason'],
            minions_vs_remote_winner=m_vs_r,
            local_vs_remote_choice=judgment_l_vs_r['choice'],
            local_vs_remote_reason=judgment_l_vs_r['reason'],
            local_vs_remote_winner=l_vs_r
        )


def main():
    parser = argparse.ArgumentParser(description="QASPER Evaluation")
    parser.add_argument("--max-queries", type=int, default=5, help="Max queries to evaluate")
    parser.add_argument("--output", type=str, default="experiments/qasper_results.json", help="Output JSON file")
    args = parser.parse_args()

    print("="*80)
    print("QASPER EVALUATION")
    print("="*80)

    # Load Dataset
    print("Loading QASPER dataset...")
    try:
        loader = QasperDatasetLoader(split="test")
        dataset = loader.load()
        print(f"✓ Loaded {len(dataset)} examples")
    except ImportError:
        print("✗ Error: 'datasets' library not found. Please pip install datasets.")
        return
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Initialize Clients
    print("\nInitializing clients...")
    
    # Local: Gemma 3 12B (configured as 4b in FinanceBench example, adjusting to match or standard?)
    # Using defaults from FinanceBench script for consistency: gemma3:4b
    local_client = OllamaClient(
        model_name="gemma3:4b",
        temperature=0.0,
        max_tokens=4096,
        num_ctx=16000, # Increased context for Papers
        use_async=False,
    )
    
    # Remote: Qwen 2.5 72B
    api_key = os.getenv("TOGETHER_API_KEY")
    
    remote_client = TogetherClient(
        model_name="Qwen/Qwen2.5-72B-Instruct-Turbo", 
        api_key=api_key,
        temperature=0.0
    )

    judge_client = GeminiJudgeClient(
        model_name="gemini-2.5-flash", 
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    print("✓ Clients initialized")

    # Run Evaluation
    evaluator = QasperEvaluator(local_client, remote_client, judge_client, args.output)
    
    results = []
    # Take subset
    indices = range(min(args.max_queries, len(dataset)))
    
    for i in indices:
        print(f"\nProcessing query {i+1}/{len(indices)}")
        print("-" * 40)
        try:
            res = evaluator.evaluate_query(dataset[i])
            results.append(asdict(res))
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            import traceback
            traceback.print_exc()

    # Save Results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "local_model": "gemma3:4b",
            "remote_model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "judge_model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "max_queries": args.max_queries
        },
        "summary": {
            "total_queries": len(results),
            "minions_beats_local": sum(1 for r in results if r['minions_vs_local_winner'] == 'minions'),
            "minions_beats_remote": sum(1 for r in results if r['minions_vs_remote_winner'] == 'minions'),
            "local_beats_remote": sum(1 for r in results if r['local_vs_remote_winner'] == 'local'),
            "ties_local_vs_remote": sum(1 for r in results if r['local_vs_remote_winner'] == 'tie'),
        },
        "results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n\nResults saved to {args.output}")
    print("Summary:")
    print(f"Local vs Remote Wins: {output_data['summary']['local_beats_remote']}/{len(results)}")
    print(f"Ties: {output_data['summary']['ties_local_vs_remote']}/{len(results)}")

if __name__ == "__main__":
    main()
