"""
FinanceBench Evaluation Script with Enhanced Routing

This script evaluates three approaches on the FinanceBench dataset:
1. Local Client (Gemma 3 12B via Ollama)
2. Enhanced Routing (Dynamic selection between Local and Remote based on complexity)
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

# Add parent directory to path to import minions modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.minions import Minions
from minions.clients import OllamaClient, TogetherClient
from experiments.enhanced_routing_eval import LLMJudge, EnhancedRoutingEvaluator, RoutingMetrics
from experiments.financebench_eval import GeminiJudgeClient
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

@dataclass
class EvalResult:
    """Result for one query evaluation"""
    query: str
    doc_name: str
    doc_link: str
    doc_period: str
    ground_truth: str
    
    # Answers
    local_answer: str
    local_time: float
    
    remote_answer: str
    remote_time: float
    
    enhanced_answer: str
    enhanced_time: float
    enhanced_route: str
    enhanced_confidence: float
    complexity_score: float
    
    # Judgments
    enhanced_vs_local_choice: Optional[int]
    enhanced_vs_local_reason: str
    enhanced_vs_local_winner: str
    
    enhanced_vs_remote_choice: Optional[int]
    enhanced_vs_remote_reason: str
    enhanced_vs_remote_winner: str

    local_vs_remote_choice: Optional[int]
    local_vs_remote_reason: str
    local_vs_remote_winner: str

    # Accuracy
    local_correct: bool
    enhanced_correct: bool
    remote_correct: bool


class FinanceBenchEvaluatorEnhanced:
    def __init__(self, local_client, remote_client, judge_client, output_file: str):
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge = LLMJudge(judge_client)
        self.routing_evaluator = EnhancedRoutingEvaluator(local_client, remote_client, judge_client)
        self.output_file = output_file
        self.judge_client = judge_client

    def check_correctness(self, query: str, answer: str, ground_truth: str) -> bool:
        """Check if an answer is correct based on ground truth"""
        prompt = f"""You are an expert evaluator. 
Question: {query}
Ground Truth: {ground_truth}
Candidate Answer: {answer}

Is the Candidate Answer factually correct and consistent with the Ground Truth? 
Ignore minor phrasing differences. Focus on key facts/numbers.
Response Format (JSON only): {{"correct": <true/false>, "reason": "explanation"}}
"""
        try:
            result = self.judge_client.chat([{"role": "user", "content": prompt}])
            if isinstance(result, tuple):
                 response = result[0][0]
            else:
                 response = str(result)
            
            # Simple parsing
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            return data.get("correct", False)
        except Exception:
            return False

    def evaluate_query(self, item: Dict[str, Any]) -> EvalResult:
        """Evaluate a single FinanceBench item"""
        query = item['question']
        ground_truth = item['answer']
        
        # Context info
        doc_name = item.get('doc_name', 'Unknown')
        doc_link = item.get('doc_link', 'Unknown')
        doc_period = item.get('doc_period', 'Unknown')

        if doc_period == 'Unknown':
            doc_period = 'Unknown'
        evidence = item.get('evidence', '')
        content = evidence

        context = [f"Document: {doc_name} ({doc_period})\nEvidence: {content}"]
        context_str = context[0]

        print(f"\nQuery: {query}")
        print(f"Ground Truth: {ground_truth[:100]}...")

        # 1. Routing Decision AND Full Execution
        # We run BOTH Local and Remote to evaluate the router's virtual performance
        # In a real system, we would only run the chosen one. Here we want to compare.
        
        print("\n--- Routing Analysis ---")
        routing_metrics = self.routing_evaluator.analyze_query(query, context)
        print(f"Complexity: {routing_metrics.complexity_score:.2f}")
        print(f"Recommended Route: {routing_metrics.recommended_route.upper()} (Confidence: {routing_metrics.confidence:.2f})")

        # 2. Local Response
        print("\n--- Local Response ---")
        local_start = time.time()
        try:
            messages = [{"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}]
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

        # 3. Remote Response (Baseline)
        print("\n--- Remote Response ---")
        remote_start = time.time()
        try:
            messages = [{"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}]
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
        
        # 4. Construct Enhanced Result
        if routing_metrics.recommended_route == "local":
            enhanced_answer = local_answer
            # Theoretically: Decision Time + Local Time
            enhanced_time = routing_metrics.decision_time + local_time
        else:
            enhanced_answer = remote_answer
            # Theoretically: Decision Time + Remote Time
            enhanced_time = routing_metrics.decision_time + remote_time

        # 5. Judging
        print("\n--- Judging ---")
        
        # Enhanced vs Local
        judgment_local = self.judge.judge(
            query=query,
            answer_a=enhanced_answer, # Candidate 0
            answer_b=local_answer,    # Candidate 1
            ground_truth=ground_truth
        )
        if judgment_local['choice'] == 0:
            e_vs_l_winner = "enhanced"
        elif judgment_local['choice'] == 1:
            e_vs_l_winner = "local"
        else:
            e_vs_l_winner = "tie"
            
        # Enhanced vs Remote
        judgment_remote = self.judge.judge(
            query=query,
            answer_a=enhanced_answer, # Candidate 0
            answer_b=remote_answer,   # Candidate 1
            ground_truth=ground_truth
        )
        if judgment_remote['choice'] == 0:
            e_vs_r_winner = "enhanced"
        elif judgment_remote['choice'] == 1:
            e_vs_r_winner = "remote"
        else:
            e_vs_r_winner = "tie"

        # Local vs Remote
        judgment_l_vs_r = self.judge.judge(
            query=query,
            answer_a=local_answer, # Candidate 0
            answer_b=remote_answer,  # Candidate 1
            ground_truth=ground_truth
        )
        if judgment_l_vs_r['choice'] == 0:
            l_vs_r_winner = "local"
        elif judgment_l_vs_r['choice'] == 1:
            l_vs_r_winner = "remote"
        else:
            l_vs_r_winner = "tie"
            
        print(f"Enhanced vs Local: {e_vs_l_winner.upper()}")
        print(f"Enhanced vs Remote: {e_vs_r_winner.upper()}")
        print(f"Local vs Remote: {l_vs_r_winner.upper()}")

        # 6. Check Accuracy
        print("\n--- Accuracy Check ---")
        local_correct = self.check_correctness(query, local_answer, ground_truth)
        remote_correct = self.check_correctness(query, remote_answer, ground_truth)
        
        if routing_metrics.recommended_route == "local":
            enhanced_correct = local_correct
        else:
            enhanced_correct = remote_correct
        
        print(f"Local Correct: {local_correct}")
        print(f"Enhanced Correct: {enhanced_correct}")
        print(f"Remote Correct: {remote_correct}")

        return EvalResult(
            query=query,
            doc_name=doc_name,
            doc_link=doc_link,
            doc_period=doc_period,
            ground_truth=ground_truth,
            local_answer=local_answer,
            local_time=local_time,
            remote_answer=remote_answer,
            remote_time=remote_time,
            enhanced_answer=enhanced_answer,
            enhanced_time=enhanced_time,
            enhanced_route=routing_metrics.recommended_route,
            enhanced_confidence=routing_metrics.confidence,
            complexity_score=routing_metrics.complexity_score,
            enhanced_vs_local_choice=judgment_local['choice'],
            enhanced_vs_local_reason=judgment_local['reason'],
            enhanced_vs_local_winner=e_vs_l_winner,
            enhanced_vs_remote_choice=judgment_remote['choice'],
            enhanced_vs_remote_reason=judgment_remote['reason'],
            enhanced_vs_remote_winner=e_vs_r_winner,
            local_vs_remote_choice=judgment_l_vs_r['choice'],
            local_vs_remote_reason=judgment_l_vs_r['reason'],
            local_vs_remote_winner=l_vs_r_winner,
            local_correct=local_correct,
            enhanced_correct=enhanced_correct,
            remote_correct=remote_correct
        )


def main():
    parser = argparse.ArgumentParser(description="FinanceBench Evaluation with Enhanced Routing")
    parser.add_argument("--max-queries", type=int, default=5, help="Max queries to evaluate")
    parser.add_argument("--output", type=str, default="experiments/financebench_results_enhanced.json", help="Output JSON file")
    args = parser.parse_args()

    print("="*80)
    print("FINANCEBENCH EVALUATION - ENHANCED ROUTING")
    print("="*80)

    # Load Dataset
    print("Loading FinanceBench dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("PatronusAI/financebench", split="train")
        print(f"✓ Loaded {len(dataset)} examples")
    except ImportError:
        print("✗ Error: 'datasets' library not found. Please pip install datasets.")
        return
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Initialize Clients
    print("\nInitializing clients...")
    
    local_client = OllamaClient(
        model_name="gemma3:4b",
        temperature=0.0,
        max_tokens=4096,
        num_ctx=128000,
        use_async=False,
    )
    
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
    evaluator = FinanceBenchEvaluatorEnhanced(local_client, remote_client, judge_client, args.output)
    
    results = []
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
            "max_queries": args.max_queries,
            "mode": "enhanced_routing"
        },
        "summary": {
            "total_queries": len(results),
            "enhanced_beats_local": sum(1 for r in results if r['enhanced_vs_local_winner'] == 'enhanced'),
            "enhanced_beats_remote": sum(1 for r in results if r['enhanced_vs_remote_winner'] == 'enhanced'),
            "local_beats_remote": sum(1 for r in results if r['local_vs_remote_winner'] == 'local'),
            "local_wins": sum(1 for r in results if r['enhanced_vs_local_winner'] == 'local'),
            "remote_wins": sum(1 for r in results if r['enhanced_vs_remote_winner'] == 'remote'),
            "remote_beats_local": sum(1 for r in results if r['local_vs_remote_winner'] == 'remote'),
            "local_accuracy": sum(1 for r in results if r['local_correct']) / len(results) if results else 0,
            "enhanced_accuracy": sum(1 for r in results if r['enhanced_correct']) / len(results) if results else 0,
            "remote_accuracy": sum(1 for r in results if r['remote_correct']) / len(results) if results else 0,
            "routing_stats": {
                "routed_local": sum(1 for r in results if r['enhanced_route'] == 'local'),
                "routed_remote": sum(1 for r in results if r['enhanced_route'] == 'remote'),
                "avg_complexity": sum(r['complexity_score'] for r in results) / len(results) if results else 0
            }
        },
        "results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n\nResults saved to {args.output}")
    print("Summary:")
    print(f"Enhanced vs Local Wins: {output_data['summary']['enhanced_beats_local']}/{len(results)}")
    print(f"Enhanced vs Remote Wins: {output_data['summary']['enhanced_beats_remote']}/{len(results)}")
    print(f"Enhanced Accuracy: {output_data['summary']['enhanced_accuracy']:.2%}")
    print(f"Routing: {output_data['summary']['routing_stats']['routed_local']} Local, {output_data['summary']['routing_stats']['routed_remote']} Remote")

if __name__ == "__main__":
    main()
