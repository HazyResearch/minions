"""
FinanceBench Evaluation Script

This script evaluates three approaches on the FinanceBench dataset:
1. Local Client (Gemma 3 12B via Ollama)
2. Minions Protocol (Local + Remote)
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
# import numpy as np # Unused
# import PyPDF2 # Unused

# Add parent directory to path to import minions modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.minions import Minions
from minions.clients import OllamaClient, TogetherClient
from experiments.enhanced_routing_eval import LLMJudge  # Reusing LLMJudge from existing eval script
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Warning: google-genai not installed. Gemini judge will fail.")



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

    # Accuracy
    local_correct: bool
    minions_correct: bool
    remote_correct: bool


class GeminiJudgeClient:
    """Wrapper for Google Gemini API to match LLMJudge interface"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, str]]) -> Tuple[List[str], Dict]:
        """
        Chat compatible with LLMJudge.
        Expects messages=[{"role": "user", "content": prompt}]
        Returns ([response_text], usage_dict)
        """
        # Extract the prompt from the last user message
        prompt = messages[-1]["content"]
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            # Return tuple (choices, usage)
            return [response.text], {"total_tokens": 0} 
        except Exception as e:
            print(f"Gemini API Error: {e}")
            raise e


class FinanceBenchEvaluator:
    def __init__(self, local_client, remote_client, judge_client, output_file: str):
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge = LLMJudge(judge_client)
        self.output_file = output_file
        self.judge_client = judge_client # Store for correctness check

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
            import re
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            return data.get("correct", False)
        except Exception as e:
            print(f"Correctness check failed: {e}")
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

        # if os.path.exists(f"/Users/eric/Documents/cs329a/financebench/pdfs/{doc_name}.txt"):
        #     with open(f"/Users/eric/Documents/cs329a/financebench/pdfs/{doc_name}.txt", 'r') as f:
        #         content = f.read()
        # else:
        #     content = evidence
        content = evidence

        context = [f"Document: {doc_name} ({doc_period})\nEvidence: {content}"]

        print(f"\nQuery: {query}")
        print(f"Ground Truth: {ground_truth[:100]}...")

        # 1. Local Response
        print("\n--- Local Response ---")
        local_start = time.time()
        try:
            messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
            result = self.local_client.chat(messages=messages)
            if isinstance(result, tuple):
                local_answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                local_answer = result[0] if isinstance(result, list) else result
            print(f"✓ Local answer received: {local_answer}")
        except Exception as e:
            local_answer = f"Error: {str(e)}"
            print(f"✗ Local error: {e}")
        local_time = time.time() - local_start

        # 2. Minions Protocol
        print("\n--- Minions Protocol ---")
        minions_start = time.time()
        try:
            minions = Minions(
                local_client=self.local_client,
                remote_client=self.remote_client,
                max_rounds=3
            )
            result = minions(
                task=query,
                doc_metadata=f"Finance document {doc_name}",
                context=context
            )
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
            messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
            # Direct call to remote client
            # TogetherClient chat usually returns ([responses], usage)
            # We access the configured client directly
            result = self.remote_client.chat(messages=messages)
             # Together/OpenAI style return
            if isinstance(result, tuple):
                 # index 0 is list of choices
                remote_answer = result[0][0]
            else:
                remote_answer = str(result)
            print(f"✓ Remote answer received: {remote_answer}")
        except Exception as e:
            remote_answer = f"Error: {str(e)}"
            print(f"✗ Remote error: {e}")
        remote_time = time.time() - remote_start
        
        # 4. Judging
        print("\n--- Judging ---")
        
        # Win for Minions vs Local?
        # Candidate 0 = Minions, Candidate 1 = Local
        # Note: Previous judge evaluated Candidate 0 vs 1. Let's keep consistent.
        # We want to see if Minions is better.
        
        # Minions vs Local
        judgment_local = self.judge.judge(
            query=query,
            answer_a=minions_answer, # Candidate 0
            answer_b=local_answer,   # Candidate 1
            ground_truth=ground_truth
        )
        # If 0 wins, Minions wins. If 1 wins, Local wins.
        if judgment_local['choice'] == 0:
            m_vs_l_winner = "minions"
        elif judgment_local['choice'] == 1:
            m_vs_l_winner = "local"
        else:
            m_vs_l_winner = "tie"
            
        # Minions vs Remote
        judgment_remote = self.judge.judge(
            query=query,
            answer_a=minions_answer, # Candidate 0
            answer_b=remote_answer,  # Candidate 1
            ground_truth=ground_truth
        )
        if judgment_remote['choice'] == 0:
            m_vs_r_winner = "minions"
        elif judgment_remote['choice'] == 1:
            m_vs_r_winner = "remote"
        else:
            m_vs_r_winner = "tie"

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
            
        print(f"Minions vs Local: {m_vs_l_winner.upper()}")
        print(f"Minions vs Remote: {m_vs_r_winner.upper()}")
        print(f"Local vs Remote: {l_vs_r_winner.upper()}")

        # 5. Check Accuracy
        print("\n--- Accuracy Check ---")
        local_correct = self.check_correctness(query, local_answer, ground_truth)
        minions_correct = self.check_correctness(query, minions_answer, ground_truth)
        remote_correct = self.check_correctness(query, remote_answer, ground_truth)
        
        print(f"Local Correct: {local_correct}")
        print(f"Minions Correct: {minions_correct}")
        print(f"Remote Correct: {remote_correct}")

        return EvalResult(
            query=query,
            doc_name=doc_name,
            doc_link=doc_link,
            doc_period=doc_period,
            ground_truth=ground_truth,
            local_answer=local_answer,
            local_time=local_time,
            minions_answer=minions_answer,
            minions_time=minions_time,
            remote_answer=remote_answer,
            remote_time=remote_time,
            minions_vs_local_choice=judgment_local['choice'],
            minions_vs_local_reason=judgment_local['reason'],
            minions_vs_local_winner=m_vs_l_winner,
            minions_vs_remote_choice=judgment_remote['choice'],
            minions_vs_remote_reason=judgment_remote['reason'],
            minions_vs_remote_winner=m_vs_r_winner,
            local_vs_remote_choice=judgment_l_vs_r['choice'],
            local_vs_remote_reason=judgment_l_vs_r['reason'],
            local_vs_remote_winner=l_vs_r_winner,
            local_correct=local_correct,
            minions_correct=minions_correct,
            remote_correct=remote_correct
        )


def main():
    parser = argparse.ArgumentParser(description="FinanceBench Evaluation")
    parser.add_argument("--max-queries", type=int, default=5, help="Max queries to evaluate")
    parser.add_argument("--output", type=str, default="experiments/financebench_results.json", help="Output JSON file")
    args = parser.parse_args()

    print("="*80)
    print("FINANCEBENCH EVALUATION")
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
    
    # Local: Gemma 3 4B
    local_client = OllamaClient(
        model_name="gemma3:4b",
        temperature=0.0,
        max_tokens=4096,
        num_ctx=128000,
        use_async=False,
    )
    
    # Remote: Qwen 2.5 72B
    # Note: Using the same API key as seen in existing codebase if environment var not set
    # The existing code had a hardcoded key in experiments/enhanced_routing_eval.py, 
    # capturing that context but preferring env var if available.
    # We will use the class default or env var mostly, but for safety can check if we should pass one.
    # In `experiments/enhanced_routing_eval.py` user used a specific key. I will reuse it if needed if env is missing.
    
    api_key = os.getenv("TOGETHER_API_KEY")
    
    remote_client = TogetherClient(
        model_name="Qwen/Qwen2.5-72B-Instruct-Turbo", 
        api_key=api_key,
        temperature=0.0
    )
    
    # Judge: Gemini 2.5 Flash
    judge_client = GeminiJudgeClient(
        model_name="gemini-2.5-flash", 
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    print("✓ Clients initialized")

    # Run Evaluation
    evaluator = FinanceBenchEvaluator(local_client, remote_client, judge_client, args.output)
    
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
            "local_wins": sum(1 for r in results if r['minions_vs_local_winner'] == 'local'),
            "remote_wins": sum(1 for r in results if r['minions_vs_remote_winner'] == 'remote'),
            "remote_beats_local": sum(1 for r in results if r['local_vs_remote_winner'] == 'remote'),
            "ties_local": sum(1 for r in results if r['minions_vs_local_winner'] == 'tie'),
            "ties_remote": sum(1 for r in results if r['minions_vs_remote_winner'] == 'tie'),
            "ties_local_vs_remote": sum(1 for r in results if r['local_vs_remote_winner'] == 'tie'),
            "local_accuracy": sum(1 for r in results if r['local_correct']) / len(results) if results else 0,
            "minions_accuracy": sum(1 for r in results if r['minions_correct']) / len(results) if results else 0,
            "remote_accuracy": sum(1 for r in results if r['remote_correct']) / len(results) if results else 0,
        },
        "results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n\nResults saved to {args.output}")
    print("Summary:")
    print(f"Minions vs Local Wins: {output_data['summary']['minions_beats_local']}/{len(results)}")
    print(f"Minions vs Remote Wins: {output_data['summary']['minions_beats_remote']}/{len(results)}")
    print(f"Local vs Remote Wins: {output_data['summary']['local_beats_remote']}/{len(results)}")
    print(f"Local Accuracy: {output_data['summary']['local_accuracy']:.2%}")
    print(f"Minions Accuracy: {output_data['summary']['minions_accuracy']:.2%}")
    print(f"Remote Accuracy: {output_data['summary']['remote_accuracy']:.2%}")

if __name__ == "__main__":
    main()
