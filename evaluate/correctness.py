"""
Remote Verdict Correctness Evaluation Module

A simple evaluation module that uses a remote LLM to determine if a predicted
answer is correct given the ground truth and original question.

Uses 10% tolerance for numerical answers.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

from minions.usage import Usage

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of correctness evaluation."""
    is_correct: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    usage: Usage = field(default_factory=Usage)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "usage": self.usage.to_dict()
        }


# =============================================================================
# Remote Verdict Evaluator
# =============================================================================

VERDICT_PROMPT = """You are evaluating whether a predicted answer to a financial question is correct.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

PREDICTED ANSWER:
{predicted}

EVALUATION RULES:
1. For numerical answers: Allow 10% tolerance (e.g., if ground truth is 100, accept 90-110)
2. For yes/no questions: The stance must match exactly
3. For qualitative answers: The key facts and conclusions must align
4. Ignore minor formatting differences (e.g., "$1.5B" vs "1.5 billion dollars")
5. If the predicted answer contains the correct information along with additional context, it's still correct

Respond with a JSON object:
{{
    "is_correct": <true or false>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation of your verdict>"
}}"""


class RemoteVerdictEvaluator:
    """
    Simple correctness evaluator using remote LLM verdict.
    
    Sends the question, ground truth, and predicted answer to a remote model
    and asks it to determine correctness with 10% numerical tolerance.
    """
    
    def __init__(self, remote_client, numerical_tolerance: float = 0.10):
        """
        Initialize the remote verdict evaluator.
        
        Args:
            remote_client: Client for the remote model (e.g., OpenAIClient)
            numerical_tolerance: Tolerance for numerical comparison (default 10%)
        """
        self.remote_client = remote_client
        self.numerical_tolerance = numerical_tolerance
        self.total_usage = Usage()
    
    def evaluate(
        self,
        predicted: str,
        ground_truth: Union[str, List[str]],
        question: str
    ) -> EvaluationResult:
        """
        Evaluate if predicted answer is correct using remote LLM.
        
        Args:
            predicted: The predicted answer
            ground_truth: Ground truth answer(s)
            question: The original question
        
        Returns:
            EvaluationResult with verdict, confidence, and reasoning
        """
        # Handle empty prediction
        predicted = str(predicted) if predicted is not None else ""
        if not predicted or not predicted.strip():
            return EvaluationResult(
                is_correct=False,
                confidence=1.0,
                reasoning="Empty prediction"
            )
        
        # Normalize ground truth to string (use first if list)
        if isinstance(ground_truth, list):
            gt_str = ground_truth[0] if ground_truth else ""
        else:
            gt_str = str(ground_truth)
        
        # Build prompt
        prompt = VERDICT_PROMPT.format(
            question=question,
            ground_truth=gt_str,
            predicted=predicted.strip()
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            result = self.remote_client.chat(messages)
            # Handle both 2-tuple (OpenAI) and 3-tuple (Ollama) returns
            if len(result) == 2:
                response, usage = result
            else:
                response, usage, _ = result
            self.total_usage += usage
            
            response_text = response[0] if response else ""
            
            # Parse JSON response
            parsed = self._parse_json_response(response_text)
            
            is_correct = parsed.get("is_correct", False)
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            return EvaluationResult(
                is_correct=is_correct,
                confidence=confidence,
                reasoning=reasoning,
                usage=usage
            )
            
        except Exception as e:
            logger.warning(f"Remote verdict failed: {e}")
            return EvaluationResult(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Evaluation failed: {e}"
            )
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse the whole response
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[List[str]],
        questions: List[str]
    ) -> Tuple[List[EvaluationResult], float]:
        """
        Evaluate a batch of predictions.
        
        Returns:
            Tuple of (list of results, accuracy rate)
        """
        results = []
        correct_count = 0
        
        for pred, gt, q in zip(predictions, ground_truths, questions):
            result = self.evaluate(pred, gt, q)
            results.append(result)
            if result.is_correct:
                correct_count += 1
        
        accuracy = correct_count / len(predictions) if predictions else 0.0
        
        return results, accuracy


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Keep CorrectnessEvaluator as an alias for backward compatibility
CorrectnessEvaluator = RemoteVerdictEvaluator


# =============================================================================
# Log Writing Functions
# =============================================================================

def update_sample_log(
    results_path: Path,
    sample_id: str,
    protocol: str,
    result: EvaluationResult
) -> None:
    """
    Update sample log files (.log and .json) with correctness verdict.
    
    Args:
        results_path: Path to results directory
        sample_id: Sample identifier (e.g., "financebench_open_source_line_4")
        protocol: Protocol name (e.g., "cvoc")
        result: EvaluationResult from correctness evaluation
    """
    sample_logs_dir = results_path / "sample_logs"
    if not sample_logs_dir.exists():
        logger.warning(f"sample_logs directory not found: {sample_logs_dir}")
        return
    
    # Construct file paths
    base_name = f"{protocol}_{sample_id}"
    json_path = sample_logs_dir / f"{base_name}.json"
    log_path = sample_logs_dir / f"{base_name}.log"
    
    # Update JSON file
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            # Add correctness fields
            data["is_correct"] = result.is_correct
            data["correctness_confidence"] = result.confidence
            data["correctness_reasoning"] = result.reasoning
            
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update JSON log {json_path}: {e}")
    
    # Update .log file
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                content = f.read()
            
            # Update the "Is correct: None" line
            content = re.sub(
                r"Is correct: None",
                f"Is correct: {result.is_correct}",
                content
            )
            
            # Remove old correctness section if present
            content = re.sub(
                r"\n--- Correctness Evaluation ---.*",
                "",
                content,
                flags=re.DOTALL
            )
            
            # Append correctness evaluation section
            correctness_section = f"""
--- Correctness Evaluation ---
Is correct: {result.is_correct}
Confidence: {result.confidence:.2f}
Reasoning: {result.reasoning}
"""
            content = content.rstrip() + "\n" + correctness_section
            
            with open(log_path, "w") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to update log file {log_path}: {e}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for evaluating result directories."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remote verdict correctness evaluation for FinanceBench results"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory containing financebench_results.json"
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default="gpt-4o",
        help="Remote model to use for verdict (default: gpt-4o)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Numerical tolerance for comparison (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Update summary.txt with accuracy rate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each sample"
    )
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results_dir)
    json_path = results_path / "financebench_results.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Get results (handle different formats)
    if "results" in data:
        # New format with protocol key
        protocol = list(data["results"].keys())[0]
        samples = data["results"][protocol]
    else:
        # Try to extract protocol from directory name
        protocol = results_path.name.split("_")[0] if "_" in results_path.name else "unknown"
        samples = data
    
    # Initialize remote client
    try:
        from minions.clients.openai import OpenAIClient
        remote_client = OpenAIClient(
            model_name=args.remote_model,
            temperature=0.0
        )
        print(f"Using remote model: {args.remote_model}")
    except Exception as e:
        print(f"Error: Could not initialize remote client: {e}")
        sys.exit(1)
    
    evaluator = RemoteVerdictEvaluator(
        remote_client=remote_client,
        numerical_tolerance=args.tolerance
    )
    
    # Evaluate samples
    print(f"\nEvaluating {len(samples)} samples...")
    print("=" * 80)
    
    correct_count = 0
    total_count = 0
    total_usage = Usage()
    
    # Prepare correctness log file
    correctness_log_path = results_path / "correctness_evaluation.log"
    correctness_log_lines = []
    correctness_log_lines.append("=" * 80)
    correctness_log_lines.append("CORRECTNESS EVALUATION DETAILS")
    correctness_log_lines.append(f"Model: {args.remote_model}")
    correctness_log_lines.append(f"Numerical Tolerance: {args.tolerance * 100:.0f}%")
    correctness_log_lines.append("=" * 80)
    correctness_log_lines.append("")
    
    for i, sample in enumerate(samples, 1):
        predicted = sample.get("predicted_answer", "")
        ground_truth = sample.get("ground_truth", [])
        question = sample.get("question", "")
        sample_id = sample.get("sample_id", f"sample_{i}")
        
        # Skip errored samples
        if sample.get("error") and not predicted:
            if args.verbose:
                print(f"\n{i}. [SKIP] {sample_id} (error)")
            correctness_log_lines.append(f"{i}. [SKIP] {sample_id} (error)")
            correctness_log_lines.append("")
            continue
        
        total_count += 1
        
        result = evaluator.evaluate(predicted, ground_truth, question)
        total_usage += result.usage
        
        if result.is_correct:
            correct_count += 1
        
        # Update sample log files with correctness verdict
        update_sample_log(results_path, sample_id, protocol, result)
        
        # Log full details to correctness log
        status = "✓ CORRECT" if result.is_correct else "✗ WRONG"
        correctness_log_lines.append("-" * 80)
        correctness_log_lines.append(f"{i}. [{status}] {sample_id}")
        correctness_log_lines.append("-" * 80)
        correctness_log_lines.append(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}")
        correctness_log_lines.append(f"Ground Truth: {ground_truth}")
        correctness_log_lines.append(f"Predicted: {predicted[:300]}{'...' if len(str(predicted)) > 300 else ''}")
        correctness_log_lines.append(f"")
        correctness_log_lines.append(f"Verdict: {'CORRECT' if result.is_correct else 'WRONG'}")
        correctness_log_lines.append(f"Confidence: {result.confidence:.2f}")
        correctness_log_lines.append(f"Reasoning: {result.reasoning}")
        correctness_log_lines.append("")
        
        if args.verbose:
            print(f"\n{i}. [{status}] {sample_id}")
            print(f"   Question: {question[:100]}...")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Predicted: {str(predicted)[:100]}...")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Reasoning: {result.reasoning}")
        else:
            # Progress indicator
            status_char = "✓" if result.is_correct else "✗"
            print(status_char, end="", flush=True)
            if i % 50 == 0:
                print(f" [{i}]")
    
    if not args.verbose:
        print()  # Newline after progress dots
    
    # Write correctness log file
    correctness_log_lines.append("=" * 80)
    correctness_log_lines.append(f"SUMMARY: {correct_count}/{total_count} correct ({correct_count / total_count * 100 if total_count > 0 else 0:.2f}%)")
    correctness_log_lines.append("=" * 80)
    
    with open(correctness_log_path, "w") as f:
        f.write("\n".join(correctness_log_lines))
    print(f"\nWrote detailed correctness log to: {correctness_log_path}")
    
    # Print summary
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Accuracy Rate: {accuracy:.2f}%")
    print(f"Remote tokens: {total_usage.prompt_tokens} prompt + {total_usage.completion_tokens} completion")
    
    # Update summary if requested
    if args.update_summary:
        summary_path = results_path / "summary.txt"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                content = f.read()
            
            # Remove old accuracy line if present
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("Accuracy rate:")]
            content = "\n".join(lines).rstrip()
            
            # Add new accuracy line
            content += f"\n\nAccuracy rate: {accuracy:.2f}% ({correct_count}/{total_count} correct, remote verdict with {args.remote_model})\n"
            
            with open(summary_path, "w") as f:
                f.write(content)
            
            print(f"\nUpdated {summary_path}")
    
    return accuracy


if __name__ == "__main__":
    main()
