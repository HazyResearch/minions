import json
import re
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from minions.prompts.minion_reasoning import (
    WORKER_REASONING_PROMPT,
    SUPERVISOR_REASONING_REVIEW_PROMPT
)

from minions.usage import Usage

#### HELPER FUNCTIONS ####
def extract_final_answer(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    matches = list(re.finditer(r'the answer is[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE))
    if matches:
        answer = matches[-1].group(1).strip()
        answer = re.sub(r'[.!?]+$', '', answer).strip()
        return answer
    
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    return lines[-1] if lines else text.strip()


def format_attempts_for_supervisor(attempts: List[Dict[str, Any]]) -> str:
    """
    Format worker attempts into a readable string for the supervisor.
    """
    attempts_text = ""
    for i, attempt in enumerate(attempts, 1):
        attempts_text += f"\n### Attempt {i}:\n"
        attempts_text += f"Extracted Answer: {attempt['extracted_answer']}\n"
        # Truncate full response for token efficiency while keeping key reasoning
        response_preview = attempt['full_response'][:800]
        if len(attempt['full_response']) > 800:
            response_preview += "..."
        attempts_text += f"Full Reasoning:\n{response_preview}\n"
    return attempts_text


#### Main - MINION REASONING CLASS ####

class MinionReasoning:
    """
    This class implements a multi-agent reasoning system where:
    1. A local worker model generates N independent solution attempts
    2. A remote supervisor model reviews attempts and selects the best based on quality
    3. Optionally, a judge model evaluates correctness against a reference answer
    """
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        num_attempts: int = 3,
        worker_temperature: float = 0.7,
        worker_max_tokens: int = 4096,
        supervisor_temperature: float = 0.3,
        supervisor_max_tokens: int = 1000,
        judge_max_tokens: int = 500,
        max_rounds: int = 1,
        callback=None,
        log_dir: str = "minion_reasoning_logs",
    ):
        self.local_client = local_client
        self.remote_client = remote_client
        self.num_attempts = num_attempts
        self.worker_temperature = worker_temperature
        self.worker_max_tokens = worker_max_tokens
        self.supervisor_temperature = supervisor_temperature
        self.supervisor_max_tokens = supervisor_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    def __call__(
        self,
        task: str,
        context: Optional[List[str]] = None,
        doc_metadata: Optional[str] = None,
        logging_id: Optional[str] = None,
        reference_answer: Optional[str] = None,
        use_judge: bool = False,
        max_rounds: Optional[int] = None,  # For API compatibility, not used in reasoning
        images: Optional[List[str]] = None,  # For API compatibility
    ) -> Dict[str, Any]:
        
        print("Start Minion Reasoning Task...")
        print(f"Task: {task[:100]}...")
        print(f"Number of worker attempts: {self.num_attempts}")
        print(f"Worker temperature: {self.worker_temperature}")
        print(f"Context provided: {len(context) if context else 0} items")
        print(f"Reference answer provided: {reference_answer is not None}")

        worker_usage = Usage()
        supervisor_usage = Usage()
        judge_usage = Usage()

        context_str = ""
        if context and len(context) > 0:
            context_str = "\n\n## Context:\n" + "\n\n".join(context)

        if doc_metadata:
            context_str += f"\n\n## Document Information:\n{doc_metadata}"

        conversation_log = {
            "task": task,
            "context": context,
            "doc_metadata": doc_metadata,
            "reference_answer": reference_answer,
            "num_attempts": self.num_attempts,
            "worker_temperature": self.worker_temperature,
            "worker_model": getattr(self.local_client, 'model_name', 'unknown'),
            "supervisor_model": getattr(self.remote_client, 'model_name', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "worker_attempts": [],
            "supervisor_analysis": {},
            "judge_evaluation": {},
        }

        # Phase 1: Worker generates multiple solution attempts
        print(f"Starting {self.num_attempts} workers in parallel...")
        print(f"Worker model: {getattr(self.local_client, 'model_name', 'unknown')}")
        print(f"Temperature: {self.worker_temperature}")
        print(f"Each worker will independently solve the task...")

        if self.callback:
            self.callback("worker", {"role": "assistant", "content": f"Starting {self.num_attempts} workers in parallel (temperature={self.worker_temperature})..."})

        worker_attempts = []

        for attempt_idx in range(self.num_attempts):
            print(f"\n--- Worker Attempt {attempt_idx + 1}/{self.num_attempts} ---")

            # Create worker prompt with context if provided
            full_task = task + context_str
            worker_prompt = WORKER_REASONING_PROMPT.format(problem=full_task)

            # Call worker model
            worker_messages = [{"role": "user", "content": worker_prompt}]

            if self.callback:
                self.callback("worker", {"role": "assistant", "content": f"👷 Worker {attempt_idx + 1}/{self.num_attempts} generating solution..."})

            try:
                # Get worker response
                # Note: temperature and max_tokens are already set in the client initialization
                response, usage, _ = self.local_client.chat(
                    messages=worker_messages,
                )
                full_response = response[0] if isinstance(response, list) else response
                extracted_answer = extract_final_answer(full_response)
                print(f"Extracted answer: {extracted_answer}")
                attempt = {
                    'attempt_number': attempt_idx + 1,
                    'full_response': full_response,
                    'extracted_answer': extracted_answer,
                }
                worker_attempts.append(attempt)
                worker_usage += usage
                if self.callback:
                    self.callback("worker", {"role": "assistant", "content": full_response})
            except Exception as e:
                print(f"Error in worker attempt {attempt_idx + 1}: {e}")
                worker_attempts.append({
                    'attempt_number': attempt_idx + 1,
                    'full_response': f"Error: {str(e)}",
                    'extracted_answer': "Error",
                })
        conversation_log["worker_attempts"] = worker_attempts
        print(f"\nGenerated {len(worker_attempts)} worker attempts")

        # Phase 2: Supervisor reviews attempts and selects best
        print(f"Supervisor reviewing {len(worker_attempts)} worker attempts...")
        if self.callback:
            self.callback("supervisor", {"role": "assistant", "content": f"🔍 Supervisor reviewing {len(worker_attempts)} worker attempts..."})

        # Format attempts for supervisor
        attempts_text = format_attempts_for_supervisor(worker_attempts)

        # Create supervisor prompt
        supervisor_prompt = SUPERVISOR_REASONING_REVIEW_PROMPT.format(
            problem=task,
            num_attempts=len(worker_attempts),
            attempts_text=attempts_text,
        )

        supervisor_messages = [{"role": "user", "content": supervisor_prompt}]

        try:
            supervisor_response, usage, _ = self.remote_client.chat(
                messages=supervisor_messages,
                response_format={"type": "json_object"},
            )
            supervisor_usage += usage
            supervisor_text = supervisor_response[0] if isinstance(supervisor_response, list) else supervisor_response
            print(f"\nSupervisor response: {supervisor_text[:200]}...")

            try:
                supervisor_json = json.loads(supervisor_text)
            except json.JSONDecodeError:
                print("Warning: Failed to parse supervisor JSON, using fallback")
                supervisor_json = {
                    "analysis": supervisor_text,
                    "verified_answer": extract_final_answer(supervisor_text),
                    "confidence": "unknown",
                    "correct_attempts": [],
                    "best_attempt": 1,
                }

            verified_answer = supervisor_json.get('verified_answer', '')
            if verified_answer.startswith('The answer is: '):
                verified_answer = verified_answer[15:]

            confidence = supervisor_json.get('confidence', 'unknown')
            analysis = supervisor_json.get('analysis', '')
            best_attempt = supervisor_json.get('best_attempt', 1)
            correct_attempts = supervisor_json.get('correct_attempts', [])

            print(f"\nSupervisor selected: {verified_answer}")
            print(f"Confidence: {confidence}")
            print(f"Best attempt: {best_attempt}")

            conversation_log["supervisor_analysis"] = {
                "verified_answer": verified_answer,
                "confidence": confidence,
                "analysis": analysis,
                "best_attempt": best_attempt,
                "correct_attempts": correct_attempts,
                "full_response": supervisor_text,
            }

            if self.callback:
                self.callback("supervisor", {"role": "assistant", "content": supervisor_text})

        except Exception as e:
            print(f"Error in supervisor review: {e}")
            verified_answer = ""
            confidence = "error"
            analysis = f"Error: {str(e)}"
            conversation_log["supervisor_analysis"] = {
                "error": str(e),
                "verified_answer": "",
                "confidence": "error",
            }

        # Save log file
        if logging_id:
            log_filename = f"{logging_id}_minion_reasoning.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:30])
            log_filename = f"{timestamp}_{safe_task}.json"

        log_path = os.path.join(self.log_dir, log_filename)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        print(f"\nLog saved to: {log_path}")

        # Return results
        print("\n" + "="*80)
        print("MINION REASONING TASK COMPLETED")
        print("="*80)

        result = {
            "final_answer": verified_answer,  #
            "verified_answer": verified_answer,  
            "confidence": confidence,
            "analysis": analysis,
            "worker_attempts": worker_attempts,
            "worker_usage": worker_usage,
            "supervisor_usage": supervisor_usage,
            "log_file": log_path,
        }
        return result

