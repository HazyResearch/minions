from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import os
import time
import re
import pprint



from minions.minion import Minion, _extract_json
from minions.usage import Usage
from minions.minions_mcp import SyncMCPClient
from minions.prompts.minion import *

SENSITIVITY_LEVELS = {
    "uber": "Maximum cost savings: Run everything locally, prioritizing efficiency over quality",
    "high": "High cost sensitivity: Use standard protocol but bias towards efficiency",
    "medium": "Balanced: Make per-turn decisions weighing cost vs quality",
    "low": "Low cost sensitivity: Prioritize quality, run everything remotely"
}

class CostAwareMinion(Minion):
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
        mcp_client: SyncMCPClient | None = None,
        is_multi_turn=False,
        max_history_turns=10,
        cost_sensitivity="high"  # Cost sensitivity level
    ):
        """Initialize CostAwareMinion with cost sensitivity level.
        
        The CostAwareMinion intelligently balances between local and remote models
        based on the specified cost sensitivity level, optimizing for both cost 
        efficiency and output quality.
        
        Args:
            cost_sensitivity (str): One of "uber", "high", "medium", "low"
                - uber: Maximum cost savings, run everything locally
                - high: Standard protocol with efficiency bias
                - medium: Balance cost and quality with per-turn decisions
                - low: Prioritize quality over cost, run everything remotely
        """
        super().__init__(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=max_rounds,
            callback=callback,
            log_dir=log_dir,
            mcp_client=mcp_client,
            is_multi_turn=is_multi_turn,
            max_history_turns=max_history_turns,
        )
        
        if cost_sensitivity not in SENSITIVITY_LEVELS:
            raise ValueError(f"Invalid cost_sensitivity level. Must be one of: {list(SENSITIVITY_LEVELS.keys())}")
        
        self.cost_sensitivity = cost_sensitivity
        self.current_round = 0
        self.doc_metadata = None
        self.local_model_name = local_client.model_name
        self.remote_model_name = remote_client.model_name

    def _decide_model_for_turn(self, task: str, previous_response: str = None) -> str:
        """Decide whether to use local or remote model for the current turn based on cost sensitivity."""
        if self.cost_sensitivity == "uber":
            return "local"
        elif self.cost_sensitivity == "high":
            return "standard"  # Follow standard minion protocol
        elif self.cost_sensitivity == "low":
            return "remote"
        elif self.cost_sensitivity == "medium":
            # Make decision based on task complexity analysis and cost-quality tradeoff
            context_length = len(previous_response) if previous_response else 0
            
            router_prompt = TASK_ROUTER_PROMPT.format(
                task=task,
                round_num=self.current_round + 1,  # 1-based for display
                max_rounds=self.max_rounds,
                context_length=context_length,
                doc_metadata=self.doc_metadata,
                local_model_name=self.local_model_name,
                remote_model_name=self.remote_model_name
            )


            messages = [{"role": "user", "content": router_prompt}]
            decision_response, _ = self.remote_client.chat(messages, response_format={"type": "json_object"})
            decision_json = json.loads(decision_response[0])
            print(f"🔄 SUBTASK: {task}")
            print(f"🚗 ROUTING DECISION: ")
            pprint.pprint(decision_json)

            # Update current round after decision
            self.current_round += 1
            
            return decision_json["routing_decision"]
        
        return "standard"  # Default to standard protocol

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,
        is_privacy=False,
        images=None,
        is_follow_up=False,
    ):
        """Run the tunable minion protocol with specified cost sensitivity."""
        print(f"\n========== TUNABLE MINION TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Cost sensitivity level: {self.cost_sensitivity}")
        print(f"Max rounds: {max_rounds or self.max_rounds}")

        self.doc_metadata = doc_metadata
        
        # For maximum cost sensitivity, run everything locally
        if self.cost_sensitivity == "uber":
            return self._run_local_only(task, context, max_rounds, images)
            
        # For low cost sensitivity, prioritize quality with remote model
        elif self.cost_sensitivity == "low":
            return self._run_remote_only(task, context, max_rounds, images)
            
        # For high cost sensitivity, use standard protocol with efficiency bias
        elif self.cost_sensitivity == "high":
            return super().__call__(
                task=task,
                context=context,
                max_rounds=max_rounds,
                doc_metadata=doc_metadata,
                logging_id=logging_id,
                is_privacy=is_privacy,
                images=images,
                is_follow_up=is_follow_up
            )
            
        # For medium sensitivity, balance cost and quality
        else:  # medium
            return self._run_adaptive(
                task=task,
                context=context,
                max_rounds=max_rounds,
                doc_metadata=doc_metadata,
                logging_id=logging_id,
                is_privacy=is_privacy,
                images=images,
                is_follow_up=is_follow_up
            )

    def _run_local_only(self, task: str, context: List[str], max_rounds=None, images=None):
        """Run the entire task using only the local model."""
        if max_rounds is None:
            max_rounds = self.max_rounds

        print(f"Running local only with max rounds: {max_rounds}")
        
            
        start_time = time.time()
        local_usage = Usage()

        # concatenate context
        context = "\n\n".join(context)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nTask: {task}",
                "images": images
            }
        ]

        
        # Get response from local model
        response, usage, _ = self.local_client.chat(messages)
        print(f"Response: {response}")
        local_usage += usage
        
        # Prepare return structure
        timing = {
            "local_call_time": time.time() - start_time,
            "remote_call_time": 0,
            "total_time": time.time() - start_time,
            "overhead_time": 0
        }
        
        return {
            "final_answer": response[0],
            "supervisor_messages": [],
            "worker_messages": messages,
            "remote_usage": Usage(),
            "local_usage": local_usage,
            "timing": timing,
            "conversation_log": {
                "task": task,
                "context": context,
                "conversation": [
                    {
                        "user": "local",
                        "prompt": task,
                        "output": response[0]
                    }
                ],
                "generated_final_answer": response[0],
                "usage": {"local": local_usage.to_dict(), "remote": {}}
            }
        }

    def _run_remote_only(self, task: str, context: List[str], max_rounds=None, images=None):
        """Run the entire task using only the remote model."""
        if max_rounds is None:
            max_rounds = self.max_rounds
            
        start_time = time.time()
        remote_usage = Usage()
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nTask: {task}",
                "images": images
            }
        ]
        
        # Get response from remote model
        response, usage = self.remote_client.chat(messages)
        remote_usage += usage
        
        # Prepare return structure
        timing = {
            "local_call_time": 0,
            "remote_call_time": time.time() - start_time,
            "total_time": time.time() - start_time,
            "overhead_time": 0
        }
        
        return {
            "final_answer": response[0],
            "supervisor_messages": messages,
            "worker_messages": [],
            "remote_usage": remote_usage,
            "local_usage": Usage(),
            "timing": timing,
            "conversation_log": {
                "task": task,
                "context": context,
                "conversation": [
                    {
                        "user": "remote",
                        "prompt": task,
                        "output": response[0]
                    }
                ],
                "generated_final_answer": response[0],
                "usage": {"remote": remote_usage.to_dict(), "local": {}}
            }
        }

    def _run_adaptive(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,
        is_privacy=False,
        images=None,
        is_follow_up=False,
    ):
        """Run the task with per-turn decisions between local and remote models while following minion protocol."""
        if max_rounds is None:
            max_rounds = self.max_rounds
            
        start_time = time.time()
        timing = {
            "local_call_time": 0,
            "remote_call_time": 0,
            "total_time": 0,
            "overhead_time": 0
        }
        
        local_usage = Usage()
        remote_usage = Usage()
        
        # Initialize conversation log
        conversation_log = {
            "task": task,
            "context": context,
            "conversation": [],
            "generated_final_answer": "",
            "usage": {"remote": {}, "local": {}}
        }

        # Join context sections
        context = "\n\n".join(context)
        print(f"Context length: {len(context)} characters")

        # Initialize message histories
        supervisor_messages = []
        worker_messages = []

        # Initial supervisor prompt
        supervisor_messages = [
            {
                "role": "user",
                "content": self.supervisor_initial_prompt.format(
                    task=task,
                    max_rounds=max_rounds,
                    mcp_tools_info=None
                ),
            }
        ]

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "remote",
                "prompt": supervisor_messages[0]["content"],
                "output": None,
            }
        )

        # Initial supervisor call to get first question
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        # First turn always uses remote model for task decomposition
        remote_start_time = time.time()
        supervisor_response, supervisor_usage = self.remote_client.chat(
            messages=supervisor_messages
        )
        timing["remote_call_time"] += time.time() - remote_start_time
        remote_usage += supervisor_usage

        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )
        conversation_log["conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1], is_final=False)

        # Extract first question for worker
        supervisor_json = _extract_json(supervisor_response[0])
        print(f"💬 SUPERVISOR MESSAGE: {supervisor_json['message']}")

        worker_messages.append({"role": "user", "content": supervisor_json["message"]})
        
        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "local",
                "prompt": supervisor_json["message"],
                "output": None,
            }
        )

        final_answer = None
        local_output = None
        sub_task = supervisor_json["message"]

        for round in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)

            # Decide which model to use for this turn
            model_decision = self._decide_model_for_turn(sub_task, local_output)
            print(f"🔄 Using {model_decision} model for round {round + 1}")

            if model_decision == "local":
                # Initialize worker system prompt if not done
                if len(worker_messages) == 1:  # Only has the user message
                    worker_messages.insert(0, {
                        "role": "system",
                        "content": WORKER_SYSTEM_PROMPT.format(
                            context=context, task=task
                        ),
                        "images": images
                    })

                local_start_time = time.time()
                worker_response, worker_usage, _ = self.local_client.chat(
                    messages=worker_messages
                )
                timing["local_call_time"] += time.time() - local_start_time
                local_usage += worker_usage
                local_output = worker_response[0]

            else:  # remote
                remote_start_time = time.time()
                if len(worker_messages) == 1:
                    worker_messages.insert(0, {
                        "role": "system",
                        "content": WORKER_SYSTEM_PROMPT.format(
                            context=context, task=task
                        ),
                        "images": images
                    })
                worker_response, worker_usage = self.remote_client.chat(
                    messages=worker_messages
                )
                timing["remote_call_time"] += time.time() - remote_start_time
                remote_usage += worker_usage
                local_output = worker_response[0]

            print(f"🔄 Worker response: {local_output}")

            worker_messages.append(
                {"role": "assistant", "content": local_output}
            )
            conversation_log["conversation"][-1]["output"] = local_output

            if self.callback:
                self.callback("worker", worker_messages[-1], is_final=False)

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=local_output
                )

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )
            else:
                # First step: Think through the synthesis
                cot_prompt = REMOTE_SYNTHESIS_COT.format(
                    response=local_output
                )

                # Add supervisor COT prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": cot_prompt, "output": None}
                )

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                # Track remote call time for step-by-step thinking
                remote_start_time = time.time()
                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages
                )
                timing["remote_call_time"] += time.time() - remote_start_time
                remote_usage += usage

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )
                conversation_log["conversation"][-1]["output"] = step_by_step_response[0]

                # Second step: Get structured output
                supervisor_prompt = self.remote_synthesis_final.format(
                    response=step_by_step_response[0]
                )

                # Add supervisor synthesis prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Always use remote for final decision
            remote_start_time = time.time()
            supervisor_response, supervisor_usage = self.remote_client.chat(
                supervisor_messages
            )
            timing["remote_call_time"] += time.time() - remote_start_time
            remote_usage += supervisor_usage

            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            conversation_log["conversation"][-1]["output"] = supervisor_response[0]

            if self.callback:
                self.callback("supervisor", supervisor_messages[-1], is_final=False)

            # Parse supervisor's decision
            supervisor_json = _extract_json(supervisor_response[0])

            if supervisor_json["decision"] == "provide_final_answer":
                final_answer = supervisor_json["answer"]
                conversation_log["generated_final_answer"] = final_answer
                
                if self.callback:
                    self.callback(
                        "supervisor",
                        {"role": "assistant", "content": final_answer},
                        is_final=True
                    )
                break
            else:
                next_question = supervisor_json["message"]
                worker_messages.append({"role": "user", "content": next_question})

                # Add next worker prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "local",
                        "prompt": next_question,
                        "output": None,
                    }
                )
                sub_task = next_question

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Calculate total time and overhead
        end_time = time.time()
        timing["total_time"] = end_time - start_time
        timing["overhead_time"] = timing["total_time"] - (
            timing["local_call_time"] + timing["remote_call_time"]
        )

        # Update usage statistics
        conversation_log["usage"]["remote"] = remote_usage.to_dict()
        conversation_log["usage"]["local"] = local_usage.to_dict()

        # Log the final result
        if logging_id:
            log_filename = f"{logging_id}_minion.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        print(f"\n=== SAVING LOG TO {log_path} ===")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log to {log_path}: {e}")

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
            "conversation_log": conversation_log,
            "timing": timing,
        } 