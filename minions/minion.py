from typing import List, Dict, Any
import json
import re
import os
from datetime import datetime

from minions.clients import *

from minions.prompts.minion import (
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_INITIAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL
)
from minions.usage import Usage


def _escape_newlines_in_strings(json_str: str) -> str:
    # This regex naively matches any content inside double quotes (including escaped quotes)
    # and replaces any literal newline characters within those quotes.
    # was especially useful for anthropic client
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    # Minimal fix: escape newlines only within quoted JSON strings.
    json_str = _escape_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        raise


class Minion:
    def __init__(
        self, local_client=None, remote_client=None, max_rounds=3, callback=None, log_dir="minion_logs"
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def __call__(
        self, task: str, context: List[str], max_rounds=None, doc_metadata=None, task_id=None
    ):
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            task_id: Optional identifier for the task, used for named log files

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """
        if max_rounds is None:
            max_rounds = self.max_rounds

        # Join context sections
        context = "\n\n".join(context)

        # Initialize the log structure
        conversation_log= {
            "TASK": task,
            "CONTEXT": context,
            "Conversation": [],
            "Generated_final_answer": ""
        }

        # Initialize message histories and usage tracking
        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
            }
        ]

        # Add initial supervisor prompt to conversation log
        conversation_log["Conversation"].append({
            "user": "remote",
            "prompt": SUPERVISOR_INITIAL_PROMPT.format(task=task),
            "output": None
        })

        worker_messages = [
            {
                "role": "system",
                "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
            }
        ]

        remote_usage = Usage()
        local_usage = Usage()

        # Initial supervisor call to get first question
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages,
                response_format={"type": "json_object"}
            )
        else:
            supervisor_response, supervisor_usage = self.remote_client.chat(messages=supervisor_messages)

        remote_usage += supervisor_usage
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        # Update the last conversation entry with the ouput
        conversation_log["Conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Extract first question for worker
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_json = json.loads(supervisor_response[0])
        else:
            supervisor_json = _extract_json(supervisor_response[0])
        worker_messages.append({"role": "user", "content": supervisor_json["message"]})

        # Add worker prompt to conversation log
        conversation_log["Conversation"].append({
            "user": "local",
            "prompt": supervisor_json["message"],
            "output": None
        })

        final_answer = None
        print("entering loop with max_rounds: ", max_rounds)
        for round in range(max_rounds):
            # Get worker's response
            print("getting worker's response")
            if self.callback:
                self.callback("worker", None, is_final=False)
            worker_response, worker_usage, done_reason = self.local_client.chat(messages=worker_messages)
            local_usage += worker_usage
            worker_messages.append({"role": "assistant", "content": worker_response[0]})

            # Update the last conversation entry with the output
            conversation_log["Conversation"][-1]["output"] = worker_response[0]

            if self.callback:
                self.callback("worker", worker_messages[-1])

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )

                # Add supervisor final prompt to conversation log
                conversation_log["Conversation"].append({
                    "user": "remote",
                    "prompt": supervisor_prompt,
                    "output": None
                })
            else:
                # First step: Think through the synthesis
                cot_prompt = REMOTE_SYNTHESIS_COT.format(
                    response=worker_response[0]
                )

                # Add supervisor COT prompt to conversation log
                conversation_log["Conversation"].append({
                    "user": "remote",
                    "prompt": cot_prompt,
                    "output": None
                })

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage = self.remote_client.chat(supervisor_messages)

                remote_usage += usage
                if self.callback:
                    self.callback("supervisor", None, is_final=False)

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["Conversation"][-1]["output"] = step_by_step_response[0]

                # Second step: Get structured output
                supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                    response = step_by_step_response[0]
                )

                # Add supervisor synthesis prompt to conversation log
                conversation_log["Conversation"].append({
                    "user": "remote",
                    "prompt": supervisor_prompt,
                    "output": None
                })

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Get supervisor's response
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"}
                )
            else:
                supervisor_response, supervisor_usage = self.remote_client.chat(messages=supervisor_messages)
            
            remote_usage += supervisor_usage
            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            if self.callback:
                self.callback("supervisor", supervisor_messages[-1])

            # Parse supervisor's decision
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_json = json.loads(supervisor_response[0])
            else:
                supervisor_json = _extract_json(supervisor_response[0])
            print("Supervisor_json:", supervisor_json)
            if supervisor_json["decision"] == "provide_final_answer":
                final_answer = supervisor_json["answer"]
                conversation_log["Generated_final_answer"] = final_answer
                break
            else:
                next_question = supervisor_json["message"]
                worker_messages.append(
                    {"role": "user", "content": next_question}
                )

                # Add next worker prompt to conversation log
                conversation_log["Conversation"].append({
                    "user": "local",
                    "prompt": next_question,
                    "output": None
                })

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["Generated_final_answer"] = final_answer

        # Log the final result
        if task_id:
            # use provided task_id
            log_filename = f"{task_id}_minion.json"
        else:
            # fall back to timestamp + task abbrev
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r'[^a-zA-Z0-9]', '_', task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path
        }
