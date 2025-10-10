"""
Minion protocol variant using Arch Router for intelligent client selection.

This module implements the Minion protocol with local Arch Router-based
model selection instead of expensive remote LLM routing (as in autominion.py).
"""

from typing import List, Dict, Any, Optional
import json
import re
import os
import logging
from datetime import datetime

from minions.clients.base import MinionsClient
from minions.clients import OpenAIClient, TogetherClient, GeminiClient
from minions.utils.arch_router import ArchRouter
from minions.usage import Usage

from minions.prompts.minion import (
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_INITIAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL,
)


def _escape_newlines_in_strings(json_str: str) -> str:
    """Escape newline characters within JSON string values."""
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

    json_str = _escape_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON: {json_str}")
        raise


class Minion:
    """
    Minion protocol with Arch Router-based client selection.

    Uses a local 1.5B Arch Router model for cheap, intelligent routing
    based on task characteristics instead of expensive remote LLM.
    """

    def __init__(
        self,
        remote_client: MinionsClient,
        local_clients: Dict[str, MinionsClient],
        client_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        arch_router: Optional[ArchRouter] = None,
        max_rounds: int = 3,
        callback=None,
        log_dir: str = "minion_logs",
        verbose: bool = False
    ):
        """
        Initialize Arch Router Minion.

        Args:
            remote_client: Remote supervisor client
            local_clients: Dictionary of available local worker clients
            client_metadata: Optional metadata for routing decisions
            arch_router: Optional pre-initialized ArchRouter instance
            max_rounds: Maximum number of supervisor-worker conversation rounds
            callback: Optional callback function to receive message updates
            log_dir: Directory for conversation logs
            verbose: Enable verbose logging
        """
        if not local_clients:
            raise ValueError("local_clients cannot be empty. Provide at least one local worker client.")

        self.remote_client = remote_client
        self.local_clients = local_clients
        self.client_metadata = client_metadata or self._default_metadata()
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        self._arch_router = arch_router

        os.makedirs(log_dir, exist_ok=True)

        if verbose:
            self.logger.info("Initialized Arch Router Minion")
            self.logger.info(f"Available local clients: {list(local_clients.keys())}")
            self.logger.info(f"Remote supervisor: {remote_client.model_name}")

    @property
    def arch_router(self) -> ArchRouter:
        """Lazy-load Arch Router to avoid loading model if not needed."""
        if self._arch_router is None:
            if self.verbose:
                self.logger.info("Loading Arch Router model")
            self._arch_router = ArchRouter(verbose=self.verbose)
        return self._arch_router

    def _default_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Generate default metadata for clients if not provided.

        Returns:
            Dictionary mapping client names to default metadata in Arch-Router format.
            Format: {"name": str, "description": str}
        """
        return {
            name: {
                "name": name,
                "description": f"General purpose model ({name}) for general domain tasks"
            }
            for name in self.local_clients.keys()
        }

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds: Optional[int] = None,
        doc_metadata: Optional[str] = None,
        logging_id: Optional[str] = None,
        images: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Run the Arch Router Minion protocol to answer a task.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task
            images: Optional list of images for multimodal tasks

        Returns:
            Dictionary containing final answer, routing decision, and usage statistics
        """
        self.logger.info("Arch Router Minion protocol started")
        self.logger.info(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")
        self.logger.info(f"Max rounds: {max_rounds or self.max_rounds}")
        self.logger.info(f"Context length: {sum(len(c) for c in context)} characters")

        if max_rounds is None:
            max_rounds = self.max_rounds

        full_context = "\n\n".join(context)

        conversation_log = {
            "task": task,
            "context": full_context,
            "selected_client": None,
            "routing_decision": None,
            "conversation": [],
            "generated_final_answer": "",
            "usage": {
                "remote": {},
                "local": {},
            },
        }

        self.logger.info(f"Using Arch Router to select worker client from: {list(self.local_clients.keys())}")

        routing_decision = self.arch_router.route(
            query=task,
            available_clients=self.client_metadata
        )

        selected_client_name = routing_decision["route"]
        self.logger.info(f"Arch Router selected: {selected_client_name}")
        if "reasoning" in routing_decision:
            self.logger.debug(f"Routing reasoning: {routing_decision['reasoning']}")

        selected_client = self.local_clients.get(selected_client_name)
        if selected_client is None:
            raise ValueError(
                f"Arch Router selected '{selected_client_name}' but it's not in local_clients. "
                f"Available: {list(self.local_clients.keys())}"
            )

        conversation_log["selected_client"] = selected_client_name
        conversation_log["routing_decision"] = routing_decision

        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
            }
        ]

        worker_messages = [
            {
                "role": "system",
                "content": WORKER_SYSTEM_PROMPT.format(context=full_context, task=task),
            }
        ]

        if images:
            worker_messages[0]["images"] = images

        remote_usage = Usage()
        local_usage = Usage()

        if self.callback:
            self.callback("supervisor", None, is_final=False)

        self.logger.debug("Starting supervisor-worker conversation")

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages,
                response_format={"type": "json_object"}
            )
        elif isinstance(self.remote_client, GeminiClient):
            from pydantic import BaseModel

            class SupervisorOutput(BaseModel):
                decision: str
                message: str
                answer: str

            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": SupervisorOutput,
                },
            )
        else:
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages
            )

        remote_usage += supervisor_usage
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        conversation_log["conversation"].append({
            "user": "remote",
            "prompt": SUPERVISOR_INITIAL_PROMPT.format(task=task),
            "output": supervisor_response[0]
        })

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        try:
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient, GeminiClient)):
                supervisor_json = json.loads(supervisor_response[0])
            else:
                supervisor_json = _extract_json(supervisor_response[0])
        except Exception as e:
            self.logger.warning(f"Failed to parse supervisor JSON: {e}")
            supervisor_json = _extract_json(supervisor_response[0])

        first_question = supervisor_json.get("message", supervisor_json.get("question", ""))
        worker_messages.append({"role": "user", "content": first_question})

        conversation_log["conversation"].append({
            "user": "local",
            "prompt": first_question,
            "output": None
        })

        final_answer = None
        for round_num in range(max_rounds):
            self.logger.debug(f"Round {round_num + 1}/{max_rounds}")

            if self.callback:
                self.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason = selected_client.chat(
                messages=worker_messages
            )

            local_usage += worker_usage
            worker_messages.append(
                {"role": "assistant", "content": worker_response[0]}
            )

            conversation_log["conversation"][-1]["output"] = worker_response[0]

            if self.callback:
                self.callback("worker", worker_messages[-1])

            if round_num == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )

                conversation_log["conversation"].append({
                    "user": "remote",
                    "prompt": supervisor_prompt,
                    "output": None
                })
            else:
                cot_prompt = REMOTE_SYNTHESIS_COT.format(response=worker_response[0])

                conversation_log["conversation"].append({
                    "user": "remote",
                    "prompt": cot_prompt,
                    "output": None
                })

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages
                )

                remote_usage += usage
                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                conversation_log["conversation"][-1]["output"] = step_by_step_response[0]

                supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                    response=step_by_step_response[0]
                )

                conversation_log["conversation"].append({
                    "user": "remote",
                    "prompt": supervisor_prompt,
                    "output": None
                })

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
                )
            elif isinstance(self.remote_client, GeminiClient):
                from pydantic import BaseModel

                class RemoteOutput(BaseModel):
                    decision: str
                    message: str
                    answer: str

                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": RemoteOutput,
                    },
                )
            else:
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages
                )

            remote_usage += supervisor_usage
            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )

            if self.callback:
                self.callback("supervisor", supervisor_messages[-1])

            conversation_log["conversation"][-1]["output"] = supervisor_response[0]

            try:
                if isinstance(self.remote_client, (OpenAIClient, TogetherClient, GeminiClient)):
                    supervisor_json = json.loads(supervisor_response[0])
                else:
                    supervisor_json = _extract_json(supervisor_response[0])
            except Exception as e:
                self.logger.warning(f"Failed to parse supervisor JSON: {e}")
                supervisor_json = _extract_json(supervisor_response[0])

            if supervisor_json.get("decision") == "provide_final_answer":
                final_answer = supervisor_json.get("answer", "")
                conversation_log["generated_final_answer"] = final_answer
                self.logger.info(f"Final answer provided in round {round_num + 1}")
                break
            else:
                next_question = supervisor_json.get("message", "")
                worker_messages.append({"role": "user", "content": next_question})

                conversation_log["conversation"].append({
                    "user": "local",
                    "prompt": next_question,
                    "output": None
                })

        if final_answer is None:
            final_answer = "No answer found within max rounds."
            conversation_log["generated_final_answer"] = final_answer
            self.logger.warning(f"Max rounds ({max_rounds}) reached without final answer")

        conversation_log["usage"]["remote"] = remote_usage.to_dict()
        conversation_log["usage"]["local"] = local_usage.to_dict()

        if logging_id:
            log_filename = f"{logging_id}_minion_arch.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}_arch.json"

        log_path = os.path.join(self.log_dir, log_filename)

        self.logger.debug(f"Saving log to: {log_path}")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        self.logger.info("Arch Router Minion protocol completed")

        return {
            "final_answer": final_answer,
            "selected_client": selected_client_name,
            "routing_decision": routing_decision,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
        }
