from typing import List, Dict, Any, Optional
import json
import re
import os
from datetime import datetime

from minions.clients import OpenAIClient, TogetherClient

from minions.prompts.minion import (
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_INITIAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL,
    WORKER_PRIVACY_SHIELD_PROMPT,
    REFORMAT_QUERY_PROMPT,
)
from minions.prompts.multi_turn import (
    MULTI_TURN_WORKER_SYSTEM_PROMPT,
    MULTI_TURN_SUPERVISOR_INITIAL_PROMPT,
    MULTI_TURN_SUPERVISOR_CONVERSATION_PROMPT,
    MULTI_TURN_SUPERVISOR_FINAL_PROMPT,
    MULTI_TURN_CONVERSATION_HISTORY_FORMAT,
)
from minions.usage import Usage
from minions.utils.conversation_history import ConversationHistory, ConversationTurn


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
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
        is_multi_turn=False,
        max_history_turns=10,
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
            log_dir: Directory for logging conversation history
            is_multi_turn: Whether to enable multi-turn conversation support
            max_history_turns: Maximum number of turns to keep in conversation history
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        self.is_multi_turn = is_multi_turn
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize conversation history for multi-turn support
        self.conversation_history = ConversationHistory(max_turns=max_history_turns) if is_multi_turn else None

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,  # this is the name/id to give to the logging .json file
        is_privacy=False,
        images=None,
        is_follow_up=False,
    ):
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Join context sections
        context = "\n\n".join(context)

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": context,
            "conversation": [],
            "generated_final_answer": "",
        }

        # Initialize message histories and usage tracking
        supervisor_messages = []
        worker_messages = []
        remote_usage = Usage()
        local_usage = Usage()

        # Format conversation history for multi-turn mode
        formatted_history = ""
        if self.is_multi_turn and len(self.conversation_history.turns) > 0:
            formatted_history = self._format_conversation_history()

        # print whether privacy is enabled
        print("Privacy is enabled: ", is_privacy)

        # if privacy import from minions.utils.pii_extraction
        if is_privacy:
            from minions.utils.pii_extraction import PIIExtractor

            # Extract PII from context
            pii_extractor = PIIExtractor()
            str_context = context
            pii_extracted = pii_extractor.extract_pii(str_context)

            # Extract PII from query
            query_pii_extracted = pii_extractor.extract_pii(task)
            reformat_query_task = REFORMAT_QUERY_PROMPT.format(
                query=task, pii_extracted=str(query_pii_extracted)
            )

            # Clean PII from query
            reformatted_task, usage, done_reason = self.local_client.chat(
                messages=[{"role": "user", "content": reformat_query_task}]
            )
            local_usage += usage
            pii_reformatted_task = reformatted_task[0]

            # Log the reformatted task
            output = f"""**PII Reformated Task:**
            {pii_reformatted_task}
            """

            if self.callback:
                self.callback("worker", output)

            # Initialize message histories based on conversation mode
            if self.is_multi_turn:
                supervisor_messages = [
                    {
                        "role": "user",
                        "content": MULTI_TURN_SUPERVISOR_INITIAL_PROMPT.format(
                            task=pii_reformatted_task,
                            conversation_history=formatted_history
                        ),
                    }
                ]
                worker_messages = [
                    {
                        "role": "system",
                        "content": MULTI_TURN_WORKER_SYSTEM_PROMPT.format(
                            context=context, 
                            task=task,
                            conversation_history=formatted_history
                        ),
                    }
                ]
            else:
                supervisor_messages = [
                    {
                        "role": "user",
                        "content": SUPERVISOR_INITIAL_PROMPT.format(
                            task=pii_reformatted_task
                        ),
                    }
                ]
                worker_messages = [
                    {
                        "role": "system",
                        "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                    }
                ]
        else:
            # Initialize message histories based on conversation mode
            if self.is_multi_turn:
                supervisor_messages = [
                    {
                        "role": "user",
                        "content": MULTI_TURN_SUPERVISOR_INITIAL_PROMPT.format(
                            task=task,
                            conversation_history=formatted_history
                        ),
                    }
                ]
                worker_messages = [
                    {
                        "role": "system",
                        "content": MULTI_TURN_WORKER_SYSTEM_PROMPT.format(
                            context=context, 
                            task=task,
                            conversation_history=formatted_history
                        ),
                        "images": images,
                    }
                ]
            else:
                supervisor_messages = [
                    {
                        "role": "user",
                        "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
                    }
                ]
                worker_messages = [
                    {
                        "role": "system",
                        "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                        "images": images,
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

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages, response_format={"type": "json_object"}
            )
        else:
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages
            )

        remote_usage += supervisor_usage
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        # Update the last conversation entry with the ouput
        conversation_log["conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Extract first question for worker
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            try:
                supervisor_json = json.loads(supervisor_response[0])

            except:
                supervisor_json = _extract_json(supervisor_response[0])
        else:
            supervisor_json = _extract_json(supervisor_response[0])

        worker_messages.append({"role": "user", "content": supervisor_json["message"]})

        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {"user": "local", "prompt": supervisor_json["message"], "output": None}
        )

        final_answer = None
        local_output = None
        
        for round in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason = self.local_client.chat(
                messages=worker_messages
            )

            local_usage += worker_usage

            if is_privacy:
                if self.callback:
                    output = f"""**_My output (pre-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)

                worker_privacy_shield_prompt = WORKER_PRIVACY_SHIELD_PROMPT.format(
                    output=worker_response[0],
                    pii_extracted=str(pii_extracted),
                )
                worker_response, worker_usage, done_reason = self.local_client.chat(
                    messages=[{"role": "user", "content": worker_privacy_shield_prompt}]
                )
                local_usage += worker_usage

                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )
                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    output = f"""**_My output (post-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)
            else:
                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    self.callback("worker", worker_messages[-1])

            # Save the local output for conversation history
            local_output = worker_response[0]

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                if self.is_multi_turn:
                    supervisor_prompt = MULTI_TURN_SUPERVISOR_FINAL_PROMPT.format(
                        response=worker_response[0],
                        conversation_history=formatted_history
                    ) + "\n\nIMPORTANT: Provide a direct answer to the user's question. DO NOT describe what the answer should contain."
                else:
                    supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                        response=worker_response[0]
                    )

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )
            else:
                # First step: Think through the synthesis
                if self.is_multi_turn:
                    cot_prompt = REMOTE_SYNTHESIS_COT.format(
                        response=worker_response[0]
                    )
                else:
                    cot_prompt = REMOTE_SYNTHESIS_COT.format(
                        response=worker_response[0]
                    )

                # Add supervisor COT prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": cot_prompt, "output": None}
                )

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages
                )

                remote_usage += usage

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = step_by_step_response[
                    0
                ]

                # Second step: Get structured output
                if self.is_multi_turn:
                    supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                        response=step_by_step_response[0]
                    ) + "\n\nIMPORTANT: Provide a direct answer to the user's question. DO NOT describe what the answer should contain."
                else:
                    supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                        response=step_by_step_response[0]
                    )

                # Add supervisor synthesis prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Get supervisor's response
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
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

            # Parse supervisor's decision
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                try:
                    supervisor_json = json.loads(supervisor_response[0])
                except:
                    supervisor_json = _extract_json(supervisor_response[0])
            else:
                supervisor_json = _extract_json(supervisor_response[0])

            if supervisor_json["decision"] == "provide_final_answer":
                final_answer = supervisor_json["answer"]
                conversation_log["generated_final_answer"] = final_answer
                break
            else:
                next_question = supervisor_json["message"]
                worker_messages.append({"role": "user", "content": next_question})

                # Add next worker prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "local", "prompt": next_question, "output": None}
                )

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Update conversation history if in multi-turn mode
        if self.is_multi_turn and final_answer:
            # Add this turn to conversation history
            turn = ConversationTurn(
                query=task,
                local_output=local_output,
                remote_output=final_answer,
                timestamp=datetime.now()
            )
            self.conversation_history.add_turn(turn, remote_client=self.remote_client)

        # Log the final result
        if logging_id:
            # use provided logging_id
            log_filename = f"{logging_id}_minion.json"
        else:
            # fall back to timestamp + task abbrev
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
        }
        
    def _format_conversation_history(self) -> str:
        """Format the conversation history for inclusion in prompts."""
        if not self.conversation_history or not self.conversation_history.turns:
            return "No previous conversation."
            
        formatted_history = ""
        
        # Include summary if it exists (from older conversation turns)
        if hasattr(self.conversation_history, 'summary') and self.conversation_history.summary:
            formatted_history += f"### Summary of Earlier Conversation\n{self.conversation_history.summary}\n\n### Recent Conversation\n"
        
        # Add recent conversation turns
        for i, turn in enumerate(self.conversation_history.turns):
            formatted_history += MULTI_TURN_CONVERSATION_HISTORY_FORMAT.format(
                query_index=i+1,
                query=turn.query,
                response=turn.remote_output
            )
        
        return formatted_history
