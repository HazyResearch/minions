from typing import List, Dict, Any
import json
import re
import os
from datetime import datetime

import mcp

from minions.clients import OpenAIClient, TogetherClient
from minions.minions_mcp import SyncMCPClient

from minions.prompts.minion_mcp import (
    SUPERVISOR_INITIAL_PROMPT_MCP,
    REMOTE_SYNTHESIS_FINAL_MCP,
    WORKER_SYSTEM_PROMPT_MCP,
)
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
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
        mcp_client: SyncMCPClient | None = None,
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
        self.mcp_client = mcp_client

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Set up prompts based on whether we have access to an MCP client
        self.supervisor_initial_prompt = SUPERVISOR_INITIAL_PROMPT
        self.remote_synthesis_final = REMOTE_SYNTHESIS_FINAL
        self.worker_system_prompt = WORKER_SYSTEM_PROMPT

        if mcp_client is not None:
            self.supervisor_initial_prompt = SUPERVISOR_INITIAL_PROMPT_MCP
            self.remote_synthesis_final = REMOTE_SYNTHESIS_FINAL_MCP
            self.worker_system_prompt = WORKER_SYSTEM_PROMPT_MCP

        self.mcp_tools_info = None if self.mcp_client is None else self._generate_mcp_tools_info()

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,  # this is the name/id to give to the logging .json file
        is_privacy=False,
        images=None,
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

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "remote",
                "prompt": self.supervisor_initial_prompt.format(task=task, mcp_tools_info=self.mcp_tools_info),
                "output": None,
            }
        )

        # print whether privacy is enabled
        print("Privacy is enabled: ", is_privacy)

        remote_usage = Usage()
        local_usage = Usage()

        worker_messages = []
        supervisor_messages = []

        # if privacy import from minions.utils.pii_extraction
        if is_privacy:
            from minions.utils.pii_extraction import PIIExtractor

            # Extract PII from context
            pii_extractor = PIIExtractor()
            str_context = "\n\n".join(context)
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

            # Initialize message histories
            supervisor_messages = [
                {
                    "role": "user",
                    "content": self.supervisor_initial_prompt.format(
                        task=pii_reformatted_task,
                        mcp_tools_info=self.mcp_tools_info,
                    ),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": self.worker_system_prompt.format(context=context, task=task),
                }
            ]
        else:
            supervisor_messages = [
                {
                    "role": "user",
                    "content": self.supervisor_initial_prompt.format(task=task, mcp_tools_info=self.mcp_tools_info),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": self.worker_system_prompt.format(context=context, task=task),
                    "images": images,
                }
            ]

        if max_rounds is None:
            max_rounds = self.max_rounds

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
            {"user": "local", "prompt": supervisor_json["message"], "output": None,
             "mcp_tool_calls": supervisor_json["mcp_tool_calls"], "mcp_tool_outputs": []}
        )

        final_answer = None
        for round in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)

            # If the supervisor has specified MCP tools to be called, then we call them
            if self.mcp_client is not None and len(supervisor_json["mcp_tool_calls"]) > 0:
                context_to_prepend = f"**New context from MCP tool calls:\n**"
                for i, tool_call in enumerate(supervisor_json["mcp_tool_calls"], start=1):
                    # Here, we assume the supervisor has stuck to the format in self._generate_mcp_tools_info()
                    tool_name = tool_call["tool_name"]
                    tool_params = tool_call["parameters"]

                    print(f"About to call MCP tool '{tool_name}' with params {tool_params}")
                    if input("OK? (Type 'y' for yes or  'n' to skip): ").lower() != "y":
                        conversation_log["conversation"][-1]["mcp_tool_outputs"].append("<skipped>")
                        continue

                    try:
                        mcp_output = self.mcp_client.execute_tool(tool_name=tool_name, **tool_params)
                    except mcp.McpError as e:
                        mcp_output = f"MCP Error: {e}"
                    conversation_log["conversation"][-1]["mcp_tool_outputs"].append(str(mcp_output))

                    # Provide MCP output to local client
                    context_to_prepend += f"_MCP tool call {i}:_\n"
                    context_to_prepend += f"Tool name: {tool_name}\n"
                    context_to_prepend += f"Tool params: {tool_params}\n"
                    context_to_prepend += f"Tool output:\n{mcp_output}\n"

                context_to_prepend += "Please use the output above to help me with the following. "
                context_to_prepend += "Remember, I can't see the output above.\n"
                worker_messages[-1]["content"] = context_to_prepend + worker_messages[-1]["content"]
                conversation_log["conversation"][-1]["prompt"] = worker_messages[-1]["content"]

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

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )
            else:
                # First step: Think through the synthesis
                cot_prompt = REMOTE_SYNTHESIS_COT.format(response=worker_response[0])

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
                    {"user": "local", "prompt": next_question, "output": None,
                     "mcp_tool_calls": supervisor_json["mcp_tool_calls"], "mcp_tool_outputs": []}
                )

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

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

    def _generate_mcp_tools_info(self):
        """Generate explanation of available MCP tools for supervisor."""
        mcp_tools_info = "### Available MCP Tools\n\n"
        for tool in self.mcp_client.available_tools:
            mcp_tools_info += f"## {tool['name']}\n\n"
            mcp_tools_info += f"**Description**: {tool['description']}\n\n"

            # Create parameter list from schema
            params = []
            if "properties" in tool["input_schema"]:
                for param_name in tool["input_schema"]["properties"].keys():
                    params.append(param_name)

            mcp_tools_info += f"""**JSON Format**:
```json
{{
    "tool_name": "{tool['name']}",
    "parameters": {{{', '.join([f'"{p}": "<{p}_value>"' for p in params])}}}
}}
```
"""
        return mcp_tools_info
