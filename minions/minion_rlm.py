"""
Recursive LM (RLM) Protocol implementation.

Based on "Recursive Language Models" (arxiv 2512.24601) by Zhang, Kraska, and Khattab.

The remote model writes Python code to decompose tasks, chunk context, and orchestrate
processing via a persistent REPL environment. The local model handles llm_query() sub-calls
to process individual context chunks.
"""

import io
import json
import os
import re
import signal
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from minions.usage import Usage
from minions.prompts.minion_rlm import (
    RLM_SYSTEM_PROMPT,
    RLM_CONTINUATION_PROMPT,
    RLM_ERROR_PROMPT,
)


def _extract_text(response_item):
    """Extract text from a client response item which may be a str or a dict."""
    if isinstance(response_item, str):
        return response_item
    if isinstance(response_item, dict):
        return response_item.get("message", response_item.get("content", str(response_item)))
    return str(response_item)


class _FinalAnswerSignal(Exception):
    """Raised when FINAL() or FINAL_VAR() is called to stop execution."""

    def __init__(self, answer):
        self.answer = answer


class _CodeExecutionTimeout(Exception):
    """Raised when a code block exceeds its time limit."""
    pass


def _timeout_handler(signum, frame):
    raise _CodeExecutionTimeout("Code execution timed out (300s limit)")


class MinionRLM:
    """
    Recursive LM protocol: the remote model writes Python code in a REPL loop,
    using llm_query() to delegate context processing to the local model.
    """

    # Builtins allowed in the sandbox
    SAFE_BUILTINS = {
        "len": len,
        "range": range,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "isinstance": isinstance,
        "type": type,
        "set": set,
        "tuple": tuple,
        "bool": bool,
        "any": any,
        "all": all,
        "reversed": reversed,
        "print": print,  # will be overridden per-execution
        "repr": repr,
        "hash": hash,
        "chr": chr,
        "ord": ord,
        "hex": hex,
        "bin": bin,
        "oct": oct,
        "divmod": divmod,
        "pow": pow,
        "slice": slice,
        "format": format,
        "id": id,
        "callable": callable,
        "iter": iter,
        "next": next,
        "super": super,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "property": property,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "delattr": delattr,
        "True": True,
        "False": False,
        "None": None,
    }

    # Modules allowed in the sandbox
    SAFE_MODULES = ["re", "json", "math", "collections", "itertools", "functools"]

    def __init__(
        self,
        remote_client,
        local_client,
        max_iterations: int = 15,
        max_chars_per_query: int = 500_000,
        stdout_prefix_chars: int = 3000,
        callback=None,
        log_dir: str = "minion_rlm_logs",
    ):
        """
        Initialize MinionRLM.

        Args:
            remote_client: The powerful remote model client (controller/code writer).
            local_client: The local model client (handles llm_query sub-calls).
            max_iterations: Maximum REPL iterations before forcing termination.
            max_chars_per_query: Maximum characters per llm_query context_chunk.
            stdout_prefix_chars: How many chars of stdout to show to the remote model.
            callback: Optional callback(role, message, is_final) for progress updates.
            log_dir: Directory for conversation logs.
        """
        self.remote_client = remote_client
        self.local_client = local_client
        self.max_iterations = max_iterations
        self.max_chars_per_query = max_chars_per_query
        self.stdout_prefix_chars = stdout_prefix_chars
        self.callback = callback
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds: Optional[int] = None,
        doc_metadata: Optional[str] = None,
        logging_id: Optional[str] = None,
        images=None,
    ) -> Dict[str, Any]:
        """
        Run the RLM protocol.

        Args:
            task: The task/question to answer.
            context: List of context strings (will be joined).
            max_rounds: Override max_iterations if provided.
            doc_metadata: Optional metadata description of the documents.
            logging_id: Optional identifier for the log file.
            images: Unused (kept for interface compatibility).

        Returns:
            Dict with final_answer, remote_usage, local_usage, log_file,
            conversation_log, and timing.
        """
        print("\n========== RLM TASK STARTED ==========")
        print(f"Task: {task}")

        start_time = time.time()
        timing = {
            "local_call_time": 0.0,
            "remote_call_time": 0.0,
            "total_time": 0.0,
        }

        max_iterations = max_rounds if max_rounds is not None else self.max_iterations

        # Join context
        full_context = "\n\n".join(context)
        print(f"Context length: {len(full_context)} characters")
        print(f"Max iterations: {max_iterations}")

        # Usage tracking
        remote_usage = Usage()
        local_usage = Usage()

        # REPL state
        llm_query_count = 0
        namespace = self._create_namespace(full_context)
        final_answer = None

        # Conversation log
        conversation_log = {
            "task": task,
            "context_length": len(full_context),
            "doc_metadata": doc_metadata,
            "iterations": [],
            "final_answer": None,
            "usage": {"remote": {}, "local": {}},
            "timing": timing,
        }

        # Build initial system prompt
        context_preview = f"Preview (first 500 chars):\n```\n{full_context[:500]}\n```"
        doc_metadata_section = (
            f"## Document Metadata\n{doc_metadata}" if doc_metadata else ""
        )

        system_prompt = RLM_SYSTEM_PROMPT.format(
            task=task,
            context_length=len(full_context),
            context_preview=context_preview,
            doc_metadata_section=doc_metadata_section,
        )

        remote_messages = [{"role": "user", "content": system_prompt}]

        # ---- Main REPL loop ----
        for iteration in range(1, max_iterations + 1):
            print(f"\n--- RLM Iteration {iteration}/{max_iterations} ---")

            if self.callback:
                self.callback(
                    "supervisor",
                    f"RLM iteration {iteration}/{max_iterations}: generating code...",
                    is_final=False,
                )

            # Call remote model to get code
            remote_start = time.time()
            response, usage = self.remote_client.chat(messages=remote_messages)
            remote_elapsed = time.time() - remote_start
            timing["remote_call_time"] += remote_elapsed
            remote_usage += usage

            assistant_text = _extract_text(response[0])
            remote_messages.append({"role": "assistant", "content": assistant_text})

            print(f"Remote response length: {len(assistant_text)} chars")

            # Extract code blocks
            code_blocks = self._extract_code_blocks(assistant_text)

            if not code_blocks:
                print("No code blocks found in response.")
                # Check if the model tried to give a final answer in text
                # (sometimes models skip the FINAL() call)
                iteration_log = {
                    "iteration": iteration,
                    "remote_response": assistant_text,
                    "code_blocks": [],
                    "stdout": "",
                    "error": "No code blocks found",
                    "variables": {},
                }
                conversation_log["iterations"].append(iteration_log)

                error_prompt = RLM_ERROR_PROMPT.format(
                    iteration=iteration,
                    max_iterations=max_iterations,
                    error_message="No ```repl``` or ```python``` code block found in your response. Please write code in a ```repl``` block.",
                    variables_info=self._format_variables(namespace),
                    llm_query_count=llm_query_count,
                    local_tokens_used=local_usage.total_tokens,
                )
                remote_messages.append({"role": "user", "content": error_prompt})
                continue

            # Execute code blocks
            stdout_capture = io.StringIO()
            execution_error = None

            def _llm_query(prompt: str, context_chunk: str = "") -> str:
                nonlocal llm_query_count, local_usage
                llm_query_count += 1

                if self.callback:
                    self.callback(
                        "worker",
                        f"llm_query() call #{llm_query_count}",
                        is_final=False,
                    )

                # Truncate context_chunk if too long
                if len(context_chunk) > self.max_chars_per_query:
                    context_chunk = context_chunk[: self.max_chars_per_query]
                    print(
                        f"[Warning] context_chunk truncated to {self.max_chars_per_query} chars",
                        file=stdout_capture,
                    )

                # Build messages for local client
                if context_chunk:
                    local_content = f"Context:\n{context_chunk}\n\nQuestion/Instruction:\n{prompt}"
                else:
                    local_content = prompt

                messages = [{"role": "user", "content": local_content}]

                local_start = time.time()
                try:
                    result = self.local_client.chat(messages=messages)
                    # Handle both 2-tuple and 3-tuple returns
                    local_response = result[0]
                    local_usage_delta = result[1]
                except Exception as e:
                    timing["local_call_time"] += time.time() - local_start
                    return f"[Error in llm_query: {e}]"

                timing["local_call_time"] += time.time() - local_start
                local_usage += local_usage_delta

                text = local_response[0] if isinstance(local_response, list) else local_response
                return _extract_text(text)

            def _final(answer):
                raise _FinalAnswerSignal(str(answer))

            def _final_var(var_name: str):
                if var_name in namespace:
                    raise _FinalAnswerSignal(str(namespace[var_name]))
                else:
                    raise NameError(
                        f"Variable '{var_name}' not found in scope. Available: {list(k for k in namespace if not k.startswith('__'))}"
                    )

            # Inject functions into namespace
            namespace["llm_query"] = _llm_query
            namespace["FINAL"] = _final
            namespace["FINAL_VAR"] = _final_var

            # Override print to capture stdout
            def _captured_print(*args, **kwargs):
                kwargs["file"] = stdout_capture
                print(*args, **kwargs)

            namespace["__builtins__"]["print"] = _captured_print

            combined_code = "\n\n".join(code_blocks)
            print(f"Executing {len(code_blocks)} code block(s) ({len(combined_code)} chars)")

            try:
                # Set up timeout (Unix only; graceful no-op on other platforms)
                old_handler = None
                try:
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(300)  # 5 minute timeout
                except (AttributeError, OSError):
                    pass  # signal.SIGALRM not available (e.g., Windows)

                exec(compile(combined_code, "<rlm_repl>", "exec"), namespace)

            except _FinalAnswerSignal as fas:
                final_answer = fas.answer
                print(f"FINAL answer received: {final_answer[:200]}...")
            except _CodeExecutionTimeout:
                execution_error = "Code execution timed out (300s limit). Try a more efficient approach."
                print(f"Timeout error")
            except Exception as e:
                execution_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"Execution error: {type(e).__name__}: {e}")
            finally:
                # Cancel timeout
                try:
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
                except (AttributeError, OSError, NameError):
                    pass

            stdout_content = stdout_capture.getvalue()

            # Log this iteration
            iteration_log = {
                "iteration": iteration,
                "remote_response": assistant_text,
                "code_blocks": code_blocks,
                "stdout": stdout_content,
                "error": execution_error,
                "final_answer": final_answer,
                "llm_query_count": llm_query_count,
                "local_tokens": local_usage.total_tokens,
                "variables": self._format_variables_for_log(namespace),
            }
            conversation_log["iterations"].append(iteration_log)

            if self.callback:
                self.callback(
                    "worker",
                    f"Iteration {iteration} complete. llm_query calls: {llm_query_count}",
                    is_final=False,
                )

            # If we got a final answer, we're done
            if final_answer is not None:
                break

            # Build next prompt for remote model
            variables_info = self._format_variables(namespace)

            if execution_error:
                next_prompt = RLM_ERROR_PROMPT.format(
                    iteration=iteration,
                    max_iterations=max_iterations,
                    error_message=execution_error,
                    variables_info=variables_info,
                    llm_query_count=llm_query_count,
                    local_tokens_used=local_usage.total_tokens,
                )
            else:
                # Truncate stdout for the remote model
                stdout_truncated_note = ""
                stdout_display = stdout_content
                if len(stdout_content) > self.stdout_prefix_chars:
                    stdout_display = stdout_content[: self.stdout_prefix_chars]
                    stdout_truncated_note = f", showing first {self.stdout_prefix_chars} of {len(stdout_content)}"

                next_prompt = RLM_CONTINUATION_PROMPT.format(
                    iteration=iteration,
                    max_iterations=max_iterations,
                    variables_info=variables_info,
                    stdout_length=len(stdout_content),
                    stdout_truncated_note=stdout_truncated_note,
                    stdout_content=stdout_display,
                    llm_query_count=llm_query_count,
                    local_tokens_used=local_usage.total_tokens,
                )

            remote_messages.append({"role": "user", "content": next_prompt})

        # If no final answer after all iterations, try to extract from last stdout
        if final_answer is None:
            print("No FINAL() called. Attempting to extract answer from last output...")
            # Ask the remote model one more time for a final answer
            force_prompt = (
                "You have run out of iterations. Based on all the work done so far, "
                "please provide your best final answer by calling FINAL(answer) in a ```repl``` block."
            )
            remote_messages.append({"role": "user", "content": force_prompt})

            remote_start = time.time()
            response, usage = self.remote_client.chat(messages=remote_messages)
            timing["remote_call_time"] += time.time() - remote_start
            remote_usage += usage

            # Try to extract FINAL() call from the response
            final_text = _extract_text(response[0])
            code_blocks = self._extract_code_blocks(final_text)
            if code_blocks:
                combined = "\n\n".join(code_blocks)
                try:
                    exec(compile(combined, "<rlm_repl_final>", "exec"), namespace)
                except _FinalAnswerSignal as fas:
                    final_answer = fas.answer
                except Exception:
                    pass

            if final_answer is None:
                # Last resort: use the raw text
                final_answer = final_text

        conversation_log["final_answer"] = final_answer

        # Finalize timing
        end_time = time.time()
        timing["total_time"] = end_time - start_time
        timing["overhead_time"] = timing["total_time"] - (
            timing["local_call_time"] + timing["remote_call_time"]
        )

        # Usage
        conversation_log["usage"]["remote"] = remote_usage.to_dict()
        conversation_log["usage"]["local"] = local_usage.to_dict()

        # Save log
        if logging_id:
            log_filename = f"{logging_id}_rlm.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        print(f"\n=== SAVING LOG TO {log_path} ===")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving log: {e}")

        if self.callback:
            self.callback(
                "supervisor",
                {"role": "assistant", "content": final_answer},
                is_final=True,
            )

        print("\n========== RLM TASK COMPLETED ==========")
        print(f"Total iterations: {len(conversation_log['iterations'])}")
        print(f"Total llm_query calls: {llm_query_count}")
        print(f"Total time: {timing['total_time']:.1f}s")

        return {
            "final_answer": final_answer,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
            "conversation_log": conversation_log,
            "timing": timing,
        }

    def _create_namespace(self, full_context: str) -> dict:
        """Create a sandboxed namespace for code execution."""
        import collections
        import functools
        import itertools
        import math

        namespace = {
            "__builtins__": dict(self.SAFE_BUILTINS),
            "context": full_context,
            # Safe modules
            "re": re,
            "json": json,
            "math": math,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
        }

        return namespace

    @staticmethod
    def _extract_code_blocks(text: str) -> List[str]:
        """Extract code from ```repl``` and ```python``` fenced blocks."""
        blocks = []
        # Match ```repl, ```python, or bare ``` blocks
        pattern = r"```(?:repl|python)?\s*\n(.*?)```"
        for match in re.finditer(pattern, text, re.DOTALL):
            code = match.group(1).strip()
            if code:
                blocks.append(code)
        return blocks

    @staticmethod
    def _format_variables(namespace: dict) -> str:
        """Format namespace variables for display to the remote model."""
        skip = {
            "__builtins__",
            "re",
            "json",
            "math",
            "collections",
            "itertools",
            "functools",
            "llm_query",
            "FINAL",
            "FINAL_VAR",
        }
        lines = []
        for name, value in sorted(namespace.items()):
            if name in skip or name.startswith("_"):
                continue
            try:
                vtype = type(value).__name__
                if isinstance(value, str):
                    lines.append(
                        f"- `{name}` (str, {len(value)} chars): {repr(value[:100])}{'...' if len(value) > 100 else ''}"
                    )
                elif isinstance(value, (list, tuple)):
                    lines.append(
                        f"- `{name}` ({vtype}, {len(value)} items): {repr(value[:3])}{'...' if len(value) > 3 else ''}"
                    )
                elif isinstance(value, dict):
                    keys = list(value.keys())[:5]
                    lines.append(
                        f"- `{name}` (dict, {len(value)} keys): keys={keys}{'...' if len(value) > 5 else ''}"
                    )
                elif isinstance(value, (int, float, bool)):
                    lines.append(f"- `{name}` ({vtype}): {value}")
                else:
                    lines.append(f"- `{name}` ({vtype})")
            except Exception:
                lines.append(f"- `{name}` (unknown)")
        return "\n".join(lines) if lines else "(no user variables)"

    @staticmethod
    def _format_variables_for_log(namespace: dict) -> dict:
        """Format namespace variables for JSON logging."""
        skip = {
            "__builtins__",
            "re",
            "json",
            "math",
            "collections",
            "itertools",
            "functools",
            "llm_query",
            "FINAL",
            "FINAL_VAR",
            "context",
        }
        result = {}
        for name, value in namespace.items():
            if name in skip or name.startswith("_"):
                continue
            try:
                if isinstance(value, str):
                    result[name] = {"type": "str", "length": len(value), "preview": value[:200]}
                elif isinstance(value, (int, float, bool)):
                    result[name] = {"type": type(value).__name__, "value": value}
                elif isinstance(value, (list, tuple)):
                    result[name] = {"type": type(value).__name__, "length": len(value)}
                elif isinstance(value, dict):
                    result[name] = {"type": "dict", "length": len(value)}
                else:
                    result[name] = {"type": type(value).__name__}
            except Exception:
                result[name] = {"type": "unknown"}
        return result
