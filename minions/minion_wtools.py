from typing import List, Dict, Any
import json
import re
import os
from datetime import datetime

# Import Minion + Gru Clients
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients import OpenAIClient, TogetherClient


# Import Pydantic
from pydantic import BaseModel

# Import Minion Prompts
from minions.prompts.minion import (
    SUPERVISOR_FINAL_PROMPT,
)

from minions.prompts.minion_wtools import (
    REMOTE_SYNTHESIS_FINAL_TOOLCALLING,
    REMOTE_SYNTHESIS_COT_TOOLCALLING,
    SUPERVISOR_INITIAL_PROMPT_TOOLCALLING,
    WORKER_SYSTEM_PROMPT_TOOLCALLING,
)

# Import Misc
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


#### BASE TOOLS DESCRIPTION AND ARGUMENTS ####

TOOL_DESCRIPTION = """\
list_directory:
Description: Lists directory contents with [FILE] or [DIR] prefixes
Arguments: path (optional) - Directory to list, defaults to current directory

read_file:
Description: Reads the contents of a file as text (with binary fallback)
Arguments: file_path (required) - Path to the file to read

read_multiple_files:
Description: Reads the contents of multiple files at once (use this when you want to read or compare multiple files)
Arguments: file_paths (required) - List of paths to the files to read; this should NOT BE A STRING, but a list of strings

write_file:
Description: Creates new file or overwrites existing file
Arguments: file_path (required) - Path to the file to write
           content (required) - Content to write to the file

create_directory:
Description: Creates a new directory or ensures it exists (creates parent dirs if needed)
Arguments: path (required) - Path to the directory to create

search_files:
Description: Recursively searches for files/directories matching a pattern
Arguments: path (required) - Starting directory for search
           pattern (required) - Search pattern (glob format)
           exclude_patterns (optional) - Patterns to exclude (glob format)
"""

#### BASE TOOL DEFINITIONS ####

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List directory contents with [FILE] or [DIR] prefixes",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory to list (defaults to current directory)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the contents of multiple files at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "description": "List of paths to the files to read; this should NOT BE A STRING, but a list of strings",
                    },
                },
                "required": ["file_paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create new file or overwrite existing file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a new directory or ensure it exists",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory to create",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

#### BASE TOOL IMPLEMENTATIONS ####


# Improved file operation functions
def list_directory(path=None):
    """List directory contents with [FILE] or [DIR] prefixes.

    Args:
        path: Path to directory (defaults to current directory)

    Returns:
        List of files and directories with type indicators
    """
    import os

    if path is None:
        path = os.getcwd()

    try:
        items = os.listdir(path)
        result = []

        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR] {item}")
            else:
                result.append(f"[FILE] {item}")

        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def read_file(file_path):
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        Contents of the file as a string
    """
    import os

    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        # Handle PDF files
        if ext == ".pdf":
            try:
                import PyPDF2

                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page.extract_text()
                    return text
            except ImportError:
                return "Error: PyPDF2 library not installed. Install with 'pip install PyPDF2' to read PDF files."
            except Exception as e:
                return f"Error reading PDF file: {str(e)}"

        # Handle text files
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try binary mode if UTF-8 fails
        try:
            with open(file_path, "rb") as f:
                return f"Binary file, first 1000 bytes: {f.read(1000)}"
        except Exception as e:
            return f"Error reading file as binary: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path, content):
    """Create new file or overwrite existing file.

    Args:
        file_path: Path to the file to write
        content: Content to write to the file

    Returns:
        Success message or error
    """
    import os

    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def create_directory(path):
    """Create a new directory or ensure it exists.

    Args:
        path: Path to the directory to create

    Returns:
        Success message or error
    """
    import os

    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory created or already exists: {path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"


def get_file_info(path):
    """Get detailed file/directory metadata.

    Args:
        path: Path to the file or directory

    Returns:
        Dictionary with file metadata
    """
    import os
    import time

    try:
        stat_info = os.stat(path)

        file_type = "directory" if os.path.isdir(path) else "file"

        info = {
            "path": path,
            "type": file_type,
            "size_bytes": stat_info.st_size,
            "created_time": time.ctime(stat_info.st_ctime),
            "modified_time": time.ctime(stat_info.st_mtime),
            "accessed_time": time.ctime(stat_info.st_atime),
            "permissions": oct(stat_info.st_mode)[-3:],
            "exists": os.path.exists(path),
        }

        return info
    except Exception as e:
        return f"Error getting file info: {str(e)}"


def search_files(
    path,
    pattern,
    exclude_patterns=None,
):
    """Recursively search for files/directories.

    Args:
        path: Starting directory for search
        pattern: Search pattern (glob format)
        exclude_patterns: Patterns to exclude (glob format)

    Returns:
        List of matching files/directories
    """
    import os
    import fnmatch
    import re

    if exclude_patterns is None:
        exclude_patterns = []

    # Convert glob patterns to regex patterns
    regex_pattern = fnmatch.translate(pattern)
    regex_excludes = [fnmatch.translate(p) for p in exclude_patterns]

    matches = []

    try:
        for root, dirnames, filenames in os.walk(path):
            # Check if root should be excluded
            if any(re.match(ex, root) for ex in regex_excludes):
                continue

            # Check directories
            for dirname in dirnames:
                full_path = os.path.join(root, dirname)
                if re.match(regex_pattern, dirname, re.IGNORECASE) and not any(
                    re.match(ex, full_path) for ex in regex_excludes
                ):
                    matches.append(f"{full_path}")

            # Check files
            for filename in filenames:
                full_path = os.path.join(root, filename)
                if re.match(regex_pattern, filename, re.IGNORECASE) and not any(
                    re.match(ex, full_path) for ex in regex_excludes
                ):
                    matches.append(f"{full_path}")

        return matches
    except Exception as e:
        return f"Error searching files: {str(e)}"


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


def read_multiple_files(file_paths):
    """Read the contents of multiple files at once.

    Args:
        file_paths: List of paths to the files to read

    Returns:
        Dictionary mapping file paths to their contents
    """
    import os

    results = {}

    for file_path in file_paths:
        full_path = file_path
        try:
            # Get file extension
            _, ext = os.path.splitext(full_path)
            ext = ext.lower()

            # Handle PDF files
            if ext == ".pdf":
                try:
                    import PyPDF2

                    with open(full_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page.extract_text()
                        results[file_path] = text
                except ImportError:
                    results[file_path] = (
                        "Error: PyPDF2 library not installed. Install with 'pip install PyPDF2' to read PDF files."
                    )
                except Exception as e:
                    results[file_path] = f"Error reading PDF file: {str(e)}"

            # Handle text files
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    results[file_path] = f.read()

        except UnicodeDecodeError:
            # Try binary mode if UTF-8 fails
            try:
                with open(full_path, "rb") as f:
                    results[file_path] = (
                        f"Binary file, first 1000 bytes: {f.read(1000)}"
                    )
            except Exception as e:
                results[file_path] = f"Error reading file as binary: {str(e)}"

        except Exception as e:
            results[file_path] = f"Error reading file: {str(e)}"

    return results


# Update execute_tool_calls function to handle the new tools
def execute_tool_calls(tool_calls):
    """Execute tool calls and return the results.

    Args:
        tool_calls: List of tool call objects from the LLM

    Returns:
        List of dictionaries with tool call results
    """
    print("\n=== EXECUTING TOOL CALLS ===")
    results = []

    if not tool_calls:
        print("No tool calls to execute")
        return results

    for tool_call in tool_calls[0]:
        print(f"Tool call: {tool_call}")
        function_name = tool_call.function.name
        print(f"Function name: {function_name}")

        # Parse arguments
        try:
            arguments = json.loads(str(tool_call.function.arguments))
        except:
            arguments = tool_call.function.arguments or {}

        print(f"Function arguments: {arguments}")

        result = None
        try:
            if function_name == "list_directory":
                result = list_directory(arguments.get("path"))
            elif function_name == "read_file":
                if "file_path" in arguments:
                    result = read_file(arguments["file_path"])
                else:
                    result = "Error: file_path argument is required"
            elif function_name == "read_multiple_files":
                if "file_paths" in arguments:
                    if isinstance(arguments["file_paths"], str):
                        file_paths = json.loads(arguments["file_paths"])
                    else:
                        file_paths = arguments["file_paths"]
                    result = read_multiple_files(file_paths)
                else:
                    result = "Error: file_paths argument is required"
            elif function_name == "write_file":
                if "file_path" in arguments and "content" in arguments:
                    result = write_file(arguments["file_path"], arguments["content"])
                else:
                    result = "Error: file_path and content arguments are required"
            elif function_name == "create_directory":
                if "path" in arguments:
                    result = create_directory(arguments["path"])
                else:
                    result = "Error: path argument is required"
            elif function_name == "get_file_info":
                if "path" in arguments:
                    result = get_file_info(arguments["path"])
                else:
                    result = "Error: path argument is required"
            elif function_name == "search_files":
                if "path" in arguments and "pattern" in arguments:
                    result = search_files(
                        arguments["path"],
                        arguments["pattern"],
                        arguments.get("exclude_patterns", []),
                    )
                else:
                    result = "Error: path and pattern arguments are required"
            else:
                result = f"Error: Unknown function {function_name}"
        except Exception as e:
            result = f"Error executing {function_name}: {str(e)}"

        print(
            f"Tool result: {result[:100]}..."
            if isinstance(result, str) and len(result) > 100
            else f"Tool result: {result}"
        )

        results.append(
            {
                "function_name": function_name,
                "result": result,
            }
        )

    print("=== TOOL EXECUTION COMPLETE ===\n")
    return results


class MinionToolCalling:
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
        custom_tools=None,  # New parameter for custom tools
        custom_tool_executors=None,  # New parameter for custom tool executors
        custom_tool_descriptions=None,  # New parameter for custom tool descriptions
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
            log_dir: Directory to store logs
            custom_tools: Optional list of additional tool definitions
            custom_tool_executors: Optional dictionary mapping function names to executable functions
            custom_tool_descriptions: Optional string with descriptions of custom tools
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        self.custom_tools = custom_tools or []
        self.custom_tool_executors = custom_tool_executors or {}
        self.custom_tool_descriptions = custom_tool_descriptions or ""
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,  # this is the name/id to give to the logging .json file
        is_privacy=False,
        images=None,
        custom_tools=None,  # Allow overriding custom tools at call time
        custom_tool_executors=None,  # Allow overriding custom tool executors at call time
        custom_tool_descriptions=None,  # Allow overriding custom tool descriptions at call time
    ):
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files
            is_privacy: Whether to enable privacy mode
            images: Optional list of image URLs or base64 encoded images
            custom_tools: Optional list of additional tool definitions to use for this call
            custom_tool_executors: Optional dictionary mapping function names to executable functions for this call
            custom_tool_descriptions: Optional string with descriptions of custom tools for this call

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """
        # Use call-specific custom tools/executors if provided, otherwise use instance defaults
        active_custom_tools = (
            custom_tools if custom_tools is not None else self.custom_tools
        )
        active_custom_tool_executors = (
            custom_tool_executors
            if custom_tool_executors is not None
            else self.custom_tool_executors
        )
        active_custom_tool_descriptions = (
            custom_tool_descriptions
            if custom_tool_descriptions is not None
            else self.custom_tool_descriptions
        )

        # Combine default tools with custom tools
        combined_tools = TOOLS + active_custom_tools

        # Combine default tool descriptions with custom tool descriptions
        combined_tool_descriptions = TOOL_DESCRIPTION
        if active_custom_tool_descriptions:
            combined_tool_descriptions += "\n\n" + active_custom_tool_descriptions

        print("\n========== MINION TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Max rounds: {max_rounds or self.max_rounds}")
        print(f"Privacy enabled: {is_privacy}")
        print(f"Images provided: {True if images else False}")
        print(f"Custom tools provided: {len(active_custom_tools) > 0}")
        print(
            f"Custom tool descriptions provided: {len(active_custom_tool_descriptions) > 0}"
        )

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Join context sections
        context = "\n\n".join(context)
        print(f"Context length: {len(context)} characters")

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": context,
            "conversation": [],
            "generated_final_answer": "",
        }

        # Initialize message histories and usage tracking
        remote_usage = Usage()
        local_usage = Usage()

        worker_messages = []
        supervisor_messages = []

        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_INITIAL_PROMPT_TOOLCALLING.format(
                    task=task, TOOL_DESCRIPTION=combined_tool_descriptions
                ),
            }
        ]
        worker_messages = [
            {
                "role": "system",
                "content": WORKER_SYSTEM_PROMPT_TOOLCALLING.format(
                    context=context,
                    task=task,
                    TOOL_DESCRIPTION=combined_tool_descriptions,
                ),
                "images": images,
            }
        ]

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Initial supervisor call to get first question
        print("\n=== SUPERVISOR INITIAL CALL ===")
        print(
            f"Prompt: {SUPERVISOR_INITIAL_PROMPT_TOOLCALLING.format(task=task, TOOL_DESCRIPTION=combined_tool_descriptions)[:100]}..."
        )

        if self.callback:
            self.callback("supervisor", None, is_final=False)

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "remote",
                "prompt": SUPERVISOR_INITIAL_PROMPT_TOOLCALLING.format(
                    task=task, TOOL_DESCRIPTION=combined_tool_descriptions
                ),
                "output": None,  # Will be filled after response
            }
        )

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages, response_format={"type": "json_object"}
            )
        else:
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages
            )

        print(f"Supervisor response: {supervisor_response[0]}...")

        remote_usage += supervisor_usage
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        # Update the conversation log with the supervisor's response
        conversation_log["conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Extract first question for worker
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            try:
                supervisor_json = json.loads(supervisor_response[0])
                print("Successfully parsed supervisor JSON directly")
            except:
                supervisor_json = _extract_json(supervisor_response[0])
                print("Used _extract_json to parse supervisor response")
        else:
            supervisor_json = _extract_json(supervisor_response[0])
            print("Used _extract_json to parse supervisor response")

        print(f"First question for worker: {supervisor_json['message']}...")
        worker_messages.append({"role": "user", "content": supervisor_json["message"]})

        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "local",
                "prompt": supervisor_json["message"],
                "output": None,  # Will be filled after response
            }
        )

        final_answer = None
        for round in range(max_rounds):
            print(f"\n=== ROUND {round+1}/{max_rounds} ===")

            # Get worker's response
            print("\n--- WORKER THINKING ---")
            print(f"Worker prompt: {worker_messages[-1]['content'][:200]}...")

            if self.callback:
                self.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason, tool_calls = (
                self.local_client.chat(
                    messages=worker_messages,
                    tools=combined_tools,
                )
            )

            print(f"Worker response: {worker_response[0]}...")
            print(f"Done reason: {done_reason}")
            print(f"Tool calls: {True if tool_calls else False}")

            # Update the worker's response in the conversation log
            conversation_log["conversation"][-1]["output"] = worker_response[0]

            # Process tool calls if any
            if tool_calls:
                tool_results = self._execute_tool_calls(
                    tool_calls, active_custom_tool_executors
                )

                # Log the tool calls and results
                print(f"Tool calls: {tool_calls}")
                print(f"Tool results: {tool_results}")

                # Add tool results as a tool response message
                for i, result in enumerate(tool_results):
                    tool_message = f"Context:\n\nYou made a call to this tool: {result['function_name']}\n\nThis is what the tool returned: {json.dumps(result['result'], ensure_ascii=False)}\n\nRecall that your task is to: {worker_messages[-1]['content']}\n\nSummarize your findings:"
                    print(f"Adding tool result to messages: {tool_message[:200]}...")

                    # Add tool call and result to conversation log BEFORE adding to worker messages
                    conversation_log["conversation"].append(
                        {
                            "user": "tool",
                            "tool": result["function_name"],
                            "arguments": (
                                json.loads(tool_calls[0][i].function.arguments)
                                if isinstance(tool_calls[0][i].function.arguments, str)
                                else tool_calls[0][i].function.arguments
                            ),
                            "output": None,  # Will be filled after worker responds
                        }
                    )

                    worker_messages.append(
                        {
                            "role": "user",
                            "content": tool_message,
                        }
                    )

                # Get worker's response after tool call
                print("\n--- WORKER THINKING AFTER TOOL CALL ---")
                worker_response, worker_usage_after_tool, _, tc = (
                    self.local_client.chat(
                        messages=worker_messages,
                        # tools=TOOLS,
                    )
                )

                print(f"Worker response after tool call: {worker_response[0]}...")

                # Update the tool call entry with the worker's response
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                # Update usage
                worker_usage += worker_usage_after_tool

            local_usage += worker_usage

            worker_messages.append({"role": "assistant", "content": worker_response[0]})

            if self.callback:
                self.callback("worker", worker_messages[-1])

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                print("\n--- SUPERVISOR FINAL EVALUATION ---")
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )
                print(f"Supervisor final prompt: {supervisor_prompt[:200]}...")

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "remote",
                        "prompt": supervisor_prompt,
                        "output": None,  # Will be filled after response
                    }
                )
            else:
                # First step: Think through the synthesis
                print("\n--- SUPERVISOR SYNTHESIS THINKING ---")
                cot_prompt = REMOTE_SYNTHESIS_COT_TOOLCALLING.format(
                    response=worker_response[0],
                    TOOL_DESCRIPTION=combined_tool_descriptions,
                )
                print(f"Supervisor CoT prompt: {cot_prompt[:200]}...")

                # Add the CoT prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "remote",
                        "prompt": cot_prompt,
                        "output": None,  # Will be filled after response
                    }
                )

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages
                )

                print(f"Supervisor CoT response: {step_by_step_response[0][:200]}...")

                remote_usage += usage

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update the conversation log with the supervisor's CoT response
                conversation_log["conversation"][-1]["output"] = step_by_step_response[
                    0
                ]

                # Second step: Get structured output
                print("\n--- SUPERVISOR STRUCTURED OUTPUT ---")
                supervisor_prompt = REMOTE_SYNTHESIS_FINAL_TOOLCALLING.format(
                    response=step_by_step_response[0],
                    TOOL_DESCRIPTION=combined_tool_descriptions,
                )
                print(f"Supervisor synthesis prompt: {supervisor_prompt[:200]}...")

                # Add the structured output prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "remote",
                        "prompt": supervisor_prompt,
                        "output": None,  # Will be filled after response
                    }
                )

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Get supervisor's response
            print("\n--- SUPERVISOR DECISION ---")
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
                )
            else:
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages
                )

            print(f"Supervisor decision response: {supervisor_response[0][:200]}...")

            remote_usage += supervisor_usage
            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )

            if self.callback:
                self.callback("supervisor", supervisor_messages[-1])

            # Update the conversation log with the supervisor's response
            conversation_log["conversation"][-1]["output"] = supervisor_response[0]

            # Parse supervisor's decision
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                try:
                    supervisor_json = json.loads(supervisor_response[0])
                    print("Successfully parsed supervisor decision JSON directly")
                except:
                    supervisor_json = _extract_json(supervisor_response[0])
                    print("Used _extract_json to parse supervisor decision")
            else:
                supervisor_json = _extract_json(supervisor_response[0])
                print("Used _extract_json to parse supervisor decision")

            if supervisor_json["decision"] == "provide_final_answer":
                print("\n=== FINAL ANSWER REACHED ===")
                final_answer = supervisor_json["answer"]
                print(f"Final answer: {final_answer[:200]}...")
                conversation_log["generated_final_answer"] = final_answer
                break
            else:
                print("\n=== CONTINUING TO NEXT ROUND ===")
                next_question = supervisor_json["message"]
                print(f"Next question: {next_question[:200]}...")
                worker_messages.append({"role": "user", "content": next_question})

                # Add next worker prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "local",
                        "prompt": next_question,
                        "output": None,  # Will be filled after response
                    }
                )

        if final_answer is None:
            print("\n=== NO FINAL ANSWER FOUND ===")
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

        print(f"\n=== SAVING LOG TO {log_path} ===")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        print("\n=== MINION TASK COMPLETED ===")

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
        }

    def _execute_tool_calls(self, tool_calls, custom_tool_executors=None):
        """Execute tool calls and return the results.

        Args:
            tool_calls: List of tool call objects from the LLM
            custom_tool_executors: Dictionary mapping function names to executable functions

        Returns:
            List of dictionaries with tool call results
        """
        print("\n=== EXECUTING TOOL CALLS ===")
        results = []

        if not tool_calls:
            print("No tool calls to execute")
            return results

        custom_tool_executors = custom_tool_executors or {}

        for tool_call in tool_calls[0]:
            print(f"Tool call: {tool_call}")
            function_name = tool_call.function.name
            print(f"Function name: {function_name}")

            # Parse arguments
            try:
                arguments = json.loads(str(tool_call.function.arguments))
            except:
                arguments = tool_call.function.arguments or {}

            print(f"Function arguments: {arguments}")

            result = None
            try:
                # Check if there's a custom executor for this function
                if function_name in custom_tool_executors:
                    # Call the custom executor with the arguments
                    result = custom_tool_executors[function_name](**arguments)
                # Otherwise use the built-in functions
                elif function_name == "list_directory":
                    result = list_directory(arguments.get("path"))
                elif function_name == "read_file":
                    if "file_path" in arguments:
                        result = read_file(arguments["file_path"])
                    else:
                        result = "Error: file_path argument is required"
                elif function_name == "read_multiple_files":
                    if "file_paths" in arguments:
                        if isinstance(arguments["file_paths"], str):
                            file_paths = json.loads(arguments["file_paths"])
                        else:
                            file_paths = arguments["file_paths"]
                        result = read_multiple_files(file_paths)
                    else:
                        result = "Error: file_paths argument is required"
                elif function_name == "write_file":
                    if "file_path" in arguments and "content" in arguments:
                        result = write_file(
                            arguments["file_path"], arguments["content"]
                        )
                    else:
                        result = "Error: file_path and content arguments are required"
                elif function_name == "create_directory":
                    if "path" in arguments:
                        result = create_directory(arguments["path"])
                    else:
                        result = "Error: path argument is required"
                elif function_name == "get_file_info":
                    if "path" in arguments:
                        result = get_file_info(arguments["path"])
                    else:
                        result = "Error: path argument is required"
                elif function_name == "search_files":
                    if "path" in arguments and "pattern" in arguments:
                        result = search_files(
                            arguments["path"],
                            arguments["pattern"],
                            arguments.get("exclude_patterns", []),
                        )
                    else:
                        result = "Error: path and pattern arguments are required"
                else:
                    result = f"Error: Unknown function {function_name}"
            except Exception as e:
                result = f"Error executing {function_name}: {str(e)}"

            print(
                f"Tool result: {result[:100]}..."
                if isinstance(result, str) and len(result) > 100
                else f"Tool result: {result}"
            )

            results.append(
                {
                    "function_name": function_name,
                    "result": result,
                }
            )

        print("=== TOOL EXECUTION COMPLETE ===\n")
        return results


if __name__ == "__main__":

    class StructuredLocalOutput(BaseModel):
        explanation: str
        citation: str | None
        answer: str | None

    remote_client = OpenAIClient(
        model_name="gpt-4o",
    )

    local_client = OllamaClient(model_name="qwen2.5:3b")

    context = """Local Filesystem"""
    task = "Based on my Desktop folder, how many times did I order from Zareens?"
    doc_metadata = "Folder with a list of all my Doordash reciepts!"

    protocol = MinionToolCalling(local_client=local_client, remote_client=remote_client)

    output = protocol(
        task=task,
        doc_metadata=doc_metadata,
        context=[context],
        max_rounds=5,
    )
