#!/usr/bin/env python3
from minions.minion import Minion
from minions.minions import Minions
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.together import TogetherClient
from minions.clients.groq import GroqClient
from minions.clients.mlx_lm import MLXLMClient
from minions.clients.perplexity import PerplexityAIClient
from minions.clients.openrouter import OpenRouterClient
import time
import argparse
import fitz  # PyMuPDF for PDF handling
import json
import os
import sys
import readline
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
import re


def extract_text_from_file(file_path):
    """Extract text from a PDF, TXT, Python, or Markdown file."""
    try:
        # Expand ~ to user's home directory if present
        file_path = os.path.expanduser(file_path)

        if file_path.lower().endswith(".pdf"):
            # Handle PDF file
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        elif file_path.lower().endswith((".txt", ".py", ".md")):
            # Handle text-based files
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(
                "Unsupported file format. Only PDF, TXT, PY, and MD files are supported."
            )
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return ""


def extract_text_from_folder(folder_path):
    """Extract text from all supported files in a folder."""
    try:
        # Expand ~ to user's home directory if present
        folder_path = os.path.expanduser(folder_path)

        if not os.path.isdir(folder_path):
            raise ValueError(f"'{folder_path}' is not a valid directory")

        # Dictionary to store file contents with filenames as keys
        file_contents = {}
        file_count = 0
        total_chars = 0

        # Walk through the directory
        for root, _, files in os.walk(folder_path):
            # skip folder with examples in it
            if "examples" in root:
                continue

            for file in files:
                file_path = os.path.join(root, file)

                # Only process supported file types
                if file.lower().endswith((".txt", ".py")):
                    try:
                        content = extract_text_from_file(file_path)
                        if content:
                            # Use relative path from the base folder as the key
                            rel_path = os.path.relpath(file_path, folder_path)
                            file_contents[rel_path] = content
                            file_count += 1
                            total_chars += len(content)
                            print(f"Loaded: {rel_path} ({len(content)} chars)")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")

        if not file_contents:
            print("No supported files found in the directory.")
            return ""

        # Combine all texts with file headers
        combined_text = ""
        for filename, content in file_contents.items():
            combined_text += f"\n\n--- BEGIN FILE: {filename} ---\n\n"
            combined_text += content
            combined_text += f"\n\n--- END FILE: {filename} ---\n\n"

        print(
            f"Successfully loaded {file_count} files with a total of {total_chars} characters."
        )
        return combined_text

    except Exception as e:
        print(f"Error processing folder: {str(e)}")
        return ""


def load_default_medical_context():
    try:
        with open("data/test_medical.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Default medical context file not found!")
        return ""


class JobOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None


def format_usage(usage, model_name):
    total_tokens = usage.prompt_tokens + usage.completion_tokens
    return (
        f"\n{model_name} Usage Statistics:\n"
        f"  Prompt Tokens: {usage.prompt_tokens}\n"
        f"  Completion Tokens: {usage.completion_tokens}\n"
        f"  Total Tokens: {total_tokens}\n"
    )


def parse_model_string(model_string):
    """Parse a model string in the format provider/model_name."""
    if "/" not in model_string:
        return "ollama", model_string  # Default to ollama if no provider specified

    provider, model_name = model_string.split("/", 1)
    return provider.lower(), model_name


# Global variables to track current message state
current_message = {"role": None, "content": ""}
is_streaming = False


def message_callback(role, message, is_final=False):
    """Stream messages from both local and remote models with real-time updates."""
    global current_message, is_streaming

    if role == "supervisor":
        prefix = "\033[1;35m[Remote]\033[0m"  # Blue for remote/supervisor
    else:
        prefix = "\033[1;36m[Local]\033[0m"  # Green for local/worker

    # If this is a new message or a different role
    if current_message["role"] != role:
        # If we were streaming a previous message, finish it
        if is_streaming:
            print()  # End the current line

        # Start a new message
        current_message = {"role": role, "content": ""}
        is_streaming = True

        # Print the prefix for the new message
        print(f"{prefix} ", end="", flush=True)

    # Handle different message types
    if isinstance(message, list):
        # For Minions protocol, messages are a list of jobs
        if is_final:
            # Display jobs in a structured format similar to app.py
            total_jobs = len(message)
            successful_jobs = sum(1 for job in message if job.include)

            print(f"Processed {successful_jobs}/{total_jobs} chunks successfully")

            # Group jobs by task
            tasks = {}
            for job in message:
                task_id = job.manifest.task_id
                if task_id not in tasks:
                    tasks[task_id] = {"task": job.manifest.task, "jobs": []}
                tasks[task_id]["jobs"].append(job)

            # Print each task and its jobs
            for task_id, task_info in tasks.items():
                print(f"\n\033[1;33mTask: {task_info['task']}\033[0m")

                # Sort jobs by chunk_id
                sorted_jobs = sorted(
                    task_info["jobs"], key=lambda x: x.manifest.chunk_id
                )

                # Print successful jobs first
                for job in [j for j in sorted_jobs if j.include]:
                    chunk_id = job.manifest.chunk_id
                    print(f"\n\033[1;32m✅ Chunk {chunk_id + 1}:\033[0m")

                    # Print chunk preview
                    chunk_preview = (
                        job.manifest.chunk[:100] + "..."
                        if len(job.manifest.chunk) > 100
                        else job.manifest.chunk
                    )
                    print(f"  \033[1;36mChunk preview:\033[0m {chunk_preview}")

                    # Print job outputs
                    if job.output.answer:
                        print(f"  \033[1;36mAnswer:\033[0m {job.output.answer}")
                    if job.output.explanation:
                        print(
                            f"  \033[1;36mExplanation:\033[0m {job.output.explanation}"
                        )
                    if job.output.citation:
                        print(f"  \033[1;36mCitation:\033[0m {job.output.citation}")

                # Optionally print unsuccessful jobs
                failed_jobs = [j for j in sorted_jobs if not j.include]
                if failed_jobs:
                    print(
                        f"\n\033[1;31mChunks without relevant information ({len(failed_jobs)}):\033[0m"
                    )
                    for job in failed_jobs[:3]:  # Show only first 3 failed jobs
                        print(f"  Chunk {job.manifest.chunk_id + 1}")

            is_streaming = False
        else:
            print("Working on chunks...", end="\r", flush=True)
    elif isinstance(message, dict) and ("content" in message or "message" in message):
        if "content" in message:
            content = message["content"]
        elif "message" in message:
            content = message["message"]

        content = content.replace("\n\n\n", "")

        if is_final:
            print(content.strip("\n") + "\n")
            is_streaming = False
        else:
            print(content, end="\r", flush=True)
    else:
        # Regular string message
        if is_final:
            print(message.strip("\n"))
            is_streaming = False
        else:
            # For streaming updates, show progress
            print(
                f"Working...\n",
                end="\r",
                flush=True,
            )
            current_message["content"] = message


def initialize_client(
    provider,
    model_name,
    temperature=0.0,
    max_tokens=4096,
    num_ctx=4096,
    structured_output=None,
    use_async=False,
):
    """Initialize a client based on provider name."""
    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            structured_output_schema=structured_output,
            use_async=use_async,
        )
    elif provider == "openai":
        return OpenAIClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    elif provider == "anthropic":
        return AnthropicClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    elif provider == "together":
        return TogetherClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    elif provider == "groq":
        return GroqClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    elif provider == "perplexity":
        return PerplexityAIClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    elif provider == "openrouter":
        return OpenRouterClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    elif provider == "mlx":
        return MLXLMClient(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def chat_loop(protocol, context, doc_metadata):
    """Run an interactive chat loop with the protocol."""
    print("\n\033[1;33m=== Minions ===\033[0m")
    print("Type 'exit', 'quit', or Ctrl+D to end the conversation.")
    print("Type your message and press Enter to chat with the document.\n")

    history = []

    while True:
        try:
            # Get user input
            user_input = input("\033[1;36m> \033[0m")

            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                break

            # Run the protocol with the user's query
            print("\n\033[1;33m=== Processing ===\033[0m")

            # Reset global state for new conversation
            global current_message, is_streaming
            current_message = {"role": None, "content": ""}
            is_streaming = False

            # Execute the protocol
            output = protocol(
                task=user_input,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=5,
            )

            # Store the conversation
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": output["final_answer"]})

            # Print a separator for the next interaction
            print("\n\033[1;33m=== Ready for next query ===\033[0m")

        except EOFError:
            # Handle Ctrl+D
            print("\nExiting chat...")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nInterrupted. Type 'exit' to quit or continue with a new query.")
            continue
        except Exception as e:
            print(f"\n\033[1;31mError: {str(e)}\033[0m")
            print("Please try again with a different query.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Minions: where local llms meet cloud llms"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Path to a PDF/TXT file or a folder containing documents",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["minion", "minions"],
        default="minion",
        help="The protocol to use (default: minion)",
    )
    parser.add_argument(
        "--doc-metadata", type=str, default="", help="Metadata describing the document"
    )
    args = parser.parse_args()

    # Get model configuration from environment variables
    local_model_env = os.environ.get("MINIONS_LOCAL", "ollama/llama3.2")
    remote_model_env = os.environ.get("MINIONS_REMOTE", "openai/gpt-4o")

    # Parse the model strings
    local_provider, local_model_name = parse_model_string(local_model_env)
    remote_provider, remote_model_name = parse_model_string(remote_model_env)

    # Default parameters
    local_temperature = 0.0
    local_max_tokens = 4096
    remote_temperature = 0.2
    remote_max_tokens = 2048

    # Load context from file or folder if provided
    context = ""
    if args.context:
        # Expand user path
        context_path = os.path.expanduser(args.context)

        # Check if it's a directory
        if os.path.isdir(context_path):
            print(f"Loading documents from folder: {context_path}")
            context = extract_text_from_folder(context_path)
            if not context:
                print("Error: Could not extract text from the specified folder")
                return
        else:
            # Treat as a single file
            print(f"Loading document: {context_path}")
            context = extract_text_from_file(context_path)
            if not context:
                print("Error: Could not extract text from the specified file")
                return

        print(f"Total context size: {len(context)} characters")
    else:
        print("No context file or folder provided. Starting with empty context.")

    # Set document metadata
    doc_metadata = args.doc_metadata
    if not doc_metadata and args.context:
        if os.path.isdir(os.path.expanduser(args.context)):
            doc_metadata = f"Multiple documents from folder: {os.path.basename(os.path.expanduser(args.context))}"
        else:
            doc_metadata = (
                f"Document: {os.path.basename(os.path.expanduser(args.context))}"
            )

    print("Initializing clients...")
    setup_start_time = time.time()

    # Configure protocol-specific settings
    if args.protocol == "minions":
        # the local worker operates on chunks of data
        num_ctx = 4096
        structured_output_schema = JobOutput
        async_mode = True
    else:  # minion protocol
        structured_output_schema = None
        async_mode = False
        # For Minion protocol, estimate tokens based on context length (4 chars ≈ 1 token)
        # Add 4000 to account for the conversation history
        estimated_tokens = int(len(context) / 4 + 4000) if context else 4096
        # Round up to nearest power of 2 from predefined list
        num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        # Find the smallest value that is >= estimated tokens
        num_ctx = min(
            [x for x in num_ctx_values if x >= estimated_tokens], default=131072
        )
        print(f"Estimated tokens: {estimated_tokens}")
        print(f"Using context window: {num_ctx}")

    # Initialize the local client
    print(f"Initializing local client with {local_provider}/{local_model_name}")
    local_client = initialize_client(
        provider=local_provider,
        model_name=local_model_name,
        temperature=local_temperature,
        max_tokens=local_max_tokens,
        num_ctx=num_ctx,
        structured_output=structured_output_schema,
        use_async=async_mode,
    )

    # Initialize the remote client
    print(f"Initializing remote client with {remote_provider}/{remote_model_name}")
    remote_client = initialize_client(
        provider=remote_provider,
        model_name=remote_model_name,
        temperature=remote_temperature,
        max_tokens=remote_max_tokens,
    )

    # Instantiate the protocol object with the clients
    print(f"Initializing {args.protocol} protocol")
    if args.protocol == "minions":
        protocol = Minions(local_client, remote_client, callback=message_callback)
    else:  # minion
        protocol = Minion(local_client, remote_client, callback=message_callback)

    setup_time = time.time() - setup_start_time
    print(f"Setup completed in {setup_time:.2f} seconds")

    # Start the interactive chat loop
    chat_loop(protocol, context, doc_metadata)


if __name__ == "__main__":
    main()
