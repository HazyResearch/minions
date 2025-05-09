#!/usr/bin/env python3
"""
Example usage of MinionToolCalling with practical examples.
This demonstrates how to use minion_wtools for file operations
and document analysis tasks.
"""

import os
import time
import tempfile
import json
from pathlib import Path

# Import the clients and MinionToolCalling class
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion_wtools import MinionToolCalling


def setup_test_env():
    """Set up a test environment with some sample files."""
    # Create a temporary directory with some sample files
    test_dir = tempfile.mkdtemp(prefix="minion_wtools_test_")

    # Create a sample text file
    with open(os.path.join(test_dir, "sample.txt"), "w") as f:
        f.write(
            "This is a sample text file.\nIt has multiple lines.\nIt can be used for testing."
        )

    # Create a sample Python file
    with open(os.path.join(test_dir, "script.py"), "w") as f:
        f.write(
            '''
def hello_world():
    """Print a greeting."""
    print("Hello, world!")

def add(a, b):
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    hello_world()
    print(f"2 + 3 = {add(2, 3)}")
'''
        )

    # Create a sample JSON file
    with open(os.path.join(test_dir, "data.json"), "w") as f:
        json.dump(
            {
                "name": "Test Data",
                "values": [1, 2, 3, 4, 5],
                "metadata": {"created": "2023-06-15", "author": "Test User"},
            },
            f,
            indent=2,
        )

    # Create a sample nested directory structure
    nested_dir = os.path.join(test_dir, "folder1", "folder2")
    os.makedirs(nested_dir)

    with open(os.path.join(nested_dir, "nested.txt"), "w") as f:
        f.write("This is a file in a nested directory.")

    print(f"Created test environment at: {test_dir}")
    return test_dir


def example_file_analysis():
    """Show how to use MinionToolCalling to analyze files in a directory."""
    # Create test environment
    test_dir = setup_test_env()

    # Set up clients (use your preferred models)
    try:
        local_client = OllamaClient(model_name="llama3.2:3b", tool_calling=True)
    except Exception as e:
        print(f"Couldn't initialize Ollama client: {e}")
        print("Using OpenAI client as fallback for local client")
        local_client = OpenAIClient(model_name="gpt-3.5-turbo")

    remote_client = OpenAIClient(model_name="gpt-4o")

    # Create the MinionToolCalling instance
    minion = MinionToolCalling(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=3,
        log_dir="minion_logs",
    )

    # Define the task
    context = f"Directory to analyze: {test_dir}"
    task = f"Analyze the directory structure and contents at {test_dir}. List all files and summarize their content."

    # Run the minion
    print("\n=== Running File Analysis Example ===")
    output = minion(
        task=task,
        context=[context],
        max_rounds=3,
        logging_id=f"file_analysis_{int(time.time())}",
    )

    # Print results
    print("\n=== File Analysis Results ===")
    print(f"Final answer: {output['final_answer']}")
    print(f"Remote tokens: {output['remote_usage'].total_tokens}")
    print(f"Local tokens: {output['local_usage'].total_tokens}")
    print(f"Log file: {output['log_file']}")

    return test_dir


def example_code_analysis():
    """Show how to use MinionToolCalling to analyze Python code."""
    # Set up clients
    try:
        local_client = OllamaClient(model_name="codellama:13b", tool_calling=True)
    except Exception as e:
        print(f"Couldn't initialize Ollama client: {e}")
        print("Using OpenAI client as fallback for local client")
        local_client = OpenAIClient(model_name="gpt-3.5-turbo")

    remote_client = OpenAIClient(model_name="gpt-4o")

    # Create the MinionToolCalling instance
    minion = MinionToolCalling(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=3,
        log_dir="minion_logs",
    )

    # Define the task - analyze Python code in current directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    file_path = os.path.join(parent_dir, "minions", "minion_wtools.py")

    context = f"File to analyze: {file_path}"
    task = f"Analyze the Python file at {file_path}. Identify its main functions and classes, and explain their purpose."

    # Run the minion
    print("\n=== Running Code Analysis Example ===")
    output = minion(
        task=task,
        context=[context],
        max_rounds=3,
        logging_id=f"code_analysis_{int(time.time())}",
    )

    # Print results
    print("\n=== Code Analysis Results ===")
    print(f"Final answer: {output['final_answer']}")
    print(f"Remote tokens: {output['remote_usage'].total_tokens}")
    print(f"Local tokens: {output['local_usage'].total_tokens}")
    print(f"Log file: {output['log_file']}")


def example_document_extraction():
    """Show how to use MinionToolCalling to extract information from a document."""
    # Create test data
    test_dir = tempfile.mkdtemp(prefix="minion_wtools_doc_")

    # Create a sample document with structured information
    doc_content = """
# Project Status Report

## Project Overview
Project Name: Alpha Integration Platform
Project Lead: Jane Smith
Start Date: 2023-01-15
Expected Completion: 2023-08-30
Current Status: In Progress (75% complete)

## Team Members
- Jane Smith (Project Lead)
- John Doe (Backend Developer)
- Alice Johnson (Frontend Developer)
- Bob Williams (QA Engineer)
- Carol Brown (DevOps)

## Key Milestones
1. Requirements Gathering - COMPLETED (2023-01-30)
2. System Design - COMPLETED (2023-03-15)
3. Backend Development - COMPLETED (2023-05-20)
4. Frontend Development - IN PROGRESS (Expected: 2023-07-10)
5. Integration Testing - NOT STARTED (Expected: 2023-07-25)
6. Deployment - NOT STARTED (Expected: 2023-08-20)

## Issues and Risks
- Frontend development delayed by 1 week due to design changes
- Integration with legacy systems requires additional testing
- Performance concerns with high-volume data processing

## Next Steps
- Complete frontend development by July 10
- Begin integration testing by July 15
- Schedule security audit for week of July 17
- Prepare deployment documentation by August 1
"""

    doc_path = os.path.join(test_dir, "project_report.md")
    with open(doc_path, "w") as f:
        f.write(doc_content)

    # Set up clients
    try:
        local_client = OllamaClient(model_name="llama3.2:3b", tool_calling=True)
    except Exception as e:
        print(f"Couldn't initialize Ollama client: {e}")
        print("Using OpenAI client as fallback for local client")
        local_client = OpenAIClient(model_name="gpt-3.5-turbo")

    remote_client = OpenAIClient(model_name="gpt-4o")

    # Create the MinionToolCalling instance
    minion = MinionToolCalling(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=3,
        log_dir="minion_logs",
    )

    # Define the task
    context = f"Document to analyze: {doc_path}"
    task = """
    Extract the following information from the project report:
    1. Project name and current completion percentage
    2. All team members and their roles
    3. The status of each milestone
    4. All identified risks
    5. Create a timeline of upcoming activities
    """

    # Run the minion
    print("\n=== Running Document Extraction Example ===")
    output = minion(
        task=task,
        context=[context],
        max_rounds=3,
        logging_id=f"doc_extraction_{int(time.time())}",
    )

    # Print results
    print("\n=== Document Extraction Results ===")
    print(f"Final answer: {output['final_answer']}")
    print(f"Remote tokens: {output['remote_usage'].total_tokens}")
    print(f"Local tokens: {output['local_usage'].total_tokens}")
    print(f"Log file: {output['log_file']}")

    return test_dir


def example_custom_tools():
    """Show how to use MinionToolCalling with custom tools."""

    # Define custom tools
    custom_tools = [
        {
            "type": "function",
            "function": {
                "name": "count_lines",
                "description": "Count the number of lines in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_pattern",
                "description": "Find all occurrences of a pattern in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Pattern to search for",
                        },
                    },
                    "required": ["file_path", "pattern"],
                },
            },
        },
    ]

    # Define custom tool executors
    def count_lines(file_path):
        try:
            with open(file_path, "r") as f:
                return len(f.readlines())
        except Exception as e:
            return f"Error counting lines: {str(e)}"

    def find_pattern(file_path, pattern):
        try:
            with open(file_path, "r") as f:
                content = f.read()

            import re

            matches = re.findall(pattern, content)
            return {
                "count": len(matches),
                "matches": matches[:10] if len(matches) > 10 else matches,
            }
        except Exception as e:
            return f"Error finding pattern: {str(e)}"

    # Custom tool executors dictionary
    custom_tool_executors = {"count_lines": count_lines, "find_pattern": find_pattern}

    # Custom tool descriptions
    custom_tool_descriptions = """
count_lines:
Description: Count the number of lines in a file
Arguments: file_path (required) - Path to the file

find_pattern:
Description: Find all occurrences of a pattern in a file
Arguments: 
  file_path (required) - Path to the file
  pattern (required) - Pattern to search for (regular expression)
"""

    # Create test data
    test_dir = setup_test_env()

    # Set up clients
    try:
        local_client = OllamaClient(model_name="llama3.2:3b", tool_calling=True)
    except Exception as e:
        print(f"Couldn't initialize Ollama client: {e}")
        print("Using OpenAI client as fallback for local client")
        local_client = OpenAIClient(model_name="gpt-3.5-turbo")

    remote_client = OpenAIClient(model_name="gpt-4o")

    # Create the MinionToolCalling instance with custom tools
    minion = MinionToolCalling(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=3,
        log_dir="minion_logs",
        custom_tools=custom_tools,
        custom_tool_executors=custom_tool_executors,
        custom_tool_descriptions=custom_tool_descriptions,
    )

    # Define the task
    context = f"Directory to analyze: {test_dir}"
    task = f"""
    For each .py file in the directory {test_dir}:
    1. Count the number of lines
    2. Find all function definitions (pattern: 'def \\w+\\(')
    3. Count the number of if statements (pattern: '\\s+if\\s+')
    
    Then summarize which file has the most functions and which has the most lines.
    """

    # Run the minion
    print("\n=== Running Custom Tools Example ===")
    output = minion(
        task=task,
        context=[context],
        max_rounds=3,
        logging_id=f"custom_tools_{int(time.time())}",
    )

    # Print results
    print("\n=== Custom Tools Results ===")
    print(f"Final answer: {output['final_answer']}")
    print(f"Remote tokens: {output['remote_usage'].total_tokens}")
    print(f"Local tokens: {output['local_usage'].total_tokens}")
    print(f"Log file: {output['log_file']}")

    return test_dir


def cleanup_test_dirs(*dirs):
    """Clean up test directories."""
    for d in dirs:
        if os.path.exists(d):
            import shutil

            shutil.rmtree(d)
            print(f"Cleaned up test directory: {d}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("minion_logs", exist_ok=True)

    # Choose which examples to run
    print("Which example would you like to run?")
    print("1. File Analysis")
    print("2. Code Analysis")
    print("3. Document Extraction")
    print("4. Custom Tools")
    print("5. Run All Examples")

    choice = input("Enter choice (1-5): ").strip()

    test_dirs = []

    try:
        if choice == "1" or choice == "5":
            test_dir = example_file_analysis()
            test_dirs.append(test_dir)

        if choice == "2" or choice == "5":
            example_code_analysis()

        if choice == "3" or choice == "5":
            test_dir = example_document_extraction()
            test_dirs.append(test_dir)

        if choice == "4" or choice == "5":
            test_dir = example_custom_tools()
            test_dirs.append(test_dir)

    finally:
        # Ask if user wants to clean up test directories
        if test_dirs:
            cleanup = input("\nClean up test directories? (y/n): ").strip().lower()
            if cleanup == "y":
                cleanup_test_dirs(*test_dirs)
