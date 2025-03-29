#!/usr/bin/env python3
"""Simple test for multi-turn conversation with the Minions protocol."""

import os
import json
import argparse
from datetime import datetime

from minions.minion import Minion
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.utils.conversation_history import ConversationTurn

# Make sure we have an API key
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set, set it with: export OPENAI_API_KEY=your-key")

def test_multi_turn(context_path):
    """Test multi-turn conversation with context from a file."""
    print("\n=== Testing Multi-Turn ===")
    
    # Get context from file
    if os.path.exists(context_path):
        print(f"Loading: {context_path}")
        with open(context_path, "r") as f:
            context = f.read()
        context_chunks = [context]
    else:
        print(f"File not found: {context_path}")
        context_chunks = ["No context provided."]
    
    # Setup clients
    print("Setting up clients...")
    try:
        local = OllamaClient(model_name="llama3.2")
        remote = OpenAIClient(model_name="gpt-4o")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create minion
    print("Creating minion...")
    minion = Minion(
        local_client=local,
        remote_client=remote,
        max_rounds=2,
        is_multi_turn=True,
        max_history_turns=5
    )
    
    # Let user choose mode
    while True:
        mode = input("\nChoose mode (i=interactive, b=batch): ").lower()
        if mode in ['i', 'interactive']:
            interactive_mode(minion, context_chunks)
            break
        elif mode in ['b', 'batch']:
            batch_mode(minion, context_chunks)
            break
        else:
            print("Invalid choice. Enter 'i' or 'b'.")

def interactive_mode(minion, context_chunks):
    """Run in interactive mode."""
    print("\nInteractive mode:")
    print("- Type 'exit' or 'quit' to end")
    print("- Type 'history' to see conversation history")
    print("- Type 'clear' to clear history")
    
    q_num = 1
    while True:
        query = input(f"\n[Q{q_num}] > ")
        
        if query.lower() in ["exit", "quit"]:
            break
        elif query.lower() == "history":
            show_history(minion)
            continue
        elif query.lower() == "clear":
            minion.conversation_history.clear()
            print("History cleared.")
            continue
        
        # Process query
        print("\n[Processing...]")
        result = minion(
            task=query,
            context=context_chunks,
            max_rounds=2
        )
        
        print(f"\n[A{q_num}]:\n{result['final_answer']}")
        print(f"\nRemote tokens: {result['remote_usage'].total_tokens}")
        print(f"Local tokens: {result['local_usage'].total_tokens}")
        
        q_num += 1

def batch_mode(minion, context_chunks):
    """Run in batch mode with predefined queries."""
    test_queries = [
        "What is the main topic of this document?",
        "Can you provide more details about it?",
        "What were the key points mentioned earlier?",
        "Based on what you told me before, what should I focus on?"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n=== Query {i+1}: {query} ===")
        result = minion(
            task=query,
            context=context_chunks,
            max_rounds=2
        )
        
        print(f"\n[Answer]:\n{result['final_answer']}")
        print(f"\nTokens: {result['remote_usage'].total_tokens} remote, {result['local_usage'].total_tokens} local")
        
        show_history(minion)

def show_history(minion):
    """Show the conversation history."""
    if not minion.is_multi_turn or not minion.conversation_history:
        print("No history available.")
        return
    
    turns = minion.conversation_history.turns
    if not turns:
        print("History is empty.")
        return
    
    print("\n=== History ===")
    
    # Show summary if we have one
    if hasattr(minion.conversation_history, 'summary') and minion.conversation_history.summary:
        print("\n[Earlier Summary]")
        print(minion.conversation_history.summary)
        print("\n[Recent Turns]")
    
    # Show turns
    for i, turn in enumerate(turns):
        print(f"\n[Turn {i+1}]")
        print(f"Q: {turn.query}")
        print(f"A: {turn.remote_output}")
        print(f"Time: {turn.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("===============\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multi-turn conversations")
    parser.add_argument("--context", help="Text file with context")
    
    args = parser.parse_args()
    
    if not args.context:
        print("Need a context file, use --context")
        exit(1)
    
    test_multi_turn(args.context) 