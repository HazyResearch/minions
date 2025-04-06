#!/usr/bin/env python3
"""Quick test for the conversation summarization feature."""

import os
import time
import argparse
from datetime import datetime

from minions.minion import Minion
from minions.clients.openai import OpenAIClient
from minions.clients.ollama import OllamaClient
from minions.utils.conversation_history import ConversationTurn

if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set, use: export OPENAI_API_KEY=your-key")

def test_summary(context_path=None):
    """Test the summarization of older conversation turns."""
    print("\n=== Testing Summarization ===")
    
    # Setup clients
    print("Setting up clients...")
    try:
        local = OllamaClient(model_name="llama3.2")
        remote = OpenAIClient(model_name="gpt-4o")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create minion for testing
    print("Creating minion...")
    minion = Minion(
        local_client=local,
        remote_client=remote,
        max_rounds=2,
        is_multi_turn=True,
        max_history_turns=5  # Small window for faster testing
    )
    
    # Setup summarization
    print("Setting up summarization...")
    minion.conversation_history.summarize_older_turns = True
    minion.conversation_history.turns_per_summary = 3  # Summarize after 3 turns
    
    # Get context if provided
    context_chunks = []
    if context_path and os.path.exists(context_path):
        print(f"Loading context: {context_path}")
        with open(context_path, "r") as f:
            context = f.read()
        context_chunks = [context]
    
    # Let user choose mode
    while True:
        mode = input("\nChoose mode (i=interactive, b=batch): ").lower()
        if mode in ['i', 'interactive']:
            interactive_mode(minion, context_chunks)
            break
        elif mode in ['b', 'batch']:
            batch_mode(minion)
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

def batch_mode(minion):
    """Run batch test of conversation summarization."""
    # Test conversations about ML topics
    test_convos = [
        ("What is machine learning?", "Machine learning is a subfield of artificial intelligence that focuses on developing algorithms and models that enable computers to learn from data without being explicitly programmed."),
        ("How does supervised learning work?", "Supervised learning works by training algorithms on labeled data, where each example has an input and expected output. The algorithm learns to map inputs to outputs, allowing it to make predictions on new, unseen data."),
        ("What's the difference between classification and regression?", "Classification predicts categorical outcomes (like spam/not spam), while regression predicts continuous values (like temperature or price). Classification assigns data to discrete categories, while regression estimates a continuous relationship between variables."),
        ("Can you explain neural networks?", "Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process information. Each connection has a weight that adjusts during learning. Neural networks can learn complex patterns, making them powerful for tasks like image recognition and language processing."),
        ("What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). These networks can automatically extract hierarchical features from raw data, learning increasingly abstract representations. Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition."),
        ("How does reinforcement learning work?", "Reinforcement learning involves an agent learning to make decisions by taking actions in an environment to maximize rewards. The agent learns through trial and error, receiving feedback in the form of rewards or penalties. Over time, it develops a policy that maps situations to actions that yield the highest rewards."),
        ("What are CNNs?", "Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to extract features, pooling layers to reduce dimensionality, and fully connected layers for classification."),
        ("How does unsupervised learning work?", "Unsupervised learning works with unlabeled data, finding patterns and structures without predefined outputs. Unlike supervised learning, there are no correct answers to guide the learning process. The algorithm must discover the inherent structure of the data on its own.")
    ]
    
    print("\nAdding turns to conversation...")
    
    # Add turns one by one
    for i, (q, a) in enumerate(test_convos):
        print(f"Turn {i+1}: {q[:30]}...")
        
        # Make a conversation turn
        turn = ConversationTurn(
            query=q,
            local_output="Local output for " + q,
            remote_output=a,
            timestamp=datetime.now()
        )
        
        # Add it to history
        minion.conversation_history.add_turn(turn, remote_client=minion.remote_client)
        
        # Show stats
        print(f"  Turns: {len(minion.conversation_history.turns)}")
        print(f"  Since summary: {minion.conversation_history.turns_since_last_summary}")
        if minion.conversation_history.summary:
            print("  Summary: Yes")
        else:
            print("  Summary: No")
        
        # Pause to see what's happening
        time.sleep(0.5)
    
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
    for i, turn in enumerate(turns):
        print(f"\n[Turn {i+1}]")
        print(f"Q: {turn.query}")
        print(f"A: {turn.remote_output[:80]}...")  # Just show start of answer
    
    print("\n=== End of History ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test conversation summarization")
    parser.add_argument("--context", help="Optional text file with context")
    
    args = parser.parse_args()
    
    test_summary(args.context) 