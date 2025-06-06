#!/usr/bin/env python
"""
Example of using the MistralClient for chat completions.

This example demonstrates how to:
1. Initialize the Mistral client
2. Generate chat completions
3. Handle API responses

Requirements:
- mistralai: pip install mistralai
- MISTRAL_API_KEY environment variable set

Note: This test requires a valid Mistral API key to run.
"""

import os
import sys

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients import MistralClient


def chat_example(client):
    """Example of using the chat API."""
    print("\n=== Chat Example ===")

    # Single message example
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    try:
        responses, usage, done_reasons = client.chat(messages)

        print(f"Response: {responses[0]}")
        print(f"Usage: {usage}")
        print(f"Done reason: {done_reasons[0]}")
    except Exception as e:
        print(f"Error during chat: {e}")


def multi_turn_example(client):
    """Example of multi-turn conversation."""
    print("\n=== Multi-turn Example ===")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Tell me a joke about programming."},
    ]

    try:
        responses, usage, done_reasons = client.chat(messages)
        print(f"Assistant: {responses[0]}")

        # Add the assistant's response and continue the conversation
        messages.append({"role": "assistant", "content": responses[0]})
        messages.append({"role": "user", "content": "That's funny! Tell me another one."})

        responses, usage, done_reasons = client.chat(messages)
        print(f"Assistant: {responses[0]}")
        print(f"Total usage: {usage}")
    except Exception as e:
        print(f"Error during multi-turn chat: {e}")


def main():
    """Main example function."""
    # Check if API key is available
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        print("Please set your Mistral API key:")
        print("export MISTRAL_API_KEY='your-api-key-here'")
        return

    # Initialize the client with different models
    models_to_test = [
        "mistral-large-2411",
    ]

    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing with model: {model_name}")
        print(f"{'='*50}")

        try:
            client = MistralClient(
                model_name=model_name,
                temperature=0.7,
                max_tokens=1024,
            )

            # Run examples
            chat_example(client)
            multi_turn_example(client)

        except Exception as e:
            print(f"Error initializing client for {model_name}: {e}")
            continue

        # Only test one model if we're doing a quick test
        break


if __name__ == "__main__":
    main() 