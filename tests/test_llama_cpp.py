#!/usr/bin/env python
"""
Example of using the LlamaCppClient for both chat and embeddings.

This example demonstrates how to:
1. Load a model from Hugging Face Hub or local path
2. Generate chat completions
3. Generate embeddings

Requirements:
- llama-cpp-python: pip install llama-cpp-python
- huggingface-hub: pip install huggingface-hub (for pulling from HF Hub)
"""

import os
import sys

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients import LlamaCppClient


def chat_example(client):
    """Example of using the chat API."""
    print("\n=== Chat Example ===")

    # Single message example
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    responses, usage, done_reasons = client.chat(messages)

    print(f"Response: {responses[0]}")
    print(f"Usage: {usage}")
    print(f"Done reason: {done_reasons[0]}")


def embedding_example(client):
    """Example of using the embedding API."""
    print("\n=== Embedding Example ===")

    # Generate embeddings for a single text
    text = "This is a sample text for embedding."
    embeddings = client.embed(text)

    print(f"Generated embedding with {len(embeddings[0])} dimensions")
    print(f"First few dimensions: {embeddings[0][:5]}...")

    # OpenAI-compatible embedding format
    embedding_response = client.create_embedding(text)
    print(
        f"OpenAI compatible response has {len(embedding_response['data'])} embeddings"
    )


def completion_example(client):
    """Example of using the completion API."""
    print("\n=== Completion Example ===")

    prompt = "Complete this sentence: The quick brown fox"
    responses, usage, done_reasons = client.complete(prompt)

    print(f"Response: {responses[0]}")
    print(f"Usage: {usage}")
    print(f"Done reason: {done_reasons[0]}")


def main():
    """Main example function."""
    # Method 1: Load a model from local path
    local_model_path = "/path/to/your/model.gguf"

    # Method 2: Load a model from Hugging Face Hub (recommended)
    # Example using Phi-3 Mini from Hugging Face
    model_repo_id = "Qwen/Qwen2-0.5B-Instruct-GGUF"
    model_file_pattern = "*q8_0.gguf"  # Use Q4_0 quantization

    # Uncomment the method you want to use
    # client = LlamaCppClient(
    #     model_path=local_model_path,
    #     chat_format="chatml",    # Adjust based on model
    #     temperature=0.7,
    #     n_gpu_layers=35,        # Set to number of layers to offload to GPU
    #     embedding=True          # Enable embeddings
    # )

    client = LlamaCppClient(
        model_path="",
        model_repo_id=model_repo_id,
        model_file_pattern=model_file_pattern,
        chat_format="chatml",  # Adjust based on model
        temperature=0.7,
        n_gpu_layers=35,  # Set to number of layers to offload to GPU
        embedding=True,  # Enable embeddings
    )

    # Run examples
    chat_example(client)
    completion_example(client)

    # Embedding example (only if model supports embeddings)
    try:
        embedding_example(client)
    except Exception as e:
        print(f"Embedding failed: {e}")
        print(
            "Note: Not all models support embeddings. Make sure your model does and 'embedding=True' is set."
        )


if __name__ == "__main__":
    main()
