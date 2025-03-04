from minions.minion import Minion
from minions.minions import Minions
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.together import TogetherClient
from minions.clients.groq import GroqClient
import time
import argparse
import fitz  # PyMuPDF for PDF handling
import json
from pydantic import BaseModel, Field

def extract_text_from_file(file_path):
    """Extract text from a PDF or TXT file."""
    try:
        if file_path.lower().endswith('.pdf'):
            # Handle PDF file
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        elif file_path.lower().endswith('.txt'):
            # Handle TXT file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Unsupported file format. Only PDF and TXT files are supported.")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run minion with specified provider')
    parser.add_argument('--provider', type=str, choices=['openai', 'anthropic', 'together', 'groq'], default='openai',
                      help='The provider to use as remote client (default: openai)')
    parser.add_argument('--remote-model', type=str, default='gpt-4o-mini',
                      help='The remote model to use (default: gpt-4o-mini)')
    parser.add_argument('--local-model', type=str, default='llama3.2',
                      help='The local model to use (default: llama3.2)')
    parser.add_argument('--protocol', type=str, choices=['minion', 'minions'], default='minion',
                      help='The protocol to use (default: minion)')
    parser.add_argument('--file', type=str, default='/Users/biderman/Downloads/amazon_report.pdf',
                      help='Path to a PDF or TXT file to process (default: /Users/biderman/Downloads/amazon_report.pdf)')
    parser.add_argument('--doc-metadata', type=str, default='amazon 10k report',
                      help='Metadata describing the document (default: amazon 10k report)')
    parser.add_argument('--task', type=str, 
                      default="How did Amazon's operating income change from 2018 to 2019?",
                      help='The task or question to be answered')
    args = parser.parse_args()

    # Default parameters
    local_model_name = args.local_model
    local_temperature = 0.0
    local_max_tokens = 4096
    
    # Provider-specific parameters
    if args.provider == 'anthropic':
        remote_model_name = "claude-3-sonnet-20240229"
    elif args.provider == 'together':
        remote_model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    elif args.provider == 'groq':
        remote_model_name = "llama-3.3-70b-versatile"
    else:  # openai
        remote_model_name = args.remote_model
    
    remote_temperature = 0.2
    remote_max_tokens = 2048
    
    # Load context from file
    context = extract_text_from_file(args.file)
    if not context:
        print("Error: Could not extract text from the specified file")
        return
    print("Truncating context to 70000 characters")
    context = context[10000:80000]
    
    print("Initializing clients...")
    setup_start_time = time.time()

    if args.protocol == "minions":
        # the local worker operates on chunks of data
        num_ctx = 4096
        structured_output_schema = JobOutput
        async_mode = True
    elif args.protocol == "minion":
        structured_output_schema = None
        async_mode = False
        # For Minion protocol, estimate tokens based on context length (4 chars ≈ 1 token)
        # Add 4000 to account for the conversation history
        estimated_tokens = int(len(context) / 4 + 4000) if context else 4096
        # Round up to nearest power of 2 from predefined list
        num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        # Find the smallest value that is >= estimated tokens
        num_ctx = min([x for x in num_ctx_values if x >= estimated_tokens], default=131072)
        print(f"Estimated tokens: {estimated_tokens}")
        print(f"Using context window: {num_ctx}")
    
    # Initialize the local client
    print(f"Initializing local client with model: {local_model_name}")
    local_client = OllamaClient(
        model_name=local_model_name,
        temperature=local_temperature,
        max_tokens=local_max_tokens,
        num_ctx=num_ctx,
        structured_output_schema=structured_output_schema,
        use_async=async_mode
    )
    
    # Initialize the remote client based on provider
    print(f"Initializing remote client with model: {remote_model_name}")
    if args.provider == 'anthropic':
        remote_client = AnthropicClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=remote_max_tokens
        )
    elif args.provider == 'together':
        remote_client = TogetherClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=remote_max_tokens
        )
    elif args.provider == 'groq':
        remote_client = GroqClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=remote_max_tokens
        )
    else:  # openai
        remote_client = OpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=remote_max_tokens
        )
    
    # Instantiate the protocol object with the clients
    print(f"Initializing {args.protocol} with clients")
    if args.protocol == 'minions':
        protocol = Minions(local_client, remote_client)
    else:  # minion
        protocol = Minion(local_client, remote_client)
    
    setup_time = time.time() - setup_start_time
    print(f"Setup completed in {setup_time:.2f} seconds")
    
    print("\nSolving task...")
    execution_start_time = time.time()
    
    # Execute the protocol
    output = protocol(
        task=args.task,
        doc_metadata=args.doc_metadata, # only for minions, to be removed soon
        context=[context],  # Use the extracted text from file as context
        max_rounds=2
    )
    
    execution_time = time.time() - execution_start_time
    total_time = setup_time + execution_time
    
    # Print results
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"\nFinal Answer:\n{output['final_answer']}")
    
    print("\nTiming Information:")
    print(f"Setup Time: {setup_time:.2f}s ({(setup_time/total_time*100):.1f}% of total)")
    print(f"Execution Time: {execution_time:.2f}s ({(execution_time/total_time*100):.1f}% of total)")
    
    # Print usage information if available
    if "local_usage" in output and "remote_usage" in output:
        print("\nUsage Information:")
        print(format_usage(output["local_usage"], local_model_name))
        print(format_usage(output["remote_usage"], remote_model_name))

    # Print protocol-specific information
    if args.protocol == 'minion':
        pass
        # print("\nSupervisor Messages:")
        # for msg in output["supervisor_messages"]:
        #     print(f"\n{msg['role'].capitalize()}: {msg['content']}")
        
        # print("\nWorker Messages:")
        # for msg in output["worker_messages"]:
        #     print(f"\n{msg['role'].capitalize()}: {msg['content']}")
    elif "meta" in output:  # minions protocol
        print("\nMeta Information:")
        for round_idx, round_meta in enumerate(output["meta"]):
            print(f"\nRound {round_idx + 1}:")
            if "local" in round_meta:
                print(f"Local jobs: {len(round_meta['local']['jobs'])}")
            if "remote" in round_meta:
                print(f"Remote messages: {len(round_meta['remote']['messages'])}")

    # Convert output to JSON-serializable format
    if "local_usage" in output and "remote_usage" in output:
        # Convert Usage objects to dictionaries
        output["local_usage"] = output["local_usage"].to_dict()
        output["remote_usage"] = output["remote_usage"].to_dict()

    # save the output to a file
    with open("/Users/biderman/Dropbox/Stanford/Dan/minions/test_output.json", "w") as f:
        json.dump(output, f)

if __name__ == "__main__":
    main() 