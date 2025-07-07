#!/usr/bin/env python3
"""
Stdin/stdout interface for Minion protocols.
Reads JSON from stdin, instantiates the appropriate protocol, and writes results to stdout.
"""

import json
import sys
import os
import logging
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Save original stdout
_original_stdout = sys.stdout

# Suppress ALL warnings and verbose output completely
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Redirect all logging to stderr with ERROR level only
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

# Monkey patch print to redirect to stderr
original_print = print
def print(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return original_print(*args, **kwargs)

# Temporarily redirect stdout to stderr during imports to capture any print statements
class StdoutToStderr:
    def __init__(self):
        self.stderr = sys.stderr
    
    def write(self, data):
        if data.strip():  # Only write non-empty data
            self.stderr.write(data)
    
    def flush(self):
        self.stderr.flush()

# Replace stdout during imports
sys.stdout = StdoutToStderr()

# Import minion protocols
from minions.minion import Minion
from minions.minions import Minions

# Import clients
from minions.clients.docker_model_runner import DockerModelRunnerClient
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.lemonade import LemonadeClient

# Restore original stdout after imports
sys.stdout = _original_stdout

def check_and_pull_ollama_model(model_name: str) -> bool:
    """Check if Ollama model exists and pull it if not."""
    import subprocess
    
    try:
        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0 and model_name in result.stdout:
            print(f"Model {model_name} already exists", file=sys.stderr)
            return True
        
        # Model doesn't exist, try to pull it
        print(f"Model {model_name} not found, pulling...", file=sys.stderr)
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"Successfully pulled model {model_name}", file=sys.stderr)
            return True
        else:
            print(f"Failed to pull model {model_name}: {result.stderr}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"Error checking/pulling Ollama model: {e}", file=sys.stderr)
        return False

def create_client(client_config: Dict[str, Any]):
    """Create a client based on configuration."""
    client_type = client_config.get("type", "docker_model_runner")
    
    if client_type == "docker_model_runner":
        return DockerModelRunnerClient(
            model_name=client_config.get("model_name", "ai/llama3.2:3B-Q4_0"),
            port=client_config.get("port", 12434),
            timeout=client_config.get("timeout", 60),
            **client_config.get("kwargs", {})
        )
    elif client_type == "ollama":
        model_name = client_config.get("model_name", "llama3.2:3b")
        
        # Check and pull model if needed
        if not check_and_pull_ollama_model(model_name):
            print(f"Warning: Could not ensure model {model_name} is available", file=sys.stderr)
        
        return OllamaClient(
            model_name=model_name,
            **client_config.get("kwargs", {})
        )
    elif client_type == "openai":
        return OpenAIClient(
            model_name=client_config.get("model_name", "gpt-4o"),
            **client_config.get("kwargs", {})
        )
    elif client_type == "anthropic":
        return AnthropicClient(
            model_name=client_config.get("model_name", "claude-3-5-sonnet-20241022"),
            **client_config.get("kwargs", {})
        )
    elif client_type == "lemonade":
        return LemonadeClient(
            model_name=client_config.get("model_name", "llama3.2:3b"),
            **client_config.get("kwargs", {})
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def create_protocol(protocol_config: Dict[str, Any], local_client, remote_client):
    """Create a protocol instance based on configuration."""
    protocol_type = protocol_config.get("type", "minion")
    
    if protocol_type == "minion":
        return Minion(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=protocol_config.get("max_rounds", 3),
            callback=protocol_config.get("callback"),
            log_dir=protocol_config.get("log_dir", "minion_logs"),
            **protocol_config.get("kwargs", {})
        )
    elif protocol_type == "minions":
        return Minions(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=protocol_config.get("max_rounds", 3),
            callback=protocol_config.get("callback"),
            log_dir=protocol_config.get("log_dir", "minion_logs"),
            **protocol_config.get("kwargs", {})
        )
    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")

def main():
    """Main function to handle stdin/stdout interface."""
    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Extract configuration
        local_client_config = input_data.get("local_client", {"type": "docker_model_runner"})
        remote_client_config = input_data.get("remote_client", {"type": "openai"})
        protocol_config = input_data.get("protocol", {"type": "minion"})
        
        # Extract call parameters
        call_params = input_data.get("call_params", {})
        
        # Temporarily redirect stdout to stderr during execution
        sys.stdout = StdoutToStderr()
        
        try:
            # Create clients
            local_client = create_client(local_client_config)
            remote_client = create_client(remote_client_config)
            
            # Create protocol
            protocol = create_protocol(protocol_config, local_client, remote_client)
            
            # Execute the protocol
            result = protocol(**call_params)
        finally:
            # Restore stdout
            sys.stdout = _original_stdout
        
        # Output result as JSON
        output = {
            "success": True,
            "result": result,
            "error": None
        }
        
        # Convert any non-serializable objects to strings
        def json_serializable(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return {k: json_serializable(v) for k, v in obj.__dict__.items()}
            else:
                return str(obj)
        
        # Use original stdout directly for final JSON output
        _original_stdout.write(json.dumps(output, default=json_serializable, indent=2))
        _original_stdout.flush()
        
    except Exception as e:
        # Output error as JSON
        output = {
            "success": False,
            "result": None,
            "error": str(e)
        }
        # Use original stdout directly for error JSON output
        _original_stdout.write(json.dumps(output, indent=2))
        _original_stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main() 