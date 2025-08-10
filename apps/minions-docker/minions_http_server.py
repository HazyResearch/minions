#!/usr/bin/env python3
"""
Flask HTTP server for the full Minions protocol with complete functionality including
retrieval, multiple LLM providers, and advanced processing capabilities.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any, Optional, List, Union
import json
import traceback
import time
from datetime import datetime
from pydantic import BaseModel

# Import the full Minions class and core clients
from minions.minions import Minions
from minions.clients.docker_model_runner import DockerModelRunnerClient
from minions.clients.openai import OpenAIClient

# Import BM25 retrieval support
try:
    from minions.utils.retrievers import bm25_retrieve_top_k_chunks
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS with more permissive settings for development
CORS(app, origins=[
    "http://localhost:8080",
    "http://127.0.0.1:8080", 
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
])

# Global minions instance
minions_instance: Optional[Minions] = None

# Extended configuration for full functionality
config = {
    # Core settings
    "remote_model_name": os.getenv("REMOTE_MODEL_NAME", "gpt-4o-mini"),
    "local_model_name": os.getenv("LOCAL_MODEL_NAME", "ai/smollm2"),
    "local_base_url": os.getenv("LOCAL_BASE_URL", "http://model-runner.docker.internal/engines/llama.cpp/v1"),
    "remote_base_url": os.getenv("REMOTE_BASE_URL", "http://model-runner.docker.internal/engines/openai/v1"),
    "max_rounds": int(os.getenv("MAX_ROUNDS", "3")),
    "log_dir": os.getenv("LOG_DIR", "minion_logs"),
    "timeout": int(os.getenv("TIMEOUT", "60")),
    
    # API Keys
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    
    # Retrieval settings
    "use_retrieval": os.getenv("USE_RETRIEVAL", "false").lower(),
    "retrieval_model": os.getenv("RETRIEVAL_MODEL", "all-MiniLM-L6-v2"),
    
    # Advanced minions settings
    "max_jobs_per_round": int(os.getenv("MAX_JOBS_PER_ROUND", "2048")),
    "num_tasks_per_round": int(os.getenv("NUM_TASKS_PER_ROUND", "3")),
    "num_samples_per_task": int(os.getenv("NUM_SAMPLES_PER_TASK", "1")),
    "chunking_function": os.getenv("CHUNKING_FUNCTION", "chunk_by_section"),
    
    # Provider selection
    "remote_provider": os.getenv("REMOTE_PROVIDER", "openai"),  # openai, anthropic, together, groq, gemini, mistral
    "local_provider": os.getenv("LOCAL_PROVIDER", "docker"),   # docker, openai (for local deployment)
}

def create_client(provider: str, model_name: str, is_local: bool = False, structured_output_schema: Optional[BaseModel] = None) -> Any:
    """Create a client based on the provider type."""
    
    if provider == "openai":
        api_key = config["openai_api_key"]
        base_url = config["local_base_url"] if is_local else config["remote_base_url"]
        if not api_key and not is_local:
            raise ValueError("OpenAI API key not configured")
        return OpenAIClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            local=is_local
        )
        
    elif provider == "docker":
        return DockerModelRunnerClient(
            model_name=model_name,
            base_url=config["local_base_url"],
            timeout=config["timeout"],
            local=True,
            structured_output_schema=structured_output_schema
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_minions_instance() -> Minions:
    """Create and return a new minions instance with configured clients."""
    logger.info("Creating minions instance with configuration:")
    logger.info(f"  Remote provider: {config['remote_provider']}")
    logger.info(f"  Remote model: {config['remote_model_name']}")
    logger.info(f"  Local provider: {config['local_provider']}")
    logger.info(f"  Local model: {config['local_model_name']}")
    logger.info(f"  Max rounds: {config['max_rounds']}")
    logger.info(f"  Use retrieval: {config['use_retrieval']}")
    logger.info(f"  Retrieval model: {config['retrieval_model']}")
    
    class StructuredLocalOutput(BaseModel):
        explanation: str
        citation: str | None
        answer: str | None

    # Create local client
    local_client = create_client(
        provider=config["local_provider"],
        model_name=config["local_model_name"],
        is_local=True,
        structured_output_schema=StructuredLocalOutput
    )
    logger.info(f"Local client created: {local_client}")
    
    # Create remote client
    remote_client = create_client(
        provider=config["remote_provider"],
        model_name=config["remote_model_name"],
        is_local=False
    )
    logger.info(f"Remote client created: {remote_client}")
    
    # Prepare retrieval model if needed
    retrieval_model = None
    
    
    # Create minions instance
    minions = Minions(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=config["max_rounds"],
        log_dir=config["log_dir"],
        max_jobs_per_round=config["max_jobs_per_round"]
    )
    logger.info("Minions instance created successfully")
    
    return minions

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "remote_provider": config["remote_provider"],
            "remote_model_name": config["remote_model_name"],
            "local_provider": config["local_provider"],
            "local_model_name": config["local_model_name"],
            "max_rounds": config["max_rounds"],
            "use_retrieval": config["use_retrieval"],
            "retrieval_model": config["retrieval_model"],
            "minions_initialized": minions_instance is not None
        },
        "features": {
            "bm25_retrieval_available": BM25_AVAILABLE,
            "retrieval_available": config["use_retrieval"] != "false" and BM25_AVAILABLE,
            "openai_available": bool(config["openai_api_key"]),
            "docker_available": True
        }
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed server status and configuration."""
    return jsonify({
        "server": {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "minions_initialized": minions_instance is not None
        },
        "configuration": config,
        "available_providers": {
            "openai": True,
            "docker": True
        }
    })

def initialize_minions_on_startup():
    """Initialize minions instance on startup if possible."""
    global minions_instance
    
    # Check if we have at least one API key for remote provider
    has_remote_key = config["openai_api_key"] 
    
    
    if has_remote_key and not minions_instance:
        try:
            logger.info("Auto-initializing minions instance on startup...")
            minions_instance = create_minions_instance()
            logger.info("Minions instance initialized successfully on startup")
        except Exception as e:
            logger.error(f"Failed to auto-initialize minions on startup: {str(e)}")
            logger.error(traceback.format_exc())

@app.route('/minions', methods=['POST'])
def run_minions():
    """
    Run the full minions protocol on a given task and context.
    
    Expected JSON body:
    {
        "task": "What is the main topic of this document?",
        "doc_metadata": "Type of document being analyzed",
        "context": [
            "Document content here...",
            "More context information..."
        ],
        "max_rounds": 3,  // optional, overrides default
        "max_jobs_per_round": 2048,  // optional
        "num_tasks_per_round": 3,  // optional
        "num_samples_per_task": 1,  // optional
        "use_retrieval": "bm25",  // optional: null, "bm25", "embedding", "multimodal-embedding"
        "retrieval_model": "all-MiniLM-L6-v2",  // optional
        "chunk_fn": "chunk_by_section",  // optional
        "logging_id": "custom_task_id",  // optional
        "mcp_tools_info": null  // optional MCP tools information
    }
    """
    global minions_instance
    
    try:
        # Initialize minions if not already done
        if minions_instance is None:
            has_remote_key = config["openai_api_key"]
            
            if not has_remote_key:
                return jsonify({
                    "error": "No API key configured",
                    "message": "Set OPENAI_API_KEY environment variable"
                }), 400
            
            try:
                minions_instance = create_minions_instance()
            except Exception as e:
                return jsonify({
                    "error": "Failed to initialize minions",
                    "message": str(e)
                }), 500
        
        # Parse request body
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Invalid request",
                "message": "JSON body required"
            }), 400
        
        # Extract required parameters
        task = data.get("task")
        doc_metadata = data.get("doc_metadata", "Unknown document type")
        context = data.get("context", [])
        
        if not task:
            return jsonify({
                "error": "Missing required parameter",
                "message": "'task' is required"
            }), 400
        
        if not isinstance(context, list):
            return jsonify({
                "error": "Invalid parameter type",
                "message": "'context' must be a list of strings"
            }), 400
        
        # Extract optional parameters with defaults
        max_rounds = data.get("max_rounds", config["max_rounds"])
        max_jobs_per_round = data.get("max_jobs_per_round", config["max_jobs_per_round"])
        num_tasks_per_round = data.get("num_tasks_per_round", config["num_tasks_per_round"])
        num_samples_per_task = data.get("num_samples_per_task", config["num_samples_per_task"])
        use_retrieval = data.get("use_retrieval", config["use_retrieval"])
        retrieval_model = data.get("retrieval_model", config["retrieval_model"])
        chunk_fn = data.get("chunk_fn", config["chunking_function"])
        logging_id = data.get("logging_id")
        mcp_tools_info = data.get("mcp_tools_info")
        
        # Convert use_retrieval to proper format
        if use_retrieval == "false" or use_retrieval == False:
            use_retrieval = None
        elif use_retrieval == "true" or use_retrieval == True:
            use_retrieval = "embedding"
        
        logger.info(f"Running minions with task: {task[:100]}...")
        logger.info(f"Context length: {len(context)} items")
        logger.info(f"Max rounds: {max_rounds}")
        logger.info(f"Use retrieval: {use_retrieval}")
        logger.info(f"Retrieval model: {retrieval_model}")
        logger.info(f"Chunk function: {chunk_fn}")
        
        # Validate retrieval method if specified
        if use_retrieval and use_retrieval not in [None, "false", "bm25"]:
            return jsonify({
                "error": "Unsupported retrieval method",
                "message": f"Only 'bm25' retrieval is supported in this build. Got: '{use_retrieval}'"
            }), 400
        
        if use_retrieval == "bm25" and not BM25_AVAILABLE:
            return jsonify({
                "error": "BM25 retrieval not available",
                "message": "rank_bm25 package is not installed"
            }), 400
        
        # Run the minions protocol
        start_time = time.time()
        result = minions_instance(
            task=task,
            doc_metadata=doc_metadata,
            context=context,
            max_rounds=max_rounds,
            max_jobs_per_round=max_jobs_per_round,
            num_tasks_per_round=num_tasks_per_round,
            num_samples_per_task=num_samples_per_task,
            use_retrieval=use_retrieval,
            chunk_fn=chunk_fn,
            logging_id=logging_id,
            mcp_tools_info=mcp_tools_info
        )
        end_time = time.time()
        
        # Prepare response
        response = {
            "success": True,
            "final_answer": result["final_answer"],
            "log_file": result["log_file"],
            "usage": {
                "remote": result["remote_usage"],
                "local": result["local_usage"]
            },
            "timing": result["timing"],
            "execution_time": end_time - start_time,
            "parameters_used": {
                "max_rounds": max_rounds,
                "max_jobs_per_round": max_jobs_per_round,
                "num_tasks_per_round": num_tasks_per_round,
                "num_samples_per_task": num_samples_per_task,
                "use_retrieval": use_retrieval,
                "retrieval_model": retrieval_model,
                "chunk_fn": chunk_fn
            }
        }
        
        logger.info(f"Minions completed successfully. Final answer length: {len(result['final_answer'])}")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running minions: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to run minions protocol",
            "message": str(e),
            "traceback": traceback.format_exc() if os.getenv("DEBUG", "false").lower() == "true" else None
        }), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    # Hide sensitive API keys in response
    safe_config = config.copy()
    for key in safe_config:
        if "api_key" in key.lower() and safe_config[key]:
            safe_config[key] = "*" * 10
    
    return jsonify({
        "config": safe_config,
        "minions_initialized": minions_instance is not None,
        "available_chunking_functions": [
            "chunk_by_section",
            "chunk_by_page", 
            "chunk_by_paragraph",
            "chunk_by_code",
            "chunk_by_function_and_class"
        ],
        "available_retrieval_methods": [
            "bm25"
        ],
        "available_providers": {
            "local": ["docker", "openai"],
            "remote": ["openai", "anthropic", "together", "groq", "gemini", "mistral"]
        }
    })

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration and reinitialize minions if needed."""
    global config, minions_instance
    
    try:
        data = request.get_json() or {}
        config_changed = False
        
        # Update configuration
        for key in config.keys():
            if key in data:
                old_value = config[key]
                if key in ["max_rounds", "timeout", "max_jobs_per_round", "num_tasks_per_round", "num_samples_per_task"]:
                    config[key] = int(data[key])
                else:
                    config[key] = data[key]
                
                if config[key] != old_value:
                    config_changed = True
                    logger.info(f"Config updated: {key} = {config[key]}")
        
        # If provider-related config changed, reinitialize minions
        provider_keys = ["remote_provider", "local_provider", "remote_model_name", "local_model_name"]
        api_key_keys = [k for k in config.keys() if "api_key" in k]
        
        if any(key in data for key in provider_keys + api_key_keys) and minions_instance:
            logger.info("Provider configuration changed, reinitializing minions...")
            minions_instance = None
            try:
                minions_instance = create_minions_instance()
                logger.info("Minions reinitialized successfully")
            except Exception as e:
                logger.error(f"Failed to reinitialize minions: {e}")
                return jsonify({
                    "error": "Failed to reinitialize minions with new configuration",
                    "message": str(e)
                }), 500
        
        # Hide sensitive information in response
        safe_config = config.copy()
        for key in safe_config:
            if "api_key" in key.lower() and safe_config[key]:
                safe_config[key] = "*" * 10
        
        return jsonify({
            "message": "Configuration updated successfully",
            "config": safe_config,
            "minions_reinitialized": config_changed and minions_instance is not None
        })
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return jsonify({
            "error": "Failed to update configuration",
            "message": str(e)
        }), 500

@app.route('/providers', methods=['GET'])
def get_providers():
    """Get information about available providers and their status."""
    providers_status = {}
    
    # Check each provider
    providers_status["openai"] = {
        "available": True,
        "configured": bool(config.get("openai_api_key")),
        "models": []  # Could be expanded to list available models per provider
    }
    
    providers_status["docker"] = {
        "available": True,
        "configured": True,
        "models": []
    }
    
    return jsonify({
        "providers": providers_status,
        "current_selection": {
            "remote": config["remote_provider"],
            "local": config["local_provider"]
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "GET /status",
            "POST /minions",
            "GET /config",
            "POST /config",
            "GET /providers"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    logger.info("Starting Minions HTTP Server with Full Functionality...")
    
    # Validate configuration
    logger.info("Configuration:")
    for key, value in config.items():
        if "api_key" in key.lower():
            logger.info(f"  {key}: {'*' * 10 if value else 'Not set'}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Check available providers
    logger.info("Available providers:")
    logger.info(f"  OpenAI: Available")
     
    # Create log directory
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Try to initialize minions on startup
    initialize_minions_on_startup()
    
    # Start Flask app
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("Server features:")
    logger.info("  - Full Minions Protocol Support")
    logger.info("  - OpenAI and Docker Client Support")
    logger.info(f"  - BM25 Retrieval: {'Available' if BM25_AVAILABLE else 'Not Available'}")
    logger.info("  - Advanced Document Processing")
    logger.info("  - Configurable Chunking Strategies")
    
    app.run(
        host=host,
        port=port,
        debug=os.getenv("DEBUG", "false").lower() == "true"
    )
