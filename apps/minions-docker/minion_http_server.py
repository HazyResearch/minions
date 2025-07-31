#!/usr/bin/env python3
"""
Flask HTTP server for the Minion protocol using DockerModelRunner and OpenAI clients.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any, Optional, List
import json
import traceback

from minions.minion import Minion
from minions.clients.docker_model_runner import DockerModelRunnerClient
from minions.clients.openai import OpenAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, origins=[
    "http://localhost:8080",
    "http://127.0.0.1:8080", 
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
])

# Global minion instance
minion_instance: Optional[Minion] = None

# Configuration parameters
config = {
    "remote_model_name": os.getenv("REMOTE_MODEL_NAME", "gpt-4o-mini"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "local_model_name": os.getenv("LOCAL_MODEL_NAME", "ai/smollm2"),
    "local_base_url": os.getenv("LOCAL_BASE_URL", "http://model-runner.docker.internal/engines/llama.cpp/v1"),
    "remote_base_url": os.getenv("REMOTE_BASE_URL", "http://model-runner.docker.internal/engines/openai/v1"),
    "max_rounds": int(os.getenv("MAX_ROUNDS", "3")),
    "log_dir": os.getenv("LOG_DIR", "minion_logs"),
    "timeout": int(os.getenv("TIMEOUT", "60")),
}

def create_minion_instance() -> Minion:
    """Create and return a new minion instance with configured clients."""
    logger.info("Creating minion instance with configuration:")
    logger.info(f"  Remote model: {config['remote_model_name']}")
    logger.info(f"  Local model: {config['local_model_name']}")
    logger.info(f"  Local base URL: {config['local_base_url']}")
    logger.info(f"  Remote base URL: {config['remote_base_url']}")
    logger.info(f"  Max rounds: {config['max_rounds']}")
    
    # Create local client (DockerModelRunner)
    local_client = DockerModelRunnerClient(
        model_name=config["local_model_name"],
        base_url=config["local_base_url"],
        timeout=config["timeout"],
        local=True
    )
    logger.info(f"Local client created: {local_client}")
    
    # Create remote client (OpenAI)
    remote_client = OpenAIClient(
        model_name=config["remote_model_name"],
        api_key=config["openai_api_key"],
        base_url=config["remote_base_url"],
        local=False
    )
    logger.info(f"Remote client created: {remote_client}")
    
    # Create minion instance
    minion = Minion(
        local_client=local_client,
        remote_client=remote_client,
        max_rounds=config["max_rounds"],
        log_dir=config["log_dir"]
    )
    logger.info("Minion instance created successfully")
    
    return minion

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "config": {
            "remote_model_name": config["remote_model_name"],
            "local_model_name": config["local_model_name"],
            "local_base_url": config["local_base_url"],
            "remote_base_url": config["remote_base_url"],
            "max_rounds": config["max_rounds"],
            "minion_initialized": minion_instance is not None
        }
    })

def initialize_minion_on_startup():
    """Initialize minion instance on startup if API key is available."""
    global minion_instance
    
    if config["openai_api_key"] and not minion_instance:
        try:
            logger.info("Auto-initializing minion instance on startup...")
            minion_instance = create_minion_instance()
            logger.info("Minion instance initialized successfully on startup")
        except Exception as e:
            logger.error(f"Failed to auto-initialize minion on startup: {str(e)}")
            logger.error(traceback.format_exc())

@app.route('/run', methods=['POST'])
def run_minion():
    """
    Run the minion protocol on a given query and context.
    
    Expected JSON body:
    {
        "query": "What is the main topic of this document?",
        "context": [
            "Document content here...",
            "More context information..."
        ],
        "max_rounds": 3,  // optional, overrides default
        "logging_id": "custom_task_id",  // optional
        "is_privacy": false,  // optional
        "images": null  // optional
    }
    """
    global minion_instance
    
    try:
        # Initialize minion if not already done
        if minion_instance is None:
            if not config["openai_api_key"]:
                return jsonify({
                    "error": "OpenAI API key not configured",
                    "message": "Set OPENAI_API_KEY environment variable"
                }), 400
            
            try:
                minion_instance = create_minion_instance()
            except Exception as e:
                return jsonify({
                    "error": "Failed to initialize minion",
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
        query = data.get("query")
        context = data.get("context", [])
        
        if not query:
            return jsonify({
                "error": "Missing required parameter",
                "message": "'query' is required"
            }), 400
        
        if not isinstance(context, list):
            return jsonify({
                "error": "Invalid parameter type",
                "message": "'context' must be a list of strings"
            }), 400
        
        # Extract optional parameters
        max_rounds = data.get("max_rounds", config["max_rounds"])
        logging_id = data.get("logging_id")
        is_privacy = data.get("is_privacy", False)
        images = data.get("images")
        
        logger.info(f"Running minion with query: {query[:100]}...")
        logger.info(f"Context length: {len(context)} items")
        logger.info(f"Max rounds: {max_rounds}")
        logger.info(f"Privacy mode: {is_privacy}")
        
        # Run the minion protocol
        result = minion_instance(
            task=query,
            context=context,
            max_rounds=max_rounds,
            logging_id=logging_id,
            is_privacy=is_privacy,
            images=images
        )
        
        # Prepare response
        response = {
            "success": True,
            "final_answer": result["final_answer"],
            "log_file": result["log_file"],
            "usage": {
                "remote": result["remote_usage"].to_dict(),
                "local": result["local_usage"].to_dict()
            },
            "timing": result["timing"]
        }
        
        logger.info(f"Minion completed successfully. Final answer length: {len(result['final_answer'])}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running minion: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to run minion protocol",
            "message": str(e)
        }), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify({
        "config": config,
        "minion_initialized": minion_instance is not None
    })

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration without reinitializing minion."""
    global config
    
    try:
        data = request.get_json() or {}
        
        # Update configuration
        for key in config.keys():
            if key in data:
                if key in ["max_rounds", "timeout"]:
                    config[key] = int(data[key])
                else:
                    config[key] = data[key]
        
        return jsonify({
            "message": "Configuration updated successfully",
            "config": config
        })
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return jsonify({
            "error": "Failed to update configuration",
            "message": str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """
    List available models from the remote OpenAI client.
    
    Returns:
        JSON response containing the list of available models
    """
    global minion_instance
    
    try:
        # Check if minion is initialized
        if minion_instance is None:
            return jsonify({
                "error": "Minion protocol not initialized",
                "message": "Call /start_protocol first to initialize the minion"
            }), 400
        
        # Get models from the remote client
        models_data = minion_instance.remote_client.list_models()
        
        logger.info(f"Retrieved {len(models_data.get('data', []))} models from remote client")
        
        return jsonify(models_data)
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to list models",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /run",
            "GET /config",
            "POST /config",
            "GET /models"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    logger.info("Starting Minion HTTP Server...")
    
    # Validate required environment variables
    if not config["openai_api_key"]:
        logger.warning("OPENAI_API_KEY not set. Minion will be initialized on first request.")
    
    logger.info("Configuration:")
    for key, value in config.items():
        if "api_key" in key.lower():
            logger.info(f"  {key}: {'*' * 10 if value else 'Not set'}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Create log directory
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Try to initialize minion on startup
    initialize_minion_on_startup()
    
    # Start Flask app
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=os.getenv("DEBUG", "false").lower() == "true"
    )
