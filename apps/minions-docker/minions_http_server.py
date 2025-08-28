#!/usr/bin/env python3
"""
Flask HTTP server for the full Minions protocol with complete functionality including
retrieval, multiple LLM providers, and advanced processing capabilities.
"""

import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel

# Try to import PyMuPDF first, fallback to pypdf
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_LIBRARY = 'pymupdf'
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_LIBRARY = 'pypdf'
        fitz = None
    except ImportError:
        PDF_LIBRARY = None
        fitz = None

# Import the full Minions class and core clients
from minions.minions import Minions
from minions.clients.docker_model_runner import DockerModelRunnerClient
from minions.clients.openai import OpenAIClient

# no retrieval support
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
            "minions_initialized": minions_instance is not None
        },
        "features": {
            "bm25_retrieval_available": BM25_AVAILABLE,
            "retrieval_available": config["use_retrieval"] != "false" and BM25_AVAILABLE,
            "openai_available": bool(config["openai_api_key"]),
            "docker_available": True
        }
    })


def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file using PyMuPDF or pypdf as fallback."""
    try:
        if PDF_LIBRARY == 'pymupdf' and fitz:
            # Use PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
                if page_num < len(doc) - 1:
                    text += "\n\n"  # Page separator
            doc.close()
            return text
        elif PDF_LIBRARY == 'pypdf':
            # Use pypdf as fallback
            import io
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += page.extract_text()
                if page_num < len(reader.pages) - 1:
                    text += "\n\n"  # Page separator
            return text
        else:
            logger.error("No PDF processing library available")
            return None
    except Exception as e:
        logger.error(f"Error processing PDF with {PDF_LIBRARY}: {str(e)}")
        return None

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
        
        # Validate that context is not empty and contains meaningful content
        if not context or all(not str(item).strip() for item in context):
            return jsonify({
                "error": "Context is required",
                "message": "Context is mandatory for the minions protocol. Please upload a PDF document or provide text content in the context field.",
                "suggestion": "Upload a PDF file or enter text content to analyze"
            }), 400
        
        # Use environment variables as the ONLY configuration source
        # All parameters come from environment variables (Docker Compose)
        max_rounds = config["max_rounds"]
        max_jobs_per_round = config["max_jobs_per_round"]
        num_tasks_per_round = config["num_tasks_per_round"]
        num_samples_per_task = config["num_samples_per_task"]
        use_retrieval = None
        chunk_fn = config["chunking_function"]
        
        # Optional user parameters
        logging_id = data.get("logging_id")
        mcp_tools_info = data.get("mcp_tools_info")
        
        logger.info(f"Running minions with task: {task[:100]}...")
        logger.info(f"Context length: {len(context)} items")
        logger.info(f"Max rounds: {max_rounds}")
        logger.info(f"Use retrieval: {use_retrieval}")
        logger.info(f"Chunk function: {chunk_fn}")

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

@app.route('/remote-only', methods=['POST'])
def run_remote_only():
    """
    Run the task using only the remote model (no minions protocol).
    This demonstrates the cost difference compared to the minions approach.
    
    Expected JSON body: Same as /minions endpoint
    """
    global minions_instance
    
    try:
        # Initialize minions if not already done (we need the remote client)
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
                    "error": "Failed to initialize remote client",
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
        
        # Validate that context is not empty and contains meaningful content
        if not context or all(not str(item).strip() for item in context):
            return jsonify({
                "error": "Context is required",
                "message": "Context is mandatory for remote processing. Please upload a PDF document or provide text content in the context field.",
                "suggestion": "Upload a PDF file or enter text content to analyze"
            }), 400
        
        logger.info(f"Running remote-only processing with task: {task[:100]}...")
        logger.info(f"Context length: {len(context)} items")
        
        # Prepare the prompt for direct remote processing
        context_text = "\n\n".join(context) if context else ""
        
        # Create a comprehensive prompt that includes all the context
        prompt = f"""You are an AI assistant tasked with analyzing the following document and answering a specific question.

Document Type: {doc_metadata}

Document Content:
{context_text}

Task: {task}

Please provide a comprehensive and accurate answer based on the document content provided above. If the document doesn't contain enough information to fully answer the question, please indicate what information is missing."""

        # Run the remote model directly
        start_time = time.time()
        
        # Get the remote client from the minions instance
        remote_client = minions_instance.remote_client
        
        # Make the direct call to the remote model using the correct method
        response_texts, usage = remote_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            temperature=0.1
        )
        
        end_time = time.time()
        
        # Extract the answer from the response
        final_answer = response_texts[0] if response_texts else "No response generated"
        
        # Get actual usage information
        remote_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
        
        # Prepare response
        response_data = {
            "success": True,
            "final_answer": final_answer,
            "log_file": None,  # No log file for remote-only processing
            "usage": {
                "remote": remote_usage,
                "local": None  # No local usage in remote-only mode
            },
            "timing": {
                "total_time": end_time - start_time,
                "remote_time": end_time - start_time,
                "local_time": 0
            },
            "execution_time": end_time - start_time,
            "parameters_used": {
                "processing_mode": "remote_only",
                "max_rounds": 1,  # Only one round for remote-only
                "max_jobs_per_round": 1,
                "num_tasks_per_round": 1,
                "num_samples_per_task": 1,
                "use_retrieval": False,
                "retrieval_model": None,
                "chunk_fn": None
            }
        }
        
        logger.info(f"Remote-only processing completed successfully. Final answer length: {len(final_answer)}")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
        logger.info(f"Estimated token usage: {remote_usage['total_tokens']} tokens")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error running remote-only processing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to run remote-only processing",
            "message": str(e),
            "traceback": traceback.format_exc() if os.getenv("DEBUG", "false").lower() == "true" else None
        }), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    Upload and process a PDF file to extract text content.
    
    Expected: multipart/form-data with 'pdf' file field
    
    Returns:
    {
        "success": true,
        "text": "Extracted text content...",
        "filename": "document.pdf",
        "pages": 5,
        "characters": 1234
    }
    """
    try:
        # Check if file is present in request
        if 'pdf' not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "PDF file is required in 'pdf' field"
            }), 400
        
        file = request.files['pdf']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a PDF file"
            }), 400
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                "error": "Invalid file type",
                "message": "Only PDF files are supported"
            }), 400
        
        # Read file content
        pdf_bytes = file.read()
        
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(pdf_bytes) > max_size:
            return jsonify({
                "error": "File too large",
                "message": f"PDF file must be smaller than {max_size // (1024*1024)}MB"
            }), 400
        
        # Validate that it's actually a PDF
        if not pdf_bytes.startswith(b'%PDF'):
            return jsonify({
                "error": "Invalid PDF file",
                "message": "File does not appear to be a valid PDF"
            }), 400
        
        logger.info(f"Processing PDF upload: {file.filename} ({len(pdf_bytes)} bytes)")
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_bytes)
        
        if extracted_text is None:
            return jsonify({
                "error": "PDF processing failed",
                "message": "Could not extract text from PDF. The file may be corrupted or contain only images."
            }), 400
        
        # Count pages by opening the PDF again (for metadata)
        try:
            if PDF_LIBRARY == 'pymupdf' and fitz:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()
            elif PDF_LIBRARY == 'pypdf':
                import io
                pdf_file = io.BytesIO(pdf_bytes)
                reader = PdfReader(pdf_file)
                page_count = len(reader.pages)
            else:
                page_count = 0
        except:
            page_count = 0
        
        logger.info(f"PDF processed successfully: {len(extracted_text)} characters extracted from {page_count} pages")
        
        return jsonify({
            "success": True,
            "text": extracted_text,
            "filename": file.filename,
            "pages": page_count,
            "characters": len(extracted_text)
        })
        
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "PDF processing failed",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /minions",
            "POST /remote-only",
            "POST /upload-pdf"
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
    logger.info("  - Advanced Document Processing")
    logger.info("  - Configurable Chunking Strategies")
    
    app.run(
        host=host,
        port=port,
        debug=os.getenv("DEBUG", "false").lower() == "true"
    )
