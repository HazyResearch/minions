from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from app import run_protocol, initialize_clients
import os
import time
from werkzeug.serving import WSGIRequestHandler
import fitz
import json
import re
import base64

app = Flask(__name__)

# Increase timeouts and buffer size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max-content-length
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Configure CORS to be more permissive for development
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://127.0.0.1:5173",  # Vite dev server
            "http://localhost:5173",
            "http://127.0.0.1:3000",  # Alternative React dev server
            "http://localhost:3000",
            # Add your production domain here when deploying
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "Content-Length"],
        "max_age": 3600
    }
})

def _extract_json(text: str) -> dict:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        raise

def _parse_json_response(response: str, max_attempts: int = 5) -> dict:
    """Parse JSON response with retries."""
    for attempt in range(max_attempts):
        try:
            return _extract_json(response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to get valid JSON response after {max_attempts} attempts")
            time.sleep(1)  # Wait before retrying

def format_together_model_name(model_name):
    """Format model name to ensure it follows the organization/model pattern."""
    if not model_name:
        return "deepseek-ai/DeepSeek-V3"  # Default model
    
    # Model name mapping for consistency
    model_mapping = {
        "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
        "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
        "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Meta-Llama-3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "QWQ-32B": "Qwen/QwQ-32B-Preview"
    }
    
    # If the model name is in our mapping, use that
    if model_name in model_mapping:
        return model_mapping[model_name]
    
    # If it already has a slash, assume it's properly formatted
    if '/' in model_name:
        return model_name
        
    # Try to guess the organization based on the model name
    if model_name.lower().startswith('deepseek'):
        return f"deepseek-ai/{model_name}"
    elif model_name.lower().startswith('llama'):
        return f"meta-llama/{model_name}"
    elif model_name.lower().startswith('qwen'):
        return f"Qwen/{model_name}"
    
    # If we can't determine the format, use the default
    print(f"Warning: Could not determine organization for model {model_name}, using default")
    return "deepseek-ai/DeepSeek-V3"

def calculate_context_window(context):
    """Calculate appropriate context window size based on input length."""
    if not context:
        return 4096
    
    # For Minion protocol, estimate tokens based on context length (4 chars ≈ 1 token)
    # Add 8000 padding tokens to account for the conversation history and safety margin
    padding = 8000
    estimated_tokens = int(len(context) / 4 + padding)
    
    # Round up to nearest power of 2 from predefined list
    num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # Find the smallest value that is >= estimated tokens
    num_ctx = min([x for x in num_ctx_values if x >= estimated_tokens], default=131072)
    print(f"Estimated tokens: {estimated_tokens}")
    print(f"Using context window: {num_ctx}")
    return num_ctx

def load_default_medical_context():
    try:
        with open("data/test_medical.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("Default medical context file not found!")
        return ""

def extract_text_from_file(file_path):
    """Extract text from a PDF, TXT, Python, or Markdown file."""
    try:
        # Expand ~ to user's home directory if present
        file_path = os.path.expanduser(file_path)

        if file_path.lower().endswith(".pdf"):
            # Handle PDF file
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        elif file_path.lower().endswith((".txt", ".py", ".md")):
            # Handle text-based files
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(
                "Unsupported file format. Only PDF, TXT, PY, and MD files are supported."
            )
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return ""

@app.route('/api/hello')
def hello():
    return jsonify({"message": "Hello, World!"})

@app.route('/api/run_protocol', methods=['POST', 'OPTIONS'])
def run_protocol_endpoint():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response

    start_time = time.time()
    try:
        data = request.json
        print("Received request data:", {
            **data,
            'remoteApiKey': '***' if 'remoteApiKey' in data else None,
            'context': data.get('context', '')[:100] + '...' if 'context' in data else None
        })

        # Validate required parameters
        required_params = ['task', 'context', 'protocol', 'localProvider', 'remoteProvider']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return jsonify({
                'error': f'Missing required parameters: {", ".join(missing_params)}'
            }), 400
        
        # Format context properly
        context = data.get('context', '')
        if isinstance(context, str):
            context = [context]
        elif isinstance(context, list):
            if not all(isinstance(c, str) for c in context):
                return jsonify({'error': 'All context items must be strings'}), 400
        else:
            return jsonify({'error': 'Context must be a string or an array of strings'}), 400

        # Create a simple status object that mimics Streamlit's status
        class Status:
            def __init__(self):
                self.container = lambda: self
                self.write = lambda x: print(f"Status: {x}")

        status = Status()
        
        # Get API key from environment or request
        api_key = os.environ.get('OPENAI_API_KEY') or data.get('remoteApiKey')
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        # Initialize clients with appropriate settings
        try:
            # Set default model based on provider
            provider = data.get('remoteProvider', "OpenAI")
            default_model = "gpt-4"
            
            # Handle Together.ai model names
            if provider == "Together":
                default_model = "deepseek-ai/DeepSeek-V3"
                # If a model name is provided, ensure it has the correct format
                if 'remoteModelName' in data:
                    model_name = data['remoteModelName']
                    remote_model_name = format_together_model_name(model_name)
                    print(f"Formatted Together.ai model name from '{model_name}' to '{remote_model_name}'")
                else:
                    remote_model_name = default_model
            else:
                remote_model_name = data.get('remoteModelName', default_model)
            
            print(f"Using remote model: {remote_model_name} for provider {provider}")
            
            # Calculate appropriate context window size
            # Join all context items with newlines for token calculation
            combined_context = '\n'.join(context)
            num_ctx = calculate_context_window(combined_context) if data['protocol'] == 'Minion' else 4096
            
            local_client, remote_client, method = initialize_clients(
                local_model_name=data.get('localModelName', "llama3.2"),
                remote_model_name=remote_model_name,
                provider=provider,
                local_provider=data.get('localProvider', "Ollama"),
                protocol=data.get('protocol', "Minion"),
                local_temperature=data.get('localTemperature', 0.0),
                local_max_tokens=data.get('localMaxTokens', 4096),
                remote_temperature=data.get('remoteTemperature', 0.2),
                remote_max_tokens=data.get('remoteMaxTokens', 2048),
                api_key=api_key,
                num_ctx=num_ctx,
                mcp_server_name=None,
                reasoning_effort=data.get('reasoningEffort', "medium")
            )
        except Exception as e:
            print("Error initializing clients:", str(e))
            return jsonify({'error': f'Failed to initialize clients: {str(e)}'}), 500
        
        # Run the protocol with the correct parameters based on protocol type
        try:
            # Base parameters common to all protocols
            params = {
                'task': data['task'],
                'context': context,  # Already a list, no need to wrap again
                'doc_metadata': data.get('doc_metadata', '')
            }
            
            if data['protocol'] == "Minion":
                # Add Minion-specific parameters
                params.update({
                    'max_rounds': 2,
                    'is_privacy': data.get('privacy_mode', False)
                })
                # Only add images if they exist
                if 'images' in data:
                    params['images'] = data['images']
            else:  # Minions or Minions-MCP
                # Add Minions-specific parameters
                params.update({
                    'max_rounds': 5,
                    'use_bm25': data.get('use_bm25', False)
                })

            # Execute the protocol with appropriate parameters
            output = method(**params)
            execution_time = time.time() - start_time

            # Build response data with all available information
            response_data = {
                'output': {
                    'final_answer': output['final_answer'],
                    'setup_time': 0,
                    'execution_time': execution_time
                }
            }

            # Add protocol-specific information
            if data['protocol'] == "Minion":
                if 'supervisor_messages' in output:
                    response_data['output']['supervisor_messages'] = output['supervisor_messages']
                if 'worker_messages' in output:
                    response_data['output']['worker_messages'] = output['worker_messages']
                if 'remote_usage' in output:
                    response_data['output']['remote_usage'] = output['remote_usage']
                if 'local_usage' in output:
                    response_data['output']['local_usage'] = output['local_usage']
                if 'log_file' in output:
                    response_data['output']['log_file'] = output['log_file']
            elif data['protocol'] in ["Minions", "Minions-MCP"]:
                if 'meta' in output:
                    response_data['output']['meta'] = output['meta']
                    # Add summary statistics
                    total_local_jobs = sum(len(round_meta['local']['jobs']) for round_meta in output['meta'] if 'local' in round_meta)
                    total_remote_messages = sum(len(round_meta['remote']['messages']) for round_meta in output['meta'] if 'remote' in round_meta)
                    response_data['output']['stats'] = {
                        'total_local_jobs': total_local_jobs,
                        'total_remote_messages': total_remote_messages
                    }
            
            response = jsonify(response_data)
            response.charset = 'utf-8' 
            response.headers.add('Content-Length', str(len(response.get_data())))
            return response

        except Exception as e:
            print("Error running protocol:", str(e))
            error_detail = str(e)
            if "API key" in error_detail:
                return jsonify({'error': 'Invalid or missing API key. Please check your API key and try again.'}), 401
            elif "rate limit" in error_detail.lower():
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            elif "model" in error_detail.lower():
                return jsonify({'error': f'Invalid model configuration: {error_detail}'}), 400
            else:
                return jsonify({'error': f'Protocol execution failed: {str(e)}'}), 500

    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    context = []
    file_metadata = []
    total_pages = 0

    for file in files:
        if file.filename == '':
            continue

        # Check file type
        if file.filename.lower().endswith('.pdf'):
            try:
                # Read PDF content
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                total_pages += len(pdf_document)
                
                if total_pages > 100:
                    return jsonify({'error': 'Total pages cannot exceed 100'}), 400

                # Extract text from each page
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    context.append(text)
                
                file_metadata.append(f"PDF: {file.filename} ({len(pdf_document)} pages)")
                pdf_document.close()
            except Exception as e:
                return jsonify({'error': f'Error processing PDF {file.filename}: {str(e)}'}), 400

        elif file.filename.lower().endswith('.txt'):
            try:
                # Read text content
                content = file.read().decode('utf-8')
                context.append(content)
                file_metadata.append(f"TXT: {file.filename}")
            except Exception as e:
                return jsonify({'error': f'Error processing text file {file.filename}: {str(e)}'}), 400

        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Convert image to base64
                image_data = file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                context.append(f"Image: {file.filename}")
                file_metadata.append(f"Image: {file.filename}")
            except Exception as e:
                return jsonify({'error': f'Error processing image {file.filename}: {str(e)}'}), 400

        else:
            return jsonify({'error': f'Unsupported file type: {file.filename}'}), 400

    if not context:
        return jsonify({'error': 'No valid content found in uploaded files'}), 400

    # Generate context description
    context_description = f"Uploaded {len(files)} file(s): {', '.join(file_metadata)}"

    return jsonify({
        'context': context,
        'file_metadata': '\n'.join(file_metadata),
        'context_description': context_description
    })

if __name__ == '__main__':
    # Enable debug mode and allow external access with increased timeouts
    app.run(
        host='0.0.0.0', 
        port=8080, 
        debug=True, 
        threaded=True, 
        processes=1,
        use_reloader=True,
        request_handler=WSGIRequestHandler
    ) 