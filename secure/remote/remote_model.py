from typing import List, Dict, Any
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SGLangClient:
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls, endpoint=None):
        """Get or create the singleton instance of SGLangClient."""
        if cls._instance is None:
            if endpoint is None:
                endpoint = os.environ.get("SGLANG_ENDPOINT", "http://localhost:5000")
            cls._instance = cls(endpoint)
        return cls._instance
    
    def __init__(self, endpoint="http://localhost:5000"):
        """Initialize a SGLang client with the specified endpoint."""
        self.endpoint = endpoint
        
        # Import SGLang
        try:
            from sglang import RuntimeEndpoint, set_default_backend
            self.backend = RuntimeEndpoint(endpoint)
            set_default_backend(self.backend)
            
            # Import necessary components for chat
            from sglang import gen, system, user, assistant, function
            self.gen = gen
            self.system = system
            self.user = user
            self.assistant = assistant
            self.function = function
            
            logger.info(f"✅ SGLang client initialized with endpoint: {endpoint}")
        except ImportError as e:
            logger.error(f"❌ Failed to import SGLang: {str(e)}")
            logger.error("Please install SGLang with: pip install sglang")
            raise RuntimeError("SGLang not installed")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SGLang client: {str(e)}")
            raise RuntimeError(f"SGLang client initialization failed: {str(e)}")
    
    def chat(self, messages, temperature=0.0, max_tokens=1024):
        """Process chat messages using SGLang."""
        try:
            # Define a function to handle the chat
            @self.function
            def chat_function(s):
                # Add system message if present
                system_messages = [msg for msg in messages if msg["role"] == "system"]
                if system_messages:
                    s += self.system(system_messages[0]["content"])
                
                # Add all user and assistant messages
                for msg in messages:
                    if msg["role"] == "user":
                        s += self.user(msg["content"])
                    elif msg["role"] == "assistant" and msg["content"]:
                        s += self.assistant(msg["content"])
                
                # Generate the response
                if messages[-1]["role"] != "assistant":
                    s += self.assistant(self.gen("response", max_tokens=max_tokens, temperature=temperature))
                print("s after gen")
                print(s)
            # Run the function
            state = chat_function.run()
            
            # Extract the full text from the state
            full_text = state.text()
            
            # Extract only the last model turn
            response_text = full_text
            if "<start_of_turn>model" in full_text:
                # Find the last model turn
                last_model_start = full_text.rindex("<start_of_turn>model")
                last_model_end = full_text.find("<end_of_turn>", last_model_start)
                
                if last_model_end > last_model_start:
                    # Extract the content between the markers, excluding the markers themselves
                    model_content_start = full_text.find("\n", last_model_start) + 1
                    response_text = full_text[model_content_start:last_model_end].strip()
            
            # Calculate token usage (approximate)
            # Note: SGLang doesn't provide token counts directly, so this is an estimation
            usage = {
                "prompt_tokens": 0,  # Placeholder
                "completion_tokens": 0,  # Placeholder
                "total_tokens": 0,  # Placeholder
            }
            
            return [response_text], usage
        except Exception as e:
            logger.error(f"❌ Error in SGLang chat: {str(e)}")
            raise RuntimeError(f"SGLang chat failed: {str(e)}")
    
    def stream_chat(self, messages, temperature=0.0, max_tokens=1024):
        """Stream chat responses using SGLang."""
        try:
            # Define a function to handle the chat
            @self.function
            def chat_function(s):
                # Add system message if present
                system_messages = [msg for msg in messages if msg["role"] == "system"]
                if system_messages:
                    s += self.system(system_messages[0]["content"])
                
                # Add all user and assistant messages
                for msg in messages:
                    if msg["role"] == "user":
                        s += self.user(msg["content"])
                    elif msg["role"] == "assistant" and msg["content"]:
                        s += self.assistant(msg["content"])
                
                # Generate the response
                if messages[-1]["role"] != "assistant":
                    s += self.assistant(self.gen("response", max_tokens=max_tokens, temperature=temperature))
            
            # Run the function with streaming enabled
            state = chat_function.run(stream=True)
            
            # Return the state for streaming
            return state
        except Exception as e:
            logger.error(f"❌ Error in SGLang stream chat: {str(e)}")
            raise RuntimeError(f"SGLang stream chat failed: {str(e)}")


def run_model(context: List[Dict[str, Any]]):
    """Run the SGLang model with the given context."""
    # Get the singleton instance of SGLangClient
    client = SGLangClient.get_instance()
    
    response, usage = client.chat(
        messages=context, temperature=0.0
    )
    
    # The response is already processed in the chat method
    return response[0]

# Function to initialize the model at server startup
def initialize_model():
    """Initialize the model at server startup."""
    endpoint = os.environ.get("SGLANG_ENDPOINT", "http://localhost:5000")
    model_path = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    streaming = os.environ.get("SGLANG_STREAMING", "false").lower() == "true"
    
    # Launch the SGLang server if needed
    print(f"🔄 Initializing SGLang server with model: {model_path}")
    new_endpoint = launch_sglang_server(model_path, endpoint, streaming)
    
    # Update the endpoint if it changed
    if new_endpoint != endpoint:
        os.environ["SGLANG_ENDPOINT"] = new_endpoint
        endpoint = new_endpoint
    
    # Initialize the SGLang client
    print(f"🔄 Initializing SGLang client with endpoint: {endpoint}")
    SGLangClient.get_instance(endpoint=endpoint)
    print(f"✅ SGLang client initialized successfully")

def launch_sglang_server(model_path, endpoint="http://localhost:5000", streaming=False):
    """Launch a SGLang server if one is not already running at the specified endpoint."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Import SGLang utilities
        from sglang.utils import launch_server_cmd, wait_for_server
        
        # Parse the endpoint to get port
        from urllib.parse import urlparse
        parsed_url = urlparse(endpoint)
        port = parsed_url.port or 5000
        host = parsed_url.hostname or "localhost"
        
        # Check if a server is already running at the endpoint
        try:
            import requests
            response = requests.get(f"{endpoint}/health")
            if response.status_code == 200:
                logger.info(f"✅ SGLang server already running at {endpoint}")
                return endpoint
            else:
                raise Exception("Server not healthy")
        except Exception:
            # Launch the server if not already running
            logger.info(f"🚀 Launching new SGLang server at {endpoint}")
            
            # Construct the launch command
            streaming_interval = "2" if streaming else "10"
            launch_cmd = f"python -m sglang.launch_server --model-path {model_path} --mem-fraction-static 0.6 --host {host} --port {port} --stream-interval {streaming_interval}"
            
            logger.info(f"Launch command: {launch_cmd}")
            
            # Launch the server
            server_process, actual_port = launch_server_cmd(launch_cmd)
            
            # Wait for the server to start
            wait_for_server(f"http://{host}:{actual_port}")
            logger.info(f"✅ SGLang server started successfully at http://{host}:{actual_port}")
            
            # Update the endpoint if the port changed
            if actual_port != port:
                new_endpoint = f"http://{host}:{actual_port}"
                logger.info(f"⚠️ Port changed from {port} to {actual_port}. Updating endpoint to {new_endpoint}")
                return new_endpoint
            
            return endpoint
    except ImportError as e:
        logger.error(f"❌ Failed to import SGLang: {str(e)}")
        logger.error("Please install SGLang with: pip install sglang")
        raise RuntimeError("SGLang not installed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SGLang server: {str(e)}")
        raise RuntimeError(f"SGLang server initialization failed: {str(e)}")
