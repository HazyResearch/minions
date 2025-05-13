from typing import List, Dict, Any
import os
import logging
import base64
import re

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
            from openai import OpenAI

            self.client = OpenAI(base_url=f"{endpoint}/v1", api_key="EMPTY")
            logger.info(f"✅ OpenAI client initialized with endpoint: {endpoint}/v1")
        except ImportError as e:
            logger.error(f"❌ Failed to import OpenAI: {str(e)}")
            logger.error("Please install OpenAI with: pip install openai")
            raise RuntimeError("OpenAI not installed")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
            raise RuntimeError(f"OpenAI client initialization failed: {str(e)}")

    def chat(self, messages, temperature=0.7, max_tokens=16384):
        """Process chat messages using SGLang."""
        try:
            # Convert messages to OpenAI format
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "user")

                # Handle image in message
                if "image_url" in msg:
                    content_list = [{"type": "text", "text": content}]

                    # Add image content
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{msg['image_url']}"
                            },
                        }
                    )

                    formatted_messages.append({"role": role, "content": content_list})
                else:
                    formatted_messages.append({"role": role, "content": content})

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=os.environ.get("SGLANG_MODEL", "google/gemma-3-27b-it"),
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Get token usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return [response_text], usage
        except Exception as e:
            logger.error(f"❌ Error in SGLang chat: {str(e)}")
            raise RuntimeError(f"SGLang chat failed: {str(e)}")

    def stream_chat(self, messages, temperature=0.7, max_tokens=16384):
        """Stream chat responses using SGLang."""
        try:
            # Convert messages to OpenAI format 
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "user")

                # Handle image in message
                if "image_url" in msg:
                    content_list = [{"type": "text", "text": content}]

                    # Add image content
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{msg['image_url']}"
                            },
                        }
                    )

                    formatted_messages.append({"role": role, "content": content_list})
                else:
                    formatted_messages.append({"role": role, "content": content})

            # Call the client with streaming enabled
            stream = self.client.chat.completions.create(
                model=os.environ.get("SGLANG_MODEL", "google/gemma-3-27b-it"),
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            # Return the stream object
            return stream
        except Exception as e:
            logger.error(f"❌ Error in SGLang stream chat: {str(e)}")
            raise RuntimeError(f"SGLang stream chat failed: {str(e)}")


def run_model(context: List[Dict[str, Any]]):
    """Run the SGLang model with the given context."""
    # Get the singleton instance of SGLangClient
    client = SGLangClient.get_instance()

    response, usage = client.chat(messages=context, temperature=0.7)

    # The response is already processed in the chat method
    return response[0]


# Function to initialize the model at server startup
def initialize_model():
    """Initialize the model at server startup."""
    endpoint = os.environ.get("SGLANG_ENDPOINT", "http://localhost:5000")
    model_path = os.environ.get("SGLANG_MODEL", "google/gemma-3-27b-it")
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
    """
    This function is kept for compatibility but now just returns the endpoint.
    The actual server should be started separately using the OpenAI-compatible API.
    """
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
            launch_cmd = f"python -m sglang.launch_server --model-path {model_path} --mem-fraction-static 0.6 --host {host} --port {port} --stream-interval {streaming_interval} --enable-multimodal --quantization fp8"

            logger.info(f"Launch command: {launch_cmd}")

            # Launch the server
            server_process, actual_port = launch_server_cmd(launch_cmd)

            # Wait for the server to start
            wait_for_server(f"http://{host}:{actual_port}")
            logger.info(
                f"✅ SGLang server started successfully at http://{host}:{actual_port}"
            )

            # Update the endpoint if the port changed
            if actual_port != port:
                new_endpoint = f"http://{host}:{actual_port}"
                logger.info(
                    f"⚠️ Port changed from {port} to {actual_port}. Updating endpoint to {new_endpoint}"
                )
                return new_endpoint

            return endpoint
    except ImportError as e:
        logger.error(f"❌ Failed to import SGLang: {str(e)}")
        logger.error("Please install SGLang with: pip install sglang")
        raise RuntimeError("SGLang not installed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SGLang server: {str(e)}")
        raise RuntimeError(f"SGLang server initialization failed: {str(e)}")
