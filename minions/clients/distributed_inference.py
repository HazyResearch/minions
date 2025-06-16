"""
Distributed Inference client for Minions.
Supports both direct node access and network coordinator routing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from urllib.parse import quote
import re
import json

from minions.usage import Usage
from minions.clients.base import MinionsClient


class DistributedInferenceClient(MinionsClient):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "http://localhost:8080",
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize the Distributed Inference client.

        Args:
            model_name: Preferred model name (optional, coordinator will select if not specified)
            api_key: API key for network coordinator authentication (optional)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            base_url: Network coordinator URL (default: "http://localhost:8080")
            timeout: Request timeout in seconds (default: 30)
            **kwargs: Additional parameters passed to base class
        """
        # For distributed inference, model_name is optional
        super().__init__(
            model_name=model_name or "auto",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        # Client-specific configuration
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key or os.getenv("MINIONS_API_KEY")
        self.timeout = timeout
        
        # Set up headers for authenticated requests
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Log initialization details
        self.logger.info(f"DistributedInferenceClient initialized:")
        self.logger.info(f"  - Base URL: {self.base_url}")
        self.logger.info(f"  - Model name: {self.model_name}")
        self.logger.info(f"  - Temperature: {self.temperature}")
        self.logger.info(f"  - Max tokens: {self.max_tokens}")
        self.logger.info(f"  - Timeout: {self.timeout}")
        self.logger.info(f"  - API key present: {bool(self.api_key)}")
        self.logger.info(f"  - Headers: {self.headers}")

    def _convert_python_to_json(self, response_text: str) -> str:
        """
        Convert Python object syntax to JSON format.
        
        If the response looks like:
            JobOutput(explanation="...", citation=None, answer="...")
        
        Convert it to:
            {"explanation": "...", "citation": null, "answer": "..."}
        """
        response_text = response_text.strip()
        
        # Check if this looks like Python object syntax
        if response_text.startswith(('JobOutput(', 'class JobOutput')):
            try:
                self.logger.info(f"DistributedInferenceClient: Converting Python syntax to JSON")
                
                # Handle class definition - return error JSON
                if response_text.startswith('class JobOutput'):
                    self.logger.warning(f"DistributedInferenceClient: Model returned class definition instead of instance")
                    return json.dumps({
                        "explanation": "Model returned class definition instead of instance",
                        "citation": None,
                        "answer": None
                    })
                
                # Try to extract JobOutput(...) content
                import re
                match = re.search(r'JobOutput\s*\(\s*(.*?)\s*\)$', response_text, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    self.logger.info(f"DistributedInferenceClient: Extracted content: {repr(content[:100])}")
                    
                    # Parse the key-value pairs
                    result = {}
                    
                    # Split by commas that are not inside quotes
                    # Use a more robust approach to handle nested quotes
                    parts = []
                    current_part = ""
                    paren_depth = 0
                    quote_char = None
                    
                    for char in content:
                        if quote_char is None:
                            if char in ['"', "'"]:
                                quote_char = char
                            elif char == '(' :
                                paren_depth += 1
                            elif char == ')':
                                paren_depth -= 1
                            elif char == ',' and paren_depth == 0:
                                parts.append(current_part.strip())
                                current_part = ""
                                continue
                        else:
                            if char == quote_char:
                                quote_char = None
                        
                        current_part += char
                    
                    if current_part.strip():
                        parts.append(current_part.strip())
                    
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Convert Python None to JSON null
                            if value == 'None':
                                result[key] = None
                            # Handle quoted strings - remove outer quotes
                            elif (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                                result[key] = value[1:-1]
                            # Handle other values
                            else:
                                # Try to parse as string, handling edge cases
                                if value != 'None':
                                    result[key] = value
                                else:
                                    result[key] = None
                    
                    converted_json = json.dumps(result)
                    self.logger.info(f"DistributedInferenceClient: Converted to JSON: {repr(converted_json[:100])}")
                    return converted_json
                    
            except Exception as e:
                self.logger.warning(f"DistributedInferenceClient: Failed to convert Python syntax to JSON: {e}")
                # Return a safe error response
                return json.dumps({
                    "explanation": f"Failed to parse model response: {str(e)}",
                    "citation": None,
                    "answer": None
                })
        
        # If it doesn't look like Python syntax, return as-is
        return response_text

    def _clean_markdown_response(self, response_text: str) -> str:
        """
        Clean markdown code blocks from response text.
        
        Distributed inference API sometimes wraps JSON in markdown code blocks like:
        ```json
        {...}
        ```
        or
        ```
        {...}
        ```
        
        This method strips those markdown wrappers to get clean JSON.
        """
        # Remove leading/trailing whitespace
        cleaned = response_text.strip()
        
        # Pattern to match markdown code blocks with optional language identifier
        # Matches: ```json\n{...}\n``` or ```\n{...}\n```
        code_block_pattern = r'^```(?:json)?\s*\n(.*?)\n```$'
        
        match = re.match(code_block_pattern, cleaned, re.DOTALL)
        if match:
            # Extract the content inside the code blocks
            return match.group(1).strip()
        
        # If no code blocks found, return original
        return cleaned

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        self.logger.info(f"DistributedInferenceClient: Making {method} request to {url}")
        
        try:
            kwargs.setdefault("timeout", self.timeout)
            kwargs.setdefault("headers", {}).update(self.headers)
            
            response = requests.request(method, url, **kwargs)
            
            self.logger.info(f"DistributedInferenceClient: Response received - Status: {response.status_code}")
            self.logger.info(f"DistributedInferenceClient: Response content length: {len(response.content)} bytes")
            
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            self.logger.error(f"DistributedInferenceClient: Request timed out after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DistributedInferenceClient: Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"DistributedInferenceClient: Error response status: {e.response.status_code}")
                self.logger.error(f"DistributedInferenceClient: Error response text: {e.response.text}")
            raise

    def get_network_status(self) -> Dict[str, Any]:
        """Get network health status from coordinator."""
        url = f"{self.base_url}/health"
        response = self._make_request("GET", url)
        return response.json()

    def list_nodes(self) -> Dict[str, Any]:
        """List all nodes and their capabilities."""
        url = f"{self.base_url}/nodes"
        response = self._make_request("GET", url)
        return response.json()

    def register_node(self, node_url: str) -> Dict[str, Any]:
        """Register a new node with the network coordinator."""
        url = f"{self.base_url}/nodes"
        data = {"node_url": node_url}
        response = self._make_request("POST", url, json=data)
        return response.json()

    def remove_node(self, node_url: str) -> Dict[str, Any]:
        """Remove a node from the network."""
        encoded_url = quote(node_url, safe='')
        url = f"{self.base_url}/nodes/{encoded_url}"
        response = self._make_request("DELETE", url)
        return response.json()

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions using the Distributed Inference API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments (not used by this API)

        Returns:
            Tuple of (List[str], Usage, List[str]) containing response strings, token usage, and done reasons
        """
        assert len(messages) > 0, "Messages cannot be empty."
        
        self.logger.info(f"DistributedInferenceClient: chat() called with {len(messages)} messages")
        # Log message summary instead of full content
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            self.logger.info(f"DistributedInferenceClient: Message {i} ({msg.get('role', 'unknown')}): {repr(content_preview)} (length: {len(content)})")
        
        # Check if we have multiple messages - use batch endpoint for parallel processing
        if len(messages) > 1:
            self.logger.info(f"DistributedInferenceClient: Using batch chat for {len(messages)} messages")
            return self._batch_chat(messages, **kwargs)
        else:
            self.logger.info(f"DistributedInferenceClient: Using single chat for 1 message")
            return self._single_chat(messages[0], **kwargs)
    
    def _single_chat(self, message: Dict[str, Any], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Handle single message chat using the /chat endpoint."""
        # For distributed inference API, we need to extract the content since
        # the API only accepts a simple query parameter
        query = message.get("content", "")
        
        self.logger.info(f"DistributedInferenceClient: Starting single chat request")
        query_preview = query[:100] + "..." if len(query) > 100 else query
        self.logger.info(f"DistributedInferenceClient: Input message role: {message.get('role', 'unknown')}")
        self.logger.info(f"DistributedInferenceClient: Query preview: {repr(query_preview)} (length: {len(query)})")
        self.logger.info(f"DistributedInferenceClient: Base URL: {self.base_url}")
        self.logger.info(f"DistributedInferenceClient: Model name: {self.model_name}")
        
        try:
            # Use network coordinator
            url = f"{self.base_url}/chat"
            params = {"query": query}
            
            # Add model preference if specified and not "auto"
            if self.model_name and self.model_name != "auto":
                params["model"] = self.model_name
            
            self.logger.info(f"DistributedInferenceClient: Request URL: {url}")
            self.logger.info(f"DistributedInferenceClient: Request params: {params}")
            self.logger.info(f"DistributedInferenceClient: Request headers: {self.headers}")
            
            response = self._make_request("POST", url, params=params)
            
            self.logger.info(f"DistributedInferenceClient: Response status: {response.status_code}")
            self.logger.info(f"DistributedInferenceClient: Response headers: {dict(response.headers)}")
            
            raw_response_text = response.text
            response_preview = raw_response_text[:200] + "..." if len(raw_response_text) > 200 else raw_response_text
            self.logger.info(f"DistributedInferenceClient: Raw response text preview: {repr(response_preview)} (length: {len(raw_response_text)})")
            
            data = response.json()
            self.logger.info(f"DistributedInferenceClient: Response contains - response: {bool(data.get('response'))}, usage: {bool(data.get('usage'))}, node_url: {data.get('node_url', 'N/A')}")
            
            # Log which node was used
            if "node_url" in data:
                self.logger.info(f"Request routed to node: {data['node_url']}")
            if "model_used" in data:
                self.logger.info(f"Model used: {data['model_used']}")
            
            # Extract response and clean any markdown formatting
            raw_response_text = data.get("response", "")
            response_preview = raw_response_text[:100] + "..." if len(raw_response_text) > 100 else raw_response_text
            self.logger.info(f"DistributedInferenceClient: Raw response from API: {repr(response_preview)} (length: {len(raw_response_text)})")
            
            # Convert Python syntax to JSON if needed
            converted_response = self._convert_python_to_json(raw_response_text)
            
            # Then clean any markdown formatting
            response_text = self._clean_markdown_response(converted_response)
            cleaned_preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            self.logger.info(f"DistributedInferenceClient: Final response: {repr(cleaned_preview)} (length: {len(response_text)})")
            
            # Extract usage information
            usage_data = data.get("usage", {})
            self.logger.info(f"DistributedInferenceClient: Usage data: {usage_data}")
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0)
            )
            
            # Extract done reason if available, default to "stop"
            done_reason = data.get("done_reason", "stop")
            self.logger.info(f"DistributedInferenceClient: Done reason: {done_reason}")
            
            result = ([response_text], usage, [done_reason])
            self.logger.info(f"DistributedInferenceClient: Final result structure: {result}")
            self.logger.info(f"DistributedInferenceClient: Response list length: {len(result[0])}")
            
            # Validate we have at least one response
            if not result[0] or not result[0][0]:
                self.logger.error(f"DistributedInferenceClient: Empty response received from API")
                # Return a minimal JSON error response that can be parsed as JobOutput
                error_response = json.dumps({
                    "explanation": "No response received from distributed inference network",
                    "citation": None,
                    "answer": None
                })
                return ([error_response], Usage(), ["stop"])
            
            return result
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"DistributedInferenceClient: HTTP Error - Status: {e.response.status_code}")
            self.logger.error(f"DistributedInferenceClient: HTTP Error - Response: {e.response.text}")
            if e.response.status_code == 404:
                self.logger.error("No nodes available with requested model")
            elif e.response.status_code == 503:
                self.logger.error("No healthy nodes available")
            elif e.response.status_code == 504:
                self.logger.error("Request to node timed out")
            elif e.response.status_code == 401:
                self.logger.error("Authentication failed - check API key")
            raise
        except Exception as e:
            self.logger.error(f"DistributedInferenceClient: Unexpected error during API call: {e}")
            self.logger.error(f"DistributedInferenceClient: Error type: {type(e)}")
            import traceback
            self.logger.error(f"DistributedInferenceClient: Traceback: {traceback.format_exc()}")
            raise
    
    def _batch_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Handle multiple messages using the /batch endpoint for parallel processing."""
        self.logger.info(f"DistributedInferenceClient: Starting batch chat request with {len(messages)} messages")
        
        # Extract queries from messages
        queries = []
        total_length = 0
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                # Single user message - use it directly (this preserves full Minions formatting)
                query = message.get("content", "")
                queries.append(query)
                query_preview = query[:100] + "..." if len(query) > 100 else query
                self.logger.info(f"DistributedInferenceClient: Message {i} (user): {repr(query_preview)} (length: {len(query)})")
                total_length += len(query)
            else:
                # For non-user messages, create a simple representation
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "system":
                    query = f"System: {content}"
                elif role == "assistant":
                    query = f"Assistant: {content}"
                else:
                    query = content
                queries.append(query)
                query_preview = query[:100] + "..." if len(query) > 100 else query
                self.logger.info(f"DistributedInferenceClient: Message {i} ({role}): {repr(query_preview)} (length: {len(query)})")
                total_length += len(query)
        
        self.logger.info(f"DistributedInferenceClient: Total queries: {len(queries)}, Total content length: {total_length} chars")
        
        try:
            # Use batch endpoint for parallel processing
            url = f"{self.base_url}/batch"
            
            # Prepare batch request
            batch_request = {"queries": queries}
            
            # Add model preference if specified and not "auto"
            if self.model_name and self.model_name != "auto":
                batch_request["model"] = self.model_name
            
            # Optional: Configure concurrency per node (default is 2, max is 10)
            batch_request["max_concurrent_per_node"] = min(len(queries), 5)  # Balance speed vs resource usage
            
            self.logger.info(f"DistributedInferenceClient: Batch request - {len(queries)} queries, model: {batch_request.get('model', 'auto')}, max_concurrent: {batch_request['max_concurrent_per_node']}")
            
            # Submit batch request
            response = self._make_request("POST", url, json=batch_request)
            initial_data = response.json()
            
            batch_id = initial_data.get("batch_id")
            if batch_id is None:
                raise ValueError(f"No batch_id returned from server: {initial_data}")
            
            batch_status = initial_data.get("status", "processing")
            self.logger.info(f"DistributedInferenceClient: Batch submitted with ID: {batch_id}, status: {batch_status}")
            
            # Poll for completion
            poll_url = f"{self.base_url}/batch/{batch_id}"
            max_polls = 300  # Max 5 minutes at 1-second intervals
            poll_count = 0
            
            import time
            
            while poll_count < max_polls:
                poll_count += 1
                time.sleep(1)  # Wait 1 second between polls
                
                poll_response = self._make_request("GET", poll_url)
                status_data = poll_response.json()
                
                status = status_data.get("status", "unknown")
                completed = status_data.get("completed", 0)
                failed = status_data.get("failed", 0)
                total = status_data.get("total_queries", len(queries))
                
                if poll_count % 5 == 1:  # Log every 5th poll
                    self.logger.info(f"DistributedInferenceClient: Batch {batch_id} status: {status}, completed: {completed}/{total}, failed: {failed}")
                
                if status == "completed":
                    self.logger.info(f"DistributedInferenceClient: Batch {batch_id} completed successfully")
                    break
                elif status == "failed":
                    self.logger.error(f"DistributedInferenceClient: Batch {batch_id} failed")
                    raise RuntimeError(f"Batch processing failed: {status_data}")
                elif status != "processing":
                    self.logger.warning(f"DistributedInferenceClient: Unknown batch status: {status}")
            
            if poll_count >= max_polls:
                raise TimeoutError(f"Batch {batch_id} timed out after {poll_count} polls")
            
            # Process final results from the completed batch
            final_data = status_data
            
            self.logger.info(f"DistributedInferenceClient: Batch response - total: {final_data.get('total_queries', 0)}, completed: {final_data.get('completed', 0)}, failed: {final_data.get('failed', 0)}, results: {len(final_data.get('results', []))}")
            
            # Process results in order
            responses = []
            total_usage = Usage()
            done_reasons = []
            
            for i, result in enumerate(final_data.get("results", [])):
                self.logger.info(f"DistributedInferenceClient: Processing result {i}: success={result.get('success', False)}")
                
                if result.get("success", False):
                    # Clean markdown formatting from response
                    raw_response = result.get("response", "")
                    response_preview = raw_response[:100] + "..." if len(raw_response) > 100 else raw_response
                    self.logger.info(f"DistributedInferenceClient: Raw response for result {i}: {repr(response_preview)} (length: {len(raw_response)})")
                    
                    # Convert Python syntax to JSON if needed
                    converted_response = self._convert_python_to_json(raw_response)
                    
                    # Then clean any markdown formatting
                    cleaned_response = self._clean_markdown_response(converted_response)
                    cleaned_preview = cleaned_response[:100] + "..." if len(cleaned_response) > 100 else cleaned_response
                    self.logger.info(f"DistributedInferenceClient: Final response for result {i}: {repr(cleaned_preview)} (length: {len(cleaned_response)})")
                    
                    responses.append(cleaned_response)
                    
                    # Aggregate usage
                    usage_data = result.get("usage", {})
                    total_usage += Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0)
                    )
                    
                    # Collect done reason
                    done_reasons.append(result.get("done_reason", "stop"))
                else:
                    # Handle failed queries - return proper JSON for JobOutput parsing
                    error_msg = result.get("error", "Unknown error")
                    self.logger.warning(f"Query {i} failed: {error_msg}")
                    
                    # Return a JSON response that can be parsed as JobOutput
                    error_response = json.dumps({
                        "explanation": f"Error from distributed inference: {error_msg}",
                        "citation": None,
                        "answer": None
                    })
                    responses.append(error_response)
                    total_usage += Usage(prompt_tokens=0, completion_tokens=0)
                    done_reasons.append("stop")
            
            # Validate we have responses for all queries
            if len(responses) != len(queries):
                self.logger.warning(f"DistributedInferenceClient: Response count mismatch - expected {len(queries)}, got {len(responses)}")
                # Pad with proper JSON error messages if needed
                while len(responses) < len(queries):
                    error_response = json.dumps({
                        "explanation": "No response received from distributed inference network",
                        "citation": None,
                        "answer": None
                    })
                    responses.append(error_response)
                    done_reasons.append("stop")
            
            # Ensure we never return empty response list
            if not responses:
                self.logger.error(f"DistributedInferenceClient: No responses received for batch of {len(queries)} queries")
                error_response = json.dumps({
                    "explanation": "No response received from distributed inference network",
                    "citation": None,
                    "answer": None
                })
                responses = [error_response] * len(queries)
                done_reasons = ["stop"] * len(queries)
            
            self.logger.info(f"DistributedInferenceClient: Final batch result summary - responses: {len(responses)}, total_usage: {total_usage}")
            
            return (responses, total_usage, done_reasons)
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"DistributedInferenceClient: Batch HTTP Error - Status: {e.response.status_code}")
            self.logger.error(f"DistributedInferenceClient: Batch HTTP Error - Response: {e.response.text}")
            if e.response.status_code == 404:
                self.logger.error("No nodes available with requested model")
            elif e.response.status_code == 503:
                self.logger.error("No healthy nodes available")
            elif e.response.status_code == 401:
                self.logger.error("Authentication failed - check API key")
            raise
        except Exception as e:
            self.logger.error(f"DistributedInferenceClient: Unexpected error during batch API call: {e}")
            self.logger.error(f"DistributedInferenceClient: Error type: {type(e)}")
            import traceback
            self.logger.error(f"DistributedInferenceClient: Traceback: {traceback.format_exc()}")
            raise

    def embed(self, content: Any, **kwargs) -> List[List[float]]:
        """Embedding not supported by Distributed Inference API."""
        raise NotImplementedError("Embedding not supported by Distributed Inference API")

    def complete(self, prompts: Any, **kwargs) -> Tuple[List[str], Usage]:
        """
        Text completion using the chat endpoint.
        
        Args:
            prompts: Single prompt or list of prompts
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (List[str], Usage) containing completions and token usage
        """
        # Convert prompts to messages format
        if isinstance(prompts, str):
            prompts = [prompts]
        
        responses = []
        total_usage = Usage()
        
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            response, usage, _ = self.chat(messages, **kwargs)  # Ignore done_reasons
            responses.extend(response)
            total_usage += usage
        
        return responses, total_usage 