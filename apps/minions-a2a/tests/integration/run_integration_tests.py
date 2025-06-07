#!/usr/bin/env python3
"""
Comprehensive integration test for A2A-Minions server.
Tests the entire system end-to-end without mocking.
"""

import sys
import os
import asyncio
import json
import time
import signal
import subprocess
import uuid
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
import httpx
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class A2AIntegrationTestClient:
    """Client for comprehensive integration testing."""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.api_key = None
        self.oauth_token = None
        self.oauth_client_id = None
        self.oauth_client_secret = None
    
    async def wait_for_server(self, timeout: int = 30):
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await self.client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("Server is ready")
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
        return False
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Get the public agent card."""
        response = await self.client.get(f"{self.base_url}/.well-known/agent.json")
        response.raise_for_status()
        return response.json()
    
    async def get_extended_agent_card(self, token: str) -> Dict[str, Any]:
        """Get the extended agent card with authentication."""
        headers = {"Authorization": f"Bearer {token}"}
        response = await self.client.get(
            f"{self.base_url}/agent/authenticatedExtendedCard",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    async def get_oauth_token(self, client_id: str, client_secret: str, 
                            scope: str = "tasks:read tasks:write minion:query minions:query") -> Dict[str, Any]:
        """Get OAuth2 access token."""
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope
        }
        response = await self.client.post(
            f"{self.base_url}/oauth/token",
            data=data
        )
        response.raise_for_status()
        return response.json()
    
    async def send_request(self, method: str, params: Dict[str, Any], 
                         auth_method: str = "api_key") -> Dict[str, Any]:
        """Send JSON-RPC request with authentication."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(uuid.uuid4())
        }
        
        headers = {"Content-Type": "application/json"}
        
        if auth_method == "api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif auth_method == "oauth" and self.oauth_token:
            headers["Authorization"] = f"Bearer {self.oauth_token}"
        
        response = await self.client.post(
            self.base_url,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    async def send_request_streaming(self, method: str, params: Dict[str, Any],
                                   auth_method: str = "api_key") -> List[Dict[str, Any]]:
        """Send JSON-RPC request with streaming response."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(uuid.uuid4())
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        if auth_method == "api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif auth_method == "oauth" and self.oauth_token:
            headers["Authorization"] = f"Bearer {self.oauth_token}"
        
        events = []
        async with self.client.stream("POST", self.base_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        event = json.loads(data)
                        events.append(event)
                        if event.get("result", {}).get("final", False):
                            break
                    except json.JSONDecodeError:
                        continue
        
        return events
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


class A2AIntegrationTest:
    """Comprehensive integration test suite."""
    
    def __init__(self):
        self.server_process = None
        self.client = A2AIntegrationTestClient()
        self.test_results = []
        self.temp_files = []
    
    def log_test(self, test_name: str, success: bool, message: str = "", details: Any = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        if details and not success:
            logger.debug(f"Details: {details}")
        self.test_results.append({
            "name": test_name,
            "success": success,
            "message": message,
            "details": details
        })
    
    async def start_server(self) -> bool:
        """Start the A2A-Minions server."""
        try:
            # Start server process
            env = os.environ.copy()
            env["PYTHONPATH"] = ":".join(sys.path)
            
            self.server_process = subprocess.Popen(
                [sys.executable, "run_server.py", "--port", "8888"],
                cwd=Path(__file__).parent.parent.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to be ready
            if await self.client.wait_for_server():
                self.log_test("Server Startup", True, "Server started successfully")
                return True
            else:
                self.log_test("Server Startup", False, "Server failed to start within timeout")
                return False
                
        except Exception as e:
            self.log_test("Server Startup", False, f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the A2A-Minions server."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
    
    async def test_health_check(self):
        """Test server health endpoint."""
        try:
            response = await self.client.client.get(f"{self.client.base_url}/health")
            data = response.json()
            
            success = (
                response.status_code == 200 and
                data.get("status") == "healthy" and
                "timestamp" in data and
                "tasks_count" in data
            )
            
            self.log_test("Health Check", success, f"Status: {data.get('status')}")
            return success
            
        except Exception as e:
            self.log_test("Health Check", False, str(e))
            return False
    
    async def test_agent_card(self):
        """Test agent card retrieval."""
        try:
            card = await self.client.get_agent_card()
            
            required_fields = ["name", "description", "url", "skills", "capabilities", 
                             "securitySchemes", "defaultInputModes", "defaultOutputModes"]
            
            success = all(field in card for field in required_fields)
            skills_valid = len(card.get("skills", [])) == 2  # Should have 2 skills
            
            self.log_test(
                "Agent Card", 
                success and skills_valid, 
                f"Found {len(card.get('skills', []))} skills",
                card if not success else None
            )
            return success and skills_valid
            
        except Exception as e:
            self.log_test("Agent Card", False, str(e))
            return False
    
    async def test_authentication_setup(self):
        """Test authentication setup and get credentials."""
        try:
            # Read generated credentials from server output
            if self.server_process:
                # Give server time to output credentials
                await asyncio.sleep(2)
                
                # Read server output
                output = self.server_process.stderr.read(4096).decode('utf-8')
                
                # Extract API key
                import re
                api_key_match = re.search(r'Generated default API key: (a2a_\w+)', output)
                if api_key_match:
                    self.client.api_key = api_key_match.group(1)
                
                # Extract OAuth2 credentials
                client_id_match = re.search(r'Client ID: (oauth2_\w+)', output)
                client_secret_match = re.search(r'Client Secret: (\w+)', output)
                
                if client_id_match and client_secret_match:
                    self.client.oauth_client_id = client_id_match.group(1)
                    self.client.oauth_client_secret = client_secret_match.group(1)
                
                success = bool(self.client.api_key and self.client.oauth_client_id)
                self.log_test(
                    "Authentication Setup",
                    success,
                    f"API Key: {'Found' if self.client.api_key else 'Not found'}, "
                    f"OAuth2: {'Found' if self.client.oauth_client_id else 'Not found'}"
                )
                return success
            
            self.log_test("Authentication Setup", False, "Server process not running")
            return False
            
        except Exception as e:
            self.log_test("Authentication Setup", False, str(e))
            return False
    
    async def test_oauth2_flow(self):
        """Test OAuth2 client credentials flow."""
        try:
            if not self.client.oauth_client_id:
                self.log_test("OAuth2 Flow", False, "No OAuth2 credentials available")
                return False
            
            # Get token
            token_response = await self.client.get_oauth_token(
                self.client.oauth_client_id,
                self.client.oauth_client_secret
            )
            
            success = (
                "access_token" in token_response and
                token_response.get("token_type") == "bearer" and
                "expires_in" in token_response
            )
            
            if success:
                self.client.oauth_token = token_response["access_token"]
            
            self.log_test(
                "OAuth2 Flow",
                success,
                f"Token type: {token_response.get('token_type')}, "
                f"Expires in: {token_response.get('expires_in')}s"
            )
            return success
            
        except Exception as e:
            self.log_test("OAuth2 Flow", False, str(e))
            return False
    
    async def test_simple_text_query(self):
        """Test simple text query without context."""
        try:
            # Test with both authentication methods
            for auth_method in ["api_key", "oauth"]:
                params = {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "What is 2 + 2? Give a brief answer."
                            }
                        ]
                    },
                    "metadata": {
                        "skill_id": "minion_query",
                        "max_rounds": 1,
                        "local_provider": "ollama",
                        "local_model": "llama3.2"
                    }
                }
                
                # Send task
                response = await self.client.send_request("tasks/send", params, auth_method)
                
                if "error" in response:
                    self.log_test(f"Simple Query ({auth_method})", False, 
                                response["error"].get("message"), response)
                    continue
                
                task_id = response["result"]["id"]
                
                # Wait for completion
                max_attempts = 30
                completed = False
                
                for _ in range(max_attempts):
                    await asyncio.sleep(2)
                    
                    get_response = await self.client.send_request(
                        "tasks/get",
                        {"id": task_id},
                        auth_method
                    )
                    
                    if "result" in get_response:
                        state = get_response["result"]["status"]["state"]
                        if state == "completed":
                            completed = True
                            artifacts = get_response["result"].get("artifacts", [])
                            if artifacts:
                                answer = artifacts[0]["parts"][0].get("text", "")
                                success = "4" in answer
                                self.log_test(
                                    f"Simple Query ({auth_method})",
                                    success,
                                    f"Answer: {answer[:100]}..."
                                )
                            break
                        elif state == "failed":
                            self.log_test(
                                f"Simple Query ({auth_method})",
                                False,
                                f"Task failed: {get_response['result']['status'].get('message')}"
                            )
                            break
                
                if not completed:
                    self.log_test(f"Simple Query ({auth_method})", False, "Task did not complete in time")
            
            return True
            
        except Exception as e:
            self.log_test("Simple Query", False, str(e))
            return False
    
    async def test_query_with_context(self):
        """Test query with document context."""
        try:
            context_text = """
            The A2A Protocol (Agent-to-Agent Protocol) is a standardized framework 
            for communication between autonomous AI agents. It was introduced by Google 
            in April 2025 and developed in collaboration with over 50 technology partners.
            
            Key features of A2A:
            1. Built on standard web technologies (HTTP, SSE, JSON-RPC)
            2. Supports multiple authentication methods
            3. Enables streaming responses
            4. Modality agnostic (text, audio, video, data)
            5. Designed for long-running tasks
            """
            
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What are the key features of the A2A Protocol?"
                        },
                        {
                            "kind": "text", 
                            "text": context_text
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query",
                    "max_rounds": 2
                }
            }
            
            response = await self.client.send_request("tasks/send", params)
            
            if "error" in response:
                self.log_test("Query with Context", False, response["error"].get("message"))
                return False
            
            task_id = response["result"]["id"]
            
            # Wait and check result
            await asyncio.sleep(5)
            
            get_response = await self.client.send_request("tasks/get", {"id": task_id})
            
            success = False
            if "result" in get_response and get_response["result"]["status"]["state"] == "completed":
                artifacts = get_response["result"].get("artifacts", [])
                if artifacts:
                    answer = artifacts[0]["parts"][0].get("text", "")
                    # Check if answer mentions key features
                    success = any(keyword in answer.lower() for keyword in 
                                ["http", "authentication", "streaming", "modality", "standard"])
            
            self.log_test("Query with Context", success, 
                        "Context properly utilized" if success else "Context not properly used")
            return success
            
        except Exception as e:
            self.log_test("Query with Context", False, str(e))
            return False
    
    async def test_pdf_processing(self):
        """Test PDF file processing."""
        try:
            # Create a simple test PDF content (base64 encoded)
            # This is a minimal PDF that says "Hello World"
            pdf_content = """JVBERi0xLjMKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovT3V0bGluZXMgMiAwIFIKL1BhZ2VzIDMgMCBSCj4+CmVuZG9iagoyIDAgb2JqCjw8Ci9UeXBlIC9PdXRsaW5lcwovQ291bnQgMAo+PgplbmRvYmoKMyAwIG9iago8PAovVHlwZSAvUGFnZXMKL0NvdW50IDEKL0tpZHMgWzQgMCBSXQo+PgplbmRvYmoKNCAwIG9iago8PAovVHlwZSAvUGFnZQovUGFyZW50IDMgMCBSCi9NZWRpYUJveCBbMCAwIDYxMiA3OTJdCi9Db250ZW50cyA1IDAgUgovUmVzb3VyY2VzIDw8Ci9Qcm9jU2V0IFsvUERGIC9UZXh0XQovRm9udCA8PAovRjEgNiAwIFIKPj4KPj4KPj4KZW5kb2JqCjUgMCBvYmoKPDwKL0xlbmd0aCA0NAo+PgpzdHJlYW0KQlQKL0YxIDEyIFRmCjEwMCA3MDAgVGQKKEhlbGxvIFdvcmxkKSBUagpFVAplbmRzdHJlYW0KZW5kb2JqCjYgMCBvYmoKPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+CmVuZG9iagp4cmVmCjAgNwowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA3NCAwMDAwMCBuIAowMDAwMDAwMTIwIDAwMDAwIG4gCjAwMDAwMDAxNzkgMDAwMDAgbiAKMDAwMDAwMDM2NCAwMDAwMCBuIAowMDAwMDAwNDY2IDAwMDAwIG4gCnRyYWlsZXIKPDwKL1NpemUgNwovUm9vdCAxIDAgUgo+PgpzdGFydHhyZWYKNTY1CiUlRU9G"""
            
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What text is in this PDF?"
                        },
                        {
                            "kind": "file",
                            "file": {
                                "name": "test.pdf",
                                "mimeType": "application/pdf",
                                "bytes": pdf_content
                            }
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query",
                    "max_rounds": 1
                }
            }
            
            response = await self.client.send_request("tasks/send", params)
            
            if "error" in response:
                self.log_test("PDF Processing", False, response["error"].get("message"))
                return False
            
            task_id = response["result"]["id"]
            
            # Wait for processing
            await asyncio.sleep(5)
            
            get_response = await self.client.send_request("tasks/get", {"id": task_id})
            
            success = False
            if "result" in get_response:
                state = get_response["result"]["status"]["state"]
                if state == "completed":
                    artifacts = get_response["result"].get("artifacts", [])
                    if artifacts:
                        answer = artifacts[0]["parts"][0].get("text", "")
                        success = "hello" in answer.lower() or "world" in answer.lower()
                
            self.log_test("PDF Processing", success, 
                        "PDF content extracted" if success else "Failed to extract PDF content")
            return success
            
        except Exception as e:
            self.log_test("PDF Processing", False, str(e))
            return False
    
    async def test_json_data_processing(self):
        """Test JSON data processing."""
        try:
            data = {
                "sales_data": [
                    {"month": "January", "revenue": 50000, "units": 100},
                    {"month": "February", "revenue": 55000, "units": 110},
                    {"month": "March", "revenue": 60000, "units": 120}
                ],
                "total_revenue": 165000,
                "average_revenue": 55000
            }
            
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What was the total revenue across all months?"
                        },
                        {
                            "kind": "data",
                            "data": data
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query"
                }
            }
            
            response = await self.client.send_request("tasks/send", params)
            
            if "error" in response:
                self.log_test("JSON Data Processing", False, response["error"].get("message"))
                return False
            
            task_id = response["result"]["id"]
            
            # Wait for processing
            await asyncio.sleep(5)
            
            get_response = await self.client.send_request("tasks/get", {"id": task_id})
            
            success = False
            if "result" in get_response and get_response["result"]["status"]["state"] == "completed":
                artifacts = get_response["result"].get("artifacts", [])
                if artifacts:
                    answer = artifacts[0]["parts"][0].get("text", "")
                    success = "165000" in answer or "165,000" in answer
            
            self.log_test("JSON Data Processing", success,
                        "JSON data analyzed correctly" if success else "Failed to analyze JSON data")
            return success
            
        except Exception as e:
            self.log_test("JSON Data Processing", False, str(e))
            return False
    
    async def test_streaming_response(self):
        """Test streaming response functionality."""
        try:
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Count from 1 to 5 slowly, showing each number."
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query",
                    "max_rounds": 1
                }
            }
            
            events = await self.client.send_request_streaming("tasks/sendSubscribe", params)
            
            # Check we got streaming events
            message_events = [e for e in events if e.get("result", {}).get("kind") == "message"]
            final_event = next((e for e in events if e.get("result", {}).get("final")), None)
            
            success = len(message_events) > 0 and final_event is not None
            
            self.log_test("Streaming Response", success,
                        f"Received {len(message_events)} message events")
            return success
            
        except Exception as e:
            self.log_test("Streaming Response", False, str(e))
            return False
    
    async def test_task_cancellation(self):
        """Test task cancellation."""
        try:
            # Start a long-running task
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Count from 1 to 1000000 very slowly."
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query",
                    "timeout": 300  # Long timeout
                }
            }
            
            response = await self.client.send_request("tasks/send", params)
            task_id = response["result"]["id"]
            
            # Wait a bit then cancel
            await asyncio.sleep(2)
            
            cancel_response = await self.client.send_request("tasks/cancel", {"id": task_id})
            
            success = "result" in cancel_response
            
            # Verify task was canceled
            if success:
                await asyncio.sleep(1)
                get_response = await self.client.send_request("tasks/get", {"id": task_id})
                if "result" in get_response:
                    state = get_response["result"]["status"]["state"]
                    success = state == "canceled"
            
            self.log_test("Task Cancellation", success,
                        "Task successfully canceled" if success else "Failed to cancel task")
            return success
            
        except Exception as e:
            self.log_test("Task Cancellation", False, str(e))
            return False
    
    async def test_parallel_minions_query(self):
        """Test parallel processing with minions_query skill."""
        try:
            # Create multiple documents
            docs = [
                "Paris is the capital of France. The Eiffel Tower is located in Paris.",
                "London is the capital of the United Kingdom. Big Ben is a famous landmark.",
                "Tokyo is the capital of Japan. Mount Fuji is visible from Tokyo on clear days."
            ]
            
            combined_context = "\n\n---\n\n".join(docs)
            
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "List all the capital cities mentioned in these documents."
                        },
                        {
                            "kind": "text",
                            "text": combined_context
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minions_query",
                    "max_rounds": 2,
                    "num_tasks_per_round": 3
                }
            }
            
            response = await self.client.send_request("tasks/send", params)
            
            if "error" in response:
                self.log_test("Parallel Minions Query", False, response["error"].get("message"))
                return False
            
            task_id = response["result"]["id"]
            
            # Wait for parallel processing
            await asyncio.sleep(10)
            
            get_response = await self.client.send_request("tasks/get", {"id": task_id})
            
            success = False
            if "result" in get_response and get_response["result"]["status"]["state"] == "completed":
                artifacts = get_response["result"].get("artifacts", [])
                if artifacts:
                    answer = artifacts[0]["parts"][0].get("text", "").lower()
                    # Check if all capitals are mentioned
                    success = all(city in answer for city in ["paris", "london", "tokyo"])
            
            self.log_test("Parallel Minions Query", success,
                        "All capitals identified" if success else "Failed to identify all capitals")
            return success
            
        except Exception as e:
            self.log_test("Parallel Minions Query", False, str(e))
            return False
    
    async def test_error_handling(self):
        """Test various error conditions."""
        test_cases = [
            {
                "name": "Invalid Method",
                "method": "invalid/method",
                "params": {},
                "expected_code": -32601
            },
            {
                "name": "Missing Parameters",
                "method": "tasks/send",
                "params": {},
                "expected_code": -32602
            },
            {
                "name": "Invalid Task ID",
                "method": "tasks/get",
                "params": {"id": "non-existent-task"},
                "expected_code": -32603  # Internal error when task not found
            },
            {
                "name": "Invalid Message Format",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "invalid-role",  # Should be 'user' or 'agent'
                        "parts": []
                    }
                },
                "expected_code": -32602
            }
        ]
        
        all_passed = True
        for test_case in test_cases:
            try:
                response = await self.client.send_request(
                    test_case["method"],
                    test_case["params"]
                )
                
                success = (
                    "error" in response and
                    response["error"]["code"] == test_case["expected_code"]
                )
                
                self.log_test(
                    f"Error Handling - {test_case['name']}",
                    success,
                    f"Got error code {response.get('error', {}).get('code')}"
                )
                
                all_passed &= success
                
            except Exception as e:
                self.log_test(f"Error Handling - {test_case['name']}", False, str(e))
                all_passed = False
        
        return all_passed
    
    async def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        try:
            response = await self.client.client.get(f"{self.client.base_url}/metrics")
            
            success = response.status_code == 200
            content = response.text
            
            # Check for expected metrics
            expected_metrics = [
                "a2a_minions_requests_total",
                "a2a_minions_tasks_total",
                "a2a_minions_active_tasks",
                "a2a_minions_auth_attempts_total"
            ]
            
            metrics_found = all(metric in content for metric in expected_metrics)
            
            self.log_test("Metrics Endpoint", success and metrics_found,
                        "All expected metrics present" if metrics_found else "Some metrics missing")
            return success and metrics_found
            
        except Exception as e:
            self.log_test("Metrics Endpoint", False, str(e))
            return False
    
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("=" * 60)
        logger.info("Starting A2A-Minions Integration Tests")
        logger.info("=" * 60)
        
        # Start server
        if not await self.start_server():
            logger.error("Failed to start server, aborting tests")
            return False
        
        try:
            # Run tests in sequence
            test_methods = [
                self.test_health_check,
                self.test_agent_card,
                self.test_authentication_setup,
                self.test_oauth2_flow,
                self.test_simple_text_query,
                self.test_query_with_context,
                self.test_pdf_processing,
                self.test_json_data_processing,
                self.test_streaming_response,
                self.test_task_cancellation,
                self.test_parallel_minions_query,
                self.test_error_handling,
                self.test_metrics_endpoint
            ]
            
            for test_method in test_methods:
                await test_method()
                await asyncio.sleep(1)  # Brief pause between tests
            
            # Print summary
            logger.info("=" * 60)
            logger.info("Test Summary")
            logger.info("=" * 60)
            
            passed = sum(1 for r in self.test_results if r["success"])
            total = len(self.test_results)
            
            logger.info(f"Total Tests: {total}")
            logger.info(f"Passed: {passed}")
            logger.info(f"Failed: {total - passed}")
            logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
            
            if total - passed > 0:
                logger.info("\nFailed Tests:")
                for result in self.test_results:
                    if not result["success"]:
                        logger.info(f"  - {result['name']}: {result['message']}")
            
            return passed == total
            
        finally:
            # Cleanup
            await self.client.close()
            self.stop_server()
            
            # Clean up temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass


async def main():
    """Main entry point."""
    test_suite = A2AIntegrationTest()
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())