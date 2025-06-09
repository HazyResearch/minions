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
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ["true", "1", "yes"]
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also set httpx logging level
logging.getLogger("httpx").setLevel(logging.WARNING)


class A2AIntegrationTestClient:
    """Client for comprehensive integration testing."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))  # Shorter timeout for faster failures
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
    
    async def wait_for_task_completion(self, task_id: str, test_name: str, 
                                     auth_method: str = "api_key", 
                                     max_wait_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for a task to complete with progress logging."""
        logger.info(f"⏳ Waiting for task {task_id} to complete (max {max_wait_seconds}s)...")
        
        max_attempts = max_wait_seconds // 2  # Check every 2 seconds
        last_state = None
        
        for attempt in range(max_attempts):
            await asyncio.sleep(2)
            
            try:
                response = await self.client.send_request(
                    "tasks/get",
                    {"id": task_id},
                    auth_method
                )
                
                if "error" in response:
                    logger.error(f"Error getting task status: {response['error']}")
                    return None
                
                result = response.get("result", {})
                status = result.get("status", {})
                state = status.get("state", "unknown")
                message = status.get("message", "")
                
                # Log state changes
                if state != last_state:
                    logger.info(f"📊 [{test_name}] Task state: {state}")
                    if message:
                        logger.info(f"   Message: {message}")
                    last_state = state
                elif attempt > 0 and attempt % 5 == 0:  # Log every 10 seconds
                    logger.info(f"⏳ [{test_name}] Still {state}... ({(attempt+1)*2}/{max_wait_seconds}s)")
                
                if state in ["completed", "failed", "canceled"]:
                    logger.info(f"🏁 [{test_name}] Task finished with state: {state}")
                    return result
                    
            except Exception as e:
                logger.error(f"Error checking task status: {type(e).__name__}: {e}")
                if hasattr(e, 'response'):
                    logger.error(f"Response details: {e.response}")
                # Continue checking instead of breaking
                
        logger.warning(f"⏱️ [{test_name}] Task did not complete within {max_wait_seconds} seconds")
        
        # Get final state
        try:
            final_response = await self.client.send_request("tasks/get", {"id": task_id}, auth_method)
            if "result" in final_response:
                return final_response["result"]
        except:
            pass
            
        return None
    
    async def start_server(self) -> bool:
        """Start the A2A-Minions server."""
        try:
            # Start server process
            env = os.environ.copy()
            env["PYTHONPATH"] = ":".join(sys.path)
            
            self.server_process = subprocess.Popen(
                [sys.executable, "run_server.py", "--port", "8001"],
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
            # Give server time to generate credentials
            await asyncio.sleep(3)
            
            # Read credentials from JSON files where the server stores them
            import json
            from pathlib import Path
            
            # Try to read API keys from api_keys.json
            api_keys_file = Path(__file__).parent.parent.parent / "api_keys.json"
            oauth_clients_file = Path(__file__).parent.parent.parent / "oauth2_clients.json"
            
            api_key_found = False
            oauth_found = False
            
            # Read API keys
            if api_keys_file.exists():
                try:
                    with open(api_keys_file, 'r') as f:
                        api_keys = json.load(f)
                    
                    # Get the first active API key
                    for key, data in api_keys.items():
                        if data.get("active", True) and key.startswith("a2a_"):
                            self.client.api_key = key
                            api_key_found = True
                            break
                except Exception as e:
                    logger.debug(f"Failed to read API keys file: {e}")
            
            # Read OAuth2 clients
            if oauth_clients_file.exists():
                try:
                    with open(oauth_clients_file, 'r') as f:
                        oauth_clients = json.load(f)
                    
                    # Get the first active OAuth2 client
                    for client_id, data in oauth_clients.items():
                        if data.get("active", True) and client_id.startswith("oauth2_"):
                            self.client.oauth_client_id = client_id
                            self.client.oauth_client_secret = data.get("client_secret")
                            if self.client.oauth_client_secret:
                                oauth_found = True
                            break
                except Exception as e:
                    logger.debug(f"Failed to read OAuth2 clients file: {e}")
            
            success = api_key_found and oauth_found
            self.log_test(
                "Authentication Setup",
                success,
                f"API Key: {'Found' if api_key_found else 'Not found'}, "
                f"OAuth2: {'Found' if oauth_found else 'Not found'}"
            )
            return success
            
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
                logger.info(f"\n🔄 Testing Simple Query with {auth_method} authentication...")
                
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
                        "remote_provider": "openai",
                        "remote_model": "gpt-4o-mini",
                        "remote_temperature": 0
                    }
                }
                
                # Send task
                logger.debug(f"Sending task with params: {json.dumps(params, indent=2)}")
                response = await self.client.send_request("tasks/send", params, auth_method)
                
                if "error" in response:
                    self.log_test(f"Simple Query ({auth_method})", False, 
                                response["error"].get("message"), response)
                    continue
                
                task_id = response["result"]["id"]
                logger.info(f"✅ Task created successfully with ID: {task_id}")
                
                # Wait for completion
                result = await self.wait_for_task_completion(
                    task_id, f"Simple Query ({auth_method})", auth_method, max_wait_seconds=20
                )
                
                if result:
                    state = result["status"]["state"]
                    if state == "completed":
                        artifacts = result.get("artifacts", [])
                        if artifacts and artifacts[0].get("parts"):
                            answer = artifacts[0]["parts"][0].get("text", "")
                            success = "4" in answer
                            self.log_test(
                                f"Simple Query ({auth_method})",
                                success,
                                f"Answer: {answer[:200]}..." if answer else "No answer received"
                            )
                            if not success:
                                logger.debug(f"Full answer: {answer}")
                        else:
                            self.log_test(
                                f"Simple Query ({auth_method})",
                                False,
                                "Task completed but no artifacts/answer found"
                            )
                            logger.debug(f"Task result: {json.dumps(result, indent=2)}")
                    else:
                        self.log_test(
                            f"Simple Query ({auth_method})",
                            False,
                            f"Task ended with state: {state}, message: {result['status'].get('message')}"
                        )
                        logger.debug(f"Task details: {json.dumps(result, indent=2)}")
                else:
                    self.log_test(f"Simple Query ({auth_method})", False, "Failed to get task result")
            
            return True
            
        except Exception as e:
            self.log_test("Simple Query", False, f"Exception: {str(e)}")
            logger.exception("Exception in test_simple_text_query")
            return False
    
    async def test_query_with_context(self):
        """Test query with document context."""
        try:
            logger.info("\n🔄 Testing Query with Context...")
            
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
                    "max_rounds": 1,
                    "remote_provider": "openai",
                    "remote_model": "gpt-4o-mini",
                    "remote_temperature": 0
                }
            }
            
            logger.debug(f"Sending task with context...")
            response = await self.client.send_request("tasks/send", params)
            
            if "error" in response:
                self.log_test("Query with Context", False, response["error"].get("message"))
                return False
            
            task_id = response["result"]["id"]
            logger.info(f"✅ Task created with ID: {task_id}")
            
            # Wait for completion
            result = await self.wait_for_task_completion(
                task_id, "Query with Context", max_wait_seconds=20
            )
            
            if result and result["status"]["state"] == "completed":
                artifacts = result.get("artifacts", [])
                if artifacts and artifacts[0].get("parts"):
                    answer = artifacts[0]["parts"][0].get("text", "")
                    # Check if answer mentions key features
                    keywords = ["http", "authentication", "streaming", "modality", "standard", "web technologies"]
                    found_keywords = [kw for kw in keywords if kw in answer.lower()]
                    success = len(found_keywords) >= 3  # At least 3 keywords
                    
                    self.log_test("Query with Context", success, 
                                f"Found {len(found_keywords)}/{len(keywords)} keywords: {found_keywords}")
                    if not success:
                        logger.info(f"Answer preview: {answer[:300]}...")
                else:
                    self.log_test("Query with Context", False, "No answer found in artifacts")
            else:
                state = result["status"]["state"] if result else "unknown"
                self.log_test("Query with Context", False, 
                            f"Task ended with state: {state}")
                if result:
                    logger.debug(f"Task details: {json.dumps(result, indent=2)}")
            
            return True
            
        except Exception as e:
            self.log_test("Query with Context", False, f"Exception: {str(e)}")
            logger.exception("Exception in test_query_with_context")
            return False
    
    async def test_pdf_processing(self):
        """Test PDF file processing."""
        try:
            # Create a simple test PDF content (base64 encoded)
            # This is a PDF with "Hello World" text created using reportlab
            pdf_content = """JVBERi0xLjMKJZOMi54gUmVwb3J0TGFiIEdlbmVyYXRlZCBQREYgZG9jdW1lbnQgaHR0cDovL3d3dy5yZXBvcnRsYWIuY29tCjEgMCBvYmoKPDwKL0YxIDIgMCBSCj4+CmVuZG9iagoyIDAgb2JqCjw8Ci9CYXNlRm9udCAvSGVsdmV0aWNhIC9FbmNvZGluZyAvV2luQW5zaUVuY29kaW5nIC9OYW1lIC9GMSAvU3VidHlwZSAvVHlwZTEgL1R5cGUgL0ZvbnQKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL0NvbnRlbnRzIDcgMCBSIC9NZWRpYUJveCBbIDAgMCA2MTIgNzkyIF0gL1BhcmVudCA2IDAgUiAvUmVzb3VyY2VzIDw8Ci9Gb250IDEgMCBSIC9Qcm9jU2V0IFsgL1BERiAvVGV4dCAvSW1hZ2VCIC9JbWFnZUMgL0ltYWdlSSBdCj4+IC9Sb3RhdGUgMCAvVHJhbnMgPDwKCj4+IAogIC9UeXBlIC9QYWdlCj4+CmVuZG9iago0IDAgb2JqCjw8Ci9QYWdlTW9kZSAvVXNlTm9uZSAvUGFnZXMgNiAwIFIgL1R5cGUgL0NhdGFsb2cKPj4KZW5kb2JqCjUgMCBvYmoKPDwKL0F1dGhvciAoYW5vbnltb3VzKSAvQ3JlYXRpb25EYXRlIChEOjIwMjUwNjA5MDIxNTQ3LTA3JzAwJykgL0NyZWF0b3IgKFJlcG9ydExhYiBQREYgTGlicmFyeSAtIHd3dy5yZXBvcnRsYWIuY29tKSAvS2V5d29yZHMgKCkgL01vZERhdGUgKEQ6MjAyNTA2MDkwMjE1NDctMDcnMDAnKSAvUHJvZHVjZXIgKFJlcG9ydExhYiBQREYgTGlicmFyeSAtIHd3dy5yZXBvcnRsYWIuY29tKSAKICAvU3ViamVjdCAodW5zcGVjaWZpZWQpIC9UaXRsZSAodW50aXRsZWQpIC9UcmFwcGVkIC9GYWxzZQo+PgplbmRvYmoKNiAwIG9iago8PAovQ291bnQgMSAvS2lkcyBbIDMgMCBSIF0gL1R5cGUgL1BhZ2VzCj4+CmVuZG9iago3IDAgb2JqCjw8Ci9GaWx0ZXIgWyAvQVNDSUk4NURlY29kZSAvRmxhdGVEZWNvZGUgXSAvTGVuZ3RoIDE3NQo+PgpzdHJlYW0KR2FybzkwYWtpUCY0WkVuTURxUCZidDZHa1ptW21uKiZzOUNVNjc/Qiw1Q2E9Zy1uaitqK2BXaG1LQV9EaFJpSGI7UHBpRTFcPz8iQVZLUipCVFtXWjpiZF5kZ246N2RlaUZVLEw6Y2Nra1M5WmRLJD1QT28tS1JLT0RJVDFlaHBgSCs1Wl5qciw0NkM9MV5qPnFVQWRIOC49PmYpaDJvZWhnMGI5J0pINGkyKXR+PmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDgKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDczIDAwMDAwIG4gCjAwMDAwMDAxMDQgMDAwMDAgbiAKMDAwMDAwMDIxMSAwMDAwMCBuIAowMDAwMDAwNDA0IDAwMDAwIG4gCjAwMDAwMDA0NzIgMDAwMDAgbiAKMDAwMDAwMDc2OCAwMDAwMCBuIAowMDAwMDAwODI3IDAwMDAwIG4gCnRyYWlsZXIKPDwKL0lEIApbPDZhNjI0NmMyMDgyNTE4M2U3OTc1NDIwOGRjNTc4NTY2Pjw2YTYyNDZjMjA4MjUxODNlNzk3NTQyMDhkYzU3ODU2Nj5dCiUgUmVwb3J0TGFiIGdlbmVyYXRlZCBQREYgZG9jdW1lbnQgLS0gZGlnZXN0IChodHRwOi8vd3d3LnJlcG9ydGxhYi5jb20pCgovSW5mbyA1IDAgUgovUm9vdCA0IDAgUgovU2l6ZSA4Cj4+CnN0YXJ0eHJlZgoxMDkyCiUlRU9GCg=="""
            
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
            
            # Wait for task completion
            result = await self.wait_for_task_completion(task_id, "PDF Processing", max_wait_seconds=15)
            
            success = False
            if result and result["status"]["state"] == "completed":
                artifacts = result.get("artifacts", [])
                logger.debug(f"PDF Processing - artifacts: {json.dumps(artifacts, indent=2)}")
                if artifacts and artifacts[0].get("parts"):
                    answer = artifacts[0]["parts"][0].get("text", "")
                    logger.debug(f"PDF Processing - answer: {answer[:200] if answer else 'No answer'}")
                    success = "hello" in answer.lower() or "world" in answer.lower()
            else:
                state = result["status"]["state"] if result else "unknown"
                logger.debug(f"PDF Processing - task state: {state}")
                if result and result["status"].get("message"):
                    logger.debug(f"PDF Processing - status message: {result['status']['message']}")
                
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
            
            # Wait for task completion
            result = await self.wait_for_task_completion(task_id, "JSON Data Processing", max_wait_seconds=15)
            
            success = False
            if result and result["status"]["state"] == "completed":
                artifacts = result.get("artifacts", [])
                logger.debug(f"JSON Processing - artifacts: {json.dumps(artifacts, indent=2)}")
                if artifacts and artifacts[0].get("parts"):
                    answer = artifacts[0]["parts"][0].get("text", "")
                    logger.debug(f"JSON Processing - answer: {answer[:200] if answer else 'No answer'}")
                    success = "165000" in answer or "165,000" in answer
            else:
                state = result["status"]["state"] if result else "unknown"
                logger.debug(f"JSON Processing - task state: {state}")
                if result and result["status"].get("message"):
                    logger.debug(f"JSON Processing - status message: {result['status']['message']}")
            
            self.log_test("JSON Data Processing", success,
                        "JSON data analyzed correctly" if success else "Failed to analyze JSON data")
            return success
            
        except Exception as e:
            self.log_test("JSON Data Processing", False, str(e))
            return False
    
    async def test_streaming_response(self):
        """Test streaming response functionality."""
        try:
            logger.info("\n🔄 Testing Streaming Response...")
            
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Count from 1 to 3, showing each number on a new line."
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query",
                    "max_rounds": 1,
                    "remote_provider": "openai",
                    "remote_model": "gpt-4o-mini",
                    "remote_temperature": 0
                }
            }
            
            logger.info("Sending streaming request...")
            events = await self.client.send_request_streaming("tasks/sendSubscribe", params)
            
            # Log all events
            logger.info(f"Received {len(events)} total events")
            message_events = []
            final_event = None
            
            for i, event in enumerate(events):
                logger.debug(f"Raw event {i+1}: {json.dumps(event, indent=2)}")
                
                event_type = event.get("result", {}).get("kind", "unknown")
                is_final = event.get("result", {}).get("final", False)
                
                logger.debug(f"Event {i+1}: type={event_type}, final={is_final}")
                
                if event_type == "message":
                    message_events.append(event)
                    # Log the message content
                    parts = event.get("result", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        logger.info(f"  Message: {parts[0]['text'][:100]}...")
                
                # Also consider taskStatusUpdate with state=completed as final
                if is_final or (event_type == "taskStatusUpdate" and event.get("result", {}).get("state") == "completed"):
                    final_event = event
            
            success = len(message_events) > 0 and final_event is not None
            
            self.log_test("Streaming Response", success,
                        f"Received {len(message_events)} message events with final event")
            
            if not success:
                logger.debug(f"All events: {json.dumps(events, indent=2)}")
                
            return success
            
        except Exception as e:
            self.log_test("Streaming Response", False, f"Exception: {str(e)}")
            logger.exception("Exception in test_streaming_response")
            return False
    
    async def test_task_cancellation(self):
        """Test task cancellation."""
        try:
            logger.info("\n🔄 Testing Task Cancellation...")
            
            # Start a long-running task
            params = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Please think step by step about the meaning of life for the next 30 seconds."
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query",
                    "remote_provider": "openai",
                    "remote_model": "gpt-4o-mini",
                    "timeout": 60  # Long timeout
                }
            }
            
            logger.info("Creating long-running task...")
            response = await self.client.send_request("tasks/send", params)
            
            if "error" in response:
                self.log_test("Task Cancellation", False, f"Failed to create task: {response['error']}")
                return False
                
            task_id = response["result"]["id"]
            logger.info(f"Task created with ID: {task_id}")
            
            # Wait a bit to ensure task has started
            await asyncio.sleep(3)
            
            # Check task is running
            check_response = await self.client.send_request("tasks/get", {"id": task_id})
            if "result" in check_response:
                current_state = check_response["result"]["status"]["state"]
                logger.info(f"Task state before cancel: {current_state}")
            
            # Cancel the task
            logger.info("Sending cancel request...")
            cancel_response = await self.client.send_request("tasks/cancel", {"id": task_id})
            
            if "error" in cancel_response:
                self.log_test("Task Cancellation", False, 
                            f"Cancel request failed: {cancel_response['error'].get('message')}")
                return False
            
            # Wait a moment for cancellation to process
            await asyncio.sleep(2)
            
            # Verify task was canceled
            final_result = await self.wait_for_task_completion(
                task_id, "Task Cancellation", max_wait_seconds=10
            )
            
            if final_result:
                final_state = final_result["status"]["state"]
                success = final_state == "canceled"
                self.log_test("Task Cancellation", success,
                            f"Final state: {final_state}" + (" (successfully canceled)" if success else ""))
            else:
                self.log_test("Task Cancellation", False, "Failed to get final task state")
            
            return True
            
        except Exception as e:
            self.log_test("Task Cancellation", False, f"Exception: {str(e)}")
            logger.exception("Exception in test_task_cancellation")
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
            
            # Wait for task completion with longer timeout for parallel processing
            result = await self.wait_for_task_completion(task_id, "Parallel Minions Query", max_wait_seconds=20)
            
            success = False
            if result and result["status"]["state"] == "completed":
                artifacts = result.get("artifacts", [])
                logger.debug(f"Parallel Query - artifacts: {json.dumps(artifacts, indent=2)}")
                if artifacts and artifacts[0].get("parts"):
                    answer = artifacts[0]["parts"][0].get("text", "").lower()
                    logger.debug(f"Parallel Query - answer: {answer[:300] if answer else 'No answer'}")
                    # Check if all capitals are mentioned
                    found_cities = {city: city in answer for city in ["paris", "london", "tokyo"]}
                    logger.debug(f"Parallel Query - found cities: {found_cities}")
                    success = all(found_cities.values())
            else:
                state = result["status"]["state"] if result else "unknown"
                logger.debug(f"Parallel Query - task state: {state}")
                if result and result["status"].get("message"):
                    logger.debug(f"Parallel Query - status message: {result['status']['message']}")
            
            self.log_test("Parallel Minions Query", success,
                        "All capitals identified" if success else "Failed to identify all capitals")
            return success
            
        except Exception as e:
            self.log_test("Parallel Minions Query", False, str(e))
            return False
    
    async def test_error_handling(self):
        """Test various error conditions."""
        logger.info("\n🔄 Testing Error Handling...")
        
        test_cases = [
            {
                "name": "Invalid Method",
                "method": "invalid/method",
                "params": {},
                "expected_code": -32601,
                "expect_http_error": False
            },
            {
                "name": "Missing Parameters",
                "method": "tasks/send",
                "params": {},
                "expected_code": -32602,
                "expect_http_error": False
            },
            {
                "name": "Invalid Task ID",
                "method": "tasks/get",
                "params": {"id": "non-existent-task"},
                "expected_code": -32001,  # Task not found
                "expect_http_error": True,  # Returns 404
                "expected_status": 404
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
                "expected_code": -32602,
                "expect_http_error": False
            }
        ]
        
        all_passed = True
        for test_case in test_cases:
            try:
                logger.info(f"  Testing {test_case['name']}...")
                response = await self.client.send_request(
                    test_case["method"],
                    test_case["params"]
                )
                
                # Check if we got the expected error
                if "error" in response:
                    error_code = response["error"]["code"]
                    success = error_code == test_case["expected_code"]
                    
                    self.log_test(
                        f"Error Handling - {test_case['name']}",
                        success,
                        f"Got error code {error_code} (expected {test_case['expected_code']})"
                    )
                    
                    if not success:
                        logger.debug(f"Error response: {json.dumps(response['error'], indent=2)}")
                else:
                    self.log_test(
                        f"Error Handling - {test_case['name']}",
                        False,
                        "No error in response when error was expected"
                    )
                    logger.debug(f"Response: {json.dumps(response, indent=2)}")
                    success = False
                
                all_passed &= success
                
            except httpx.HTTPStatusError as e:
                # Some errors return HTTP error codes instead of JSON-RPC errors
                if test_case.get("expect_http_error"):
                    success = e.response.status_code == test_case.get("expected_status", 400)
                    self.log_test(
                        f"Error Handling - {test_case['name']}",
                        success,
                        f"Got HTTP {e.response.status_code} (expected {test_case.get('expected_status', 400)})"
                    )
                else:
                    # Try to parse JSON-RPC error from response
                    try:
                        error_response = e.response.json()
                        if "error" in error_response:
                            error_code = error_response["error"]["code"]
                            success = error_code == test_case["expected_code"]
                            self.log_test(
                                f"Error Handling - {test_case['name']}",
                                success,
                                f"Got error code {error_code} from HTTP {e.response.status_code} response"
                            )
                        else:
                            self.log_test(f"Error Handling - {test_case['name']}", False, 
                                        f"HTTP {e.response.status_code} error with no JSON-RPC error")
                            success = False
                    except:
                        self.log_test(f"Error Handling - {test_case['name']}", False, 
                                    f"HTTP {e.response.status_code} error: {str(e)}")
                        success = False
                
                all_passed &= success
                
            except Exception as e:
                self.log_test(f"Error Handling - {test_case['name']}", False, f"Exception: {str(e)}")
                logger.exception(f"Exception in error handling test {test_case['name']}")
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
        
        # Skip server startup when using external server
        if self.server_process is None:
            logger.info("Using external server - skipping server startup")
        else:
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
                logger.info("\n❌ Failed Tests:")
                for result in self.test_results:
                    if not result["success"]:
                        logger.info(f"  - {result['name']}: {result['message']}")
            
            # Also show passed tests for clarity
            logger.info("\n✅ Passed Tests:")
            for result in self.test_results:
                if result["success"]:
                    logger.info(f"  - {result['name']}")
            
            return passed == total
            
        finally:
            # Cleanup
            await self.client.close()
            
            # Only stop server if we started it
            if self.server_process is not None:
                self.stop_server()
            else:
                logger.info("Using external server - skipping server shutdown")
            
            # Clean up temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass


async def main():
    """Main entry point."""
    logger.info("NOTE: Using external server on port 8001 - make sure to run server separately!")
    logger.info("Run: cd apps/minions-a2a && python run_server.py --port 8001")
    
    test_suite = A2AIntegrationTest()
    
    # Check if server is available first
    logger.info("Checking if server is available on port 8001...")
    if not await test_suite.client.wait_for_server():
        logger.error("Server not available on port 8001! Please start the server manually.")
        sys.exit(1)
    
    logger.info("Server is ready on port 8001")
    
    # Run tests without starting server
    test_suite.server_process = None  # Skip server management
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())