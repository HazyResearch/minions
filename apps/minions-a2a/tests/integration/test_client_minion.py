#!/usr/bin/env python3
"""
Comprehensive test client for A2A-Minions server.
Tests all functionality including edge cases and error handling.
"""

import asyncio
import json
import uuid
import time
from typing import Dict, Any, Optional, List
import httpx
from pathlib import Path


class A2AMinionsTestClient:
    """Enhanced test client for A2A-Minions server."""
    
    def __init__(self, base_url: str = "http://localhost:8001", api_key: str = "abcd"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(timeout=30.0, headers=self.headers)
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Fetch the public agent card."""
        response = await self.client.get(f"{self.base_url}/.well-known/agent.json")
        response.raise_for_status()
        return response.json()
    
    async def get_extended_agent_card(self) -> Dict[str, Any]:
        """Fetch the extended agent card."""
        response = await self.client.get(f"{self.base_url}/agent/authenticatedExtendedCard")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def send_task(self, message: Dict[str, Any], metadata: Dict[str, Any] = None, 
                       task_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a task to the A2A server."""
        
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": message,
                "metadata": metadata or {}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await self.client.post(self.base_url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def send_task_streaming(self, message: Dict[str, Any], metadata: Dict[str, Any] = None,
                                 task_id: Optional[str] = None) -> str:
        """Send a task with streaming response."""
        
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0", 
            "method": "tasks/sendSubscribe",
            "params": {
                "id": task_id,
                "message": message,
                "metadata": metadata or {}
            },
            "id": str(uuid.uuid4())
        }
        
        stream_headers = self.headers.copy()
        stream_headers["Accept"] = "text/event-stream"
        
        async with self.client.stream("POST", self.base_url, json=payload,
                                     headers=stream_headers) as response:
            response.raise_for_status()
            
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    try:
                        event = json.loads(data)
                        events.append(event)
                        
                        # Print streaming updates
                        if "result" in event:
                            result = event["result"]
                            if result.get("kind") == "message":
                                role = result.get("metadata", {}).get("original_role", "unknown")
                                parts = result.get("parts", [])
                                for part in parts:
                                    if part.get("kind") == "text":
                                        text = part.get("text", "")[:100]
                                        print(f"   🔄 {role}: {text}...")
                            elif result.get("final"):
                                print("   ✅ Streaming completed")
                                break
                    except json.JSONDecodeError:
                        continue
            
            return task_id
    
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get task status and results."""
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "id": task_id
            },
            "id": str(uuid.uuid4())
        }
        
        response = await self.client.post(self.base_url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task."""
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {
                "id": task_id
            },
            "id": str(uuid.uuid4())
        }
        
        response = await self.client.post(self.base_url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def wait_for_completion(self, task_id: str, timeout: int = 60, 
                                 poll_interval: float = 1.0) -> Dict[str, Any]:
        """Wait for task completion with configurable polling."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_task(task_id)
            
            if "result" in result:
                task = result["result"]
                state = task.get("status", {}).get("state")
                
                if state in ["completed", "failed", "canceled"]:
                    return task
                
                print(f"   ⏳ Task state: {state}")
            elif "error" in result:
                raise Exception(f"Task error: {result['error']}")
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


class A2ATestSuite:
    """Comprehensive test suite for A2A-Minions server."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.client = A2AMinionsTestClient(base_url)
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({"name": test_name, "success": success, "message": message})
    
    async def test_health_check(self):
        """Test server health endpoint."""
        try:
            health = await self.client.health_check()
            self.log_test("Health Check", True, f"Status: {health.get('status')}")
            return True
        except Exception as e:
            self.log_test("Health Check", False, str(e))
            return False
    
    async def test_agent_cards(self):
        """Test agent card endpoints."""
        try:
            # Test public agent card
            public_card = await self.client.get_agent_card()
            assert "name" in public_card
            assert "skills" in public_card
            self.log_test("Public Agent Card", True, f"Found {len(public_card['skills'])} skills")
            
            # Test extended agent card
            extended_card = await self.client.get_extended_agent_card()
            assert "name" in extended_card
            self.log_test("Extended Agent Card", True, "Retrieved successfully")
            
            return True
        except Exception as e:
            self.log_test("Agent Cards", False, str(e))
            return False
    
    async def test_simple_query(self):
        """Test simple query without context."""
        try:
            message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "What is the capital of France?"
                    }
                ]
            }
            
            metadata = {
                "skill_id": "minion_query",
                "local_provider": "ollama",
                "local_model": "llama3.2",
                "remote_provider": "openai",
                "remote_model": "gpt-4o-mini",
                "max_rounds": 1
            }
            
            # Send task
            task_response = await self.client.send_task(message, metadata)
            assert "result" in task_response
            
            task_id = task_response["result"]["id"]
            
            # Wait for completion
            completed_task = await self.client.wait_for_completion(task_id, timeout=30)
            
            # Verify results
            assert completed_task["status"]["state"] == "completed"
            assert len(completed_task["artifacts"]) > 0
            
            # Extract answer
            artifact = completed_task["artifacts"][0]
            answer_text = ""
            for part in artifact["parts"]:
                if part["kind"] == "text" and part.get("metadata", {}).get("type") == "final_answer":
                    answer_text = part["text"]
                    break
            
            assert "Paris" in answer_text
            self.log_test("Simple Query", True, "Correct answer received")
            return True
            
        except Exception as e:
            self.log_test("Simple Query", False, str(e))
            return False
    
    async def test_query_with_context(self):
        """Test query with text context."""
        try:
            message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "When was the Treaty of Versailles signed and what were its key provisions?"
                    },
                    {
                        "kind": "text",
                        "text": "The Treaty of Versailles was signed on June 28, 1919, officially ending World War I between Germany and the Allied Powers. The treaty included harsh terms for Germany, including territorial losses, military restrictions, and war reparations. Key provisions included the War Guilt Clause, which assigned full responsibility for the war to Germany and its allies."
                    }
                ]
            }
            
            metadata = {
                "skill_id": "minion_query",
                "max_rounds": 2
            }
            
            # Send task
            task_response = await self.client.send_task(message, metadata)
            task_id = task_response["result"]["id"]
            
            # Wait for completion
            completed_task = await self.client.wait_for_completion(task_id, timeout=45)
            
            # Verify results
            assert completed_task["status"]["state"] == "completed"
            
            # Extract answer
            artifact = completed_task["artifacts"][0]
            answer_text = ""
            for part in artifact["parts"]:
                if part["kind"] == "text" and part.get("metadata", {}).get("type") == "final_answer":
                    answer_text = part["text"]
                    break
            
            assert "June 28, 1919" in answer_text
            assert "War Guilt Clause" in answer_text
            self.log_test("Query with Context", True, "Context properly utilized")
            return True
            
        except Exception as e:
            self.log_test("Query with Context", False, str(e))
            return False
    
    async def test_json_data_processing(self):
        """Test processing JSON data."""
        try:
            sales_data = {
                "quarterly_sales": {
                    "Q1": {"revenue": 100000, "units": 450, "region": "North America"},
                    "Q2": {"revenue": 120000, "units": 520, "region": "North America"},
                    "Q3": {"revenue": 110000, "units": 480, "region": "Europe"},
                    "Q4": {"revenue": 140000, "units": 600, "region": "Asia Pacific"}
                },
                "total_revenue": 470000,
                "total_units": 2050,
                "growth_rate": 0.18
            }
            
            message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Analyze the quarterly sales performance and identify trends"
                    },
                    {
                        "kind": "data",
                        "data": sales_data
                    }
                ]
            }
            
            metadata = {
                "skill_id": "minion_query",
                "max_rounds": 2
            }
            
            # Send task
            task_response = await self.client.send_task(message, metadata)
            task_id = task_response["result"]["id"]
            
            # Wait for completion
            completed_task = await self.client.wait_for_completion(task_id, timeout=45)
            
            # Verify results
            assert completed_task["status"]["state"] == "completed"
            self.log_test("JSON Data Processing", True, "JSON data analyzed successfully")
            return True
            
        except Exception as e:
            self.log_test("JSON Data Processing", False, str(e))
            return False
    
    async def test_streaming_response(self):
        """Test streaming response functionality."""
        try:
            message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Explain the benefits of renewable energy in detail"
                    }
                ]
            }
            
            metadata = {
                "skill_id": "minion_query",
                "max_rounds": 1
            }
            
            print("   🔄 Testing streaming response...")
            task_id = await self.client.send_task_streaming(message, metadata)
            
            # Verify task was created
            task_status = await self.client.get_task(task_id)
            assert "result" in task_status
            
            self.log_test("Streaming Response", True, "Streaming completed successfully")
            return True
            
        except Exception as e:
            self.log_test("Streaming Response", False, str(e))
            return False
    
    async def test_error_handling(self):
        """Test error handling for malformed requests."""
        try:
            # Test 1: Empty message
            try:
                await self.client.send_task({})
                self.log_test("Error Handling - Empty Message", False, "Should have failed")
                return False
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    self.log_test("Error Handling - Empty Message", True, "Correctly rejected")
                else:
                    self.log_test("Error Handling - Empty Message", False, f"Wrong status: {e.response.status_code}")
                    return False
            
            # Test 2: Missing parts
            try:
                message = {"role": "user", "parts": []}
                await self.client.send_task(message)
                self.log_test("Error Handling - Empty Parts", False, "Should have failed")
                return False
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    self.log_test("Error Handling - Empty Parts", True, "Correctly rejected")
                else:
                    self.log_test("Error Handling - Empty Parts", False, f"Wrong status: {e.response.status_code}")
                    return False
            
            # Test 3: Invalid method
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "invalid/method",
                    "params": {},
                    "id": str(uuid.uuid4())
                }
                response = await self.client.client.post(self.client.base_url, json=payload)
                if response.status_code == 400:
                    self.log_test("Error Handling - Invalid Method", True, "Correctly rejected")
                else:
                    self.log_test("Error Handling - Invalid Method", False, f"Wrong status: {response.status_code}")
                    return False
            except Exception as e:
                self.log_test("Error Handling - Invalid Method", False, str(e))
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Error Handling", False, str(e))
            return False
    
    async def test_task_management(self):
        """Test task management operations."""
        try:
            # Create a task
            message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Count to 10 slowly"
                    }
                ]
            }
            
            metadata = {"skill_id": "minion_query", "max_rounds": 1}
            task_response = await self.client.send_task(message, metadata)
            task_id = task_response["result"]["id"]
            
            # Get task status
            task_status = await self.client.get_task(task_id)
            assert "result" in task_status
            
            # Wait a moment then try to cancel (might already be completed)
            await asyncio.sleep(1)
            cancel_response = await self.client.cancel_task(task_id)
            
            self.log_test("Task Management", True, "Task operations completed")
            return True
            
        except Exception as e:
            self.log_test("Task Management", False, str(e))
            return False
    
    async def test_long_context_minions_paper(self):
        """Test long-context processing with the full minions paper."""
        try:
            # Read the minions paper
            paper_path = Path(__file__).parent / "minions_paper.txt"
            if not paper_path.exists():
                self.log_test("Long Context - Minions Paper", False, "minions_paper.txt not found")
                return False
            
            with open(paper_path, "r", encoding="utf-8") as f:
                minions_paper_content = f.read()
            
            print(f"   📄 Loaded minions paper: {len(minions_paper_content)} characters")
            
            message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "According to the paper, what is the specific percentage improvement in accuracy that the Minions protocol achieved over baseline methods? Please provide the exact statistic mentioned in the evaluation section."
                    },
                    {
                        "kind": "text",
                        "text": minions_paper_content
                    }
                ]
            }
            
            metadata = {
                "skill_id": "minion_query",
                "local_provider": "ollama",
                "local_model": "llama3.2",
                "remote_provider": "openai",
                "remote_model": "gpt-4o-mini",
                "max_rounds": 3
            }
            
            # Send task
            task_response = await self.client.send_task(message, metadata)
            assert "result" in task_response
            
            task_id = task_response["result"]["id"]
            print(f"   📤 Long context task submitted: {task_id}")
            
            # Wait for completion with longer timeout for complex processing
            completed_task = await self.client.wait_for_completion(task_id, timeout=90)
            
            # Verify results
            assert completed_task["status"]["state"] == "completed"
            assert len(completed_task["artifacts"]) > 0
            
            # Extract and display the answer for manual review
            artifact = completed_task["artifacts"][0]
            answer_text = ""
            for part in artifact["parts"]:
                if part["kind"] == "text" and part.get("metadata", {}).get("type") == "final_answer":
                    answer_text = part["text"]
                    break
            
            print(f"\n   📊 MINIONS PAPER ANALYSIS RESULT:")
            print(f"   " + "="*50)
            print(f"   {answer_text}")
            print(f"   " + "="*50)
            
            # Check if we got a substantive response (not just an error)
            if len(answer_text) > 50 and not answer_text.startswith("I don't have access"):
                self.log_test("Long Context - Minions Paper", True, f"Processed {len(minions_paper_content)} chars successfully")
                return True
            else:
                self.log_test("Long Context - Minions Paper", False, "Response too short or indicated no access")
                return False
            
        except Exception as e:
            self.log_test("Long Context - Minions Paper", False, str(e))
            return False

    async def run_all_tests(self) -> bool:
        """Run all tests in the suite."""
        print("🧪 A2A-Minions Comprehensive Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_long_context_minions_paper,  # First test - long context capabilities
            self.test_health_check,
            self.test_agent_cards,
            self.test_simple_query,
            self.test_query_with_context,
            self.test_json_data_processing,
            # self.test_streaming_response,
            self.test_error_handling,
            self.test_task_management,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            print(f"\n🔍 Running {test.__name__.replace('test_', '').replace('_', ' ').title()}...")
            try:
                success = await test()
                if success:
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        
        if passed == total:
            print("\n🎉 All tests passed! The A2A-Minions server is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        
        return passed == total
    
    async def close(self):
        """Close the test client."""
        await self.client.close()


async def main():
    """Main test execution."""
    test_suite = A2ATestSuite()
    
    try:
        success = await test_suite.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        return 1
    finally:
        await test_suite.close()


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)