#!/usr/bin/env python3
"""
Integration tests for minion_query skill.
Run as: python test_minion_query.py

Prerequisites: A2A-Minions server must be running at http://localhost:8001
"""

import asyncio
import json
import base64
import httpx
import uuid
import sys
from typing import Dict, Any


BASE_URL = "http://localhost:8001"
TIMEOUT = 30.0


async def check_server_health():
    """Check if server is running and healthy."""
    print("🔍 Checking server health...")
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            health_data = response.json()
            assert health_data["status"] == "healthy", f"Server not healthy: {health_data}"
            assert health_data["service"] == "a2a-minions", f"Wrong service: {health_data}"
            
        print("✅ Server is healthy")
        return True
        
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False


async def test_agent_card():
    """Test agent card endpoint."""
    print("🧪 Testing agent card...")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{BASE_URL}/.well-known/agent.json")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        agent_card = response.json()
        assert agent_card["name"] == "Minions Protocol Agent", f"Wrong agent name: {agent_card['name']}"
        assert len(agent_card["skills"]) == 2, f"Expected 2 skills, got {len(agent_card['skills'])}"
        
        # Check that minion_query skill is present
        skill_ids = [skill["id"] for skill in agent_card["skills"]]
        assert "minion_query" in skill_ids, f"minion_query skill missing: {skill_ids}"
        
    print("✅ Agent card test passed")


async def test_minion_query_text_only():
    """Test minion_query with text-only question."""
    print("🧪 Testing minion_query with text-only question...")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "What is the capital of France?"}
                    ]
                },
                "metadata": {"skill_id": "minion_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        # Send task
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        assert result["jsonrpc"] == "2.0", f"Wrong JSON-RPC version: {result.get('jsonrpc')}"
        assert "result" in result, f"No result in response: {result}"
        
        task = result["result"]
        task_id = task["id"]
        assert task["status"]["state"] == "submitted", f"Task not submitted: {task['status']}"
        
        # Wait for completion
        completed_task = await wait_for_completion(client, task_id)
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        assert len(completed_task["artifacts"]) > 0, "No artifacts returned"
        
        # Check artifact content
        artifact = completed_task["artifacts"][0]
        assert artifact["name"] == "Minions Execution Result", f"Wrong artifact name: {artifact['name']}"
        assert len(artifact["parts"]) >= 1, "No parts in artifact"
        assert artifact["parts"][0]["kind"] == "text", f"Wrong part kind: {artifact['parts'][0]['kind']}"
        
        # Should have a reasonable response
        answer_text = artifact["parts"][0]["text"]
        assert len(answer_text) > 10, f"Answer too short: {answer_text}"
        
    print("✅ Text-only query test passed")


async def test_minion_query_with_document():
    """Test minion_query with question and document."""
    print("🧪 Testing minion_query with document...")
    
    document_text = """
    The Treaty of Versailles was signed on June 28, 1919, officially ending World War I 
    between Germany and the Allied Powers. The treaty included harsh terms for Germany, 
    including territorial losses, military restrictions, and war reparations.
    Key provisions included the War Guilt Clause, which assigned full responsibility 
    for the war to Germany and its allies.
    """
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "When was the Treaty of Versailles signed and what were its key provisions?"},
                        {"kind": "text", "text": document_text}
                    ]
                },
                "metadata": {"skill_id": "minion_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        task = result["result"]
        
        completed_task = await wait_for_completion(client, task["id"])
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        
        artifact = completed_task["artifacts"][0]
        answer_text = artifact["parts"][0]["text"].lower()
        
        # Should reference the date and key provisions
        has_date = "1919" in answer_text or "june" in answer_text
        assert has_date, f"Date not mentioned in answer: {answer_text[:200]}..."
        assert len(answer_text) > 50, f"Answer too short: {answer_text}"
        
    print("✅ Document query test passed")


async def test_minion_query_with_file():
    """Test minion_query with file document."""
    print("🧪 Testing minion_query with file...")
    
    # Create a mock text file
    file_content = """
    BUSINESS REPORT - Q3 2024
    
    Executive Summary:
    Our company achieved record revenue of $2.8M in Q3 2024, representing 
    a 23% increase over the previous quarter. Key drivers included:
    
    1. Launch of new product line (contributed $600K)
    2. Expansion into European markets  
    3. Improved operational efficiency
    
    Challenges:
    - Supply chain disruptions in September
    - Increased competition in core markets
    
    Outlook:
    We expect continued growth in Q4 with strong pipeline of new customers.
    """
    
    file_bytes = file_content.encode('utf-8')
    file_b64 = base64.b64encode(file_bytes).decode('utf-8')
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user", 
                    "parts": [
                        {"kind": "text", "text": "What were the key achievements and challenges in Q3 2024?"},
                        {
                            "kind": "file",
                            "file": {
                                "name": "q3_report.txt",
                                "mimeType": "text/plain",
                                "bytes": file_b64
                            }
                        }
                    ]
                },
                "metadata": {"skill_id": "minion_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        task = result["result"]
        
        completed_task = await wait_for_completion(client, task["id"])
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        
        artifact = completed_task["artifacts"][0]
        answer_text = artifact["parts"][0]["text"].lower()
        
        # Should mention achievements and challenges
        has_revenue = "2.8m" in answer_text or "revenue" in answer_text
        assert has_revenue, f"Revenue not mentioned: {answer_text[:200]}..."
        assert len(answer_text) > 100, f"Answer too short: {answer_text}"
        
    print("✅ File query test passed")


async def test_minion_query_with_data():
    """Test minion_query with structured data."""
    print("🧪 Testing minion_query with structured data...")
    
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
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "Analyze the quarterly sales performance and identify trends"},
                        {"kind": "data", "data": sales_data}
                    ]
                },
                "metadata": {"skill_id": "minion_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        task = result["result"]
        
        completed_task = await wait_for_completion(client, task["id"])
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        
        artifact = completed_task["artifacts"][0]
        answer_text = artifact["parts"][0]["text"].lower()
        
        # Should analyze the data and mention trends
        has_analysis = "revenue" in answer_text or "sales" in answer_text
        assert has_analysis, f"Analysis keywords missing: {answer_text[:200]}..."
        assert len(answer_text) > 50, f"Answer too short: {answer_text}"
        
    print("✅ Data query test passed")


async def test_error_handling():
    """Test error handling with empty query."""
    print("🧪 Testing error handling...")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send", 
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": []
                },
                "metadata": {"skill_id": "minion_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        
        # Should either return an error or create a task that fails gracefully
        if "error" in result:
            assert result["error"]["code"] is not None, "Error should have code"
            print("✅ Error handling test passed (returned error)")
        else:
            task = result["result"]
            completed_task = await wait_for_completion(client, task["id"])
            # Should handle gracefully (either complete or fail cleanly)
            final_state = completed_task["status"]["state"]
            assert final_state in ["completed", "failed"], f"Unexpected final state: {final_state}"
            print("✅ Error handling test passed (graceful handling)")


async def wait_for_completion(client: httpx.AsyncClient, task_id: str, timeout: int = 30) -> Dict[str, Any]:
    """Wait for task completion and return final task state."""
    
    request = {
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "params": {"id": task_id},
        "id": str(uuid.uuid4())
    }
    
    for attempt in range(timeout):
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        if "error" in result:
            raise Exception(f"Task get failed: {result['error']}")
        
        task = result["result"]
        state = task["status"]["state"]
        
        if state in ["completed", "failed", "canceled"]:
            return task
            
        await asyncio.sleep(1)
    
    raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")


async def run_all_tests():
    """Run all minion_query tests."""
    
    print("🚀 Running minion_query integration tests")
    print("=" * 50)
    
    # Check server health first
    if not await check_server_health():
        print("💡 Start the server with: python run_server.py")
        return False
    
    tests = [
        test_agent_card,
        test_minion_query_text_only,
        test_minion_query_with_document,
        test_minion_query_with_file,
        test_minion_query_with_data,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 