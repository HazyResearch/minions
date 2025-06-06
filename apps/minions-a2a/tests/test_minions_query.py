#!/usr/bin/env python3
"""
Integration tests for minions_query skill.
Run as: python test_minions_query.py

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
TIMEOUT = 60.0  # Longer timeout for parallel processing


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


async def test_minions_query_multiple_questions():
    """Test minions_query with complex questions requiring parallel processing."""
    print("🧪 Testing minions_query with complex analysis...")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "Compare renewable energy technologies and their efficiency across multiple dimensions"}
                    ]
                },
                "metadata": {"skill_id": "minions_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        assert result["jsonrpc"] == "2.0", f"Wrong JSON-RPC version: {result.get('jsonrpc')}"
        assert "result" in result, f"No result in response: {result}"
        
        task = result["result"]
        task_id = task["id"]
        assert task["status"]["state"] == "submitted", f"Task not submitted: {task['status']}"
        
        # Wait for completion (parallel processing takes longer)
        completed_task = await wait_for_completion(client, task_id, timeout=90)
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        assert len(completed_task["artifacts"]) > 0, "No artifacts returned"
        
        artifact = completed_task["artifacts"][0]
        assert artifact["name"] == "Parallel Minions Execution Result", f"Wrong artifact name: {artifact['name']}"
        assert len(artifact["parts"]) >= 1, "No parts in artifact"
        assert artifact["parts"][0]["kind"] == "text", f"Wrong part kind: {artifact['parts'][0]['kind']}"
        
        # Should have comprehensive analysis due to parallel processing
        answer_text = artifact["parts"][0]["text"].lower()
        assert len(answer_text) > 100, f"Answer too short for parallel processing: {len(answer_text)}"
        has_energy_content = "energy" in answer_text or "renewable" in answer_text
        assert has_energy_content, f"Missing energy content: {answer_text[:200]}..."
        
    print("✅ Complex analysis test passed")


async def test_minions_query_with_large_document():
    """Test minions_query with large document for parallel processing."""
    print("🧪 Testing minions_query with large document...")
    
    # Create a comprehensive research document
    research_doc = """
    RENEWABLE ENERGY TECHNOLOGIES REPORT 2024
    
    Solar Energy:
    Solar photovoltaic (PV) technology has seen dramatic cost reductions, with efficiency rates now reaching 26% 
    for commercial panels. Concentrated solar power (CSP) offers thermal storage capabilities, enabling 24/7 
    power generation. Current global solar capacity exceeds 1.2 TW with annual growth of 25%.
    
    Wind Energy:
    Offshore wind farms are achieving capacity factors above 60% with new turbine designs reaching 15MW per unit.
    Onshore wind remains cost-competitive at $0.03-0.05/kWh in optimal locations. Total global wind capacity 
    approaches 900 GW with particularly strong growth in Asia-Pacific regions.
    
    Hydroelectric Power:
    Pumped hydro storage provides grid stability with round-trip efficiency of 80-90%. Small modular hydro 
    systems enable distributed generation in remote areas. Environmental concerns drive innovation in 
    fish-friendly turbine designs and sediment management.
    
    Geothermal Energy:
    Enhanced geothermal systems (EGS) expand viable locations beyond traditional hot springs. Binary cycle 
    plants achieve efficiency improvements while reducing environmental impact. Current global capacity 
    totals 16 GW with high potential for baseload power generation.
    
    Storage Technologies:
    Lithium-ion battery costs have dropped 90% since 2010, enabling grid-scale deployment. Flow batteries 
    offer longer duration storage for renewable integration. Hydrogen production via electrolysis creates 
    long-term storage potential and industrial applications.
    
    Economic Analysis:
    Levelized cost of electricity (LCOE) for renewables continues declining, with solar and wind becoming 
    the cheapest sources in most markets. Grid integration costs remain a challenge requiring smart grid 
    investments and demand response programs.
    """ * 3  # Make it larger to trigger parallel processing
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "Analyze the renewable energy landscape and provide investment recommendations"},
                        {"kind": "text", "text": research_doc}
                    ]
                },
                "metadata": {"skill_id": "minions_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        task = result["result"]
        
        completed_task = await wait_for_completion(client, task["id"], timeout=120)
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        
        artifact = completed_task["artifacts"][0]
        analysis = artifact["parts"][0]["text"].lower()
        
        # Should provide detailed analysis leveraging parallel processing
        assert len(analysis) > 300, f"Analysis too short: {len(analysis)}"
        has_solar = "solar" in analysis or "wind" in analysis
        has_investment = "investment" in analysis or "recommendation" in analysis
        assert has_solar, f"Missing energy analysis: {analysis[:200]}..."
        assert has_investment, f"Missing investment content: {analysis[:200]}..."
        
    print("✅ Large document test passed")


async def test_minions_query_with_structured_data():
    """Test minions_query with complex structured data."""
    print("🧪 Testing minions_query with structured data...")
    
    energy_data = {
        "renewable_capacity_2024": {
            "solar": {"capacity_gw": 1200, "growth_rate": 0.25, "lcoe_per_kwh": 0.048},
            "wind": {"capacity_gw": 900, "growth_rate": 0.18, "lcoe_per_kwh": 0.041},
            "hydro": {"capacity_gw": 1350, "growth_rate": 0.02, "lcoe_per_kwh": 0.037},
            "geothermal": {"capacity_gw": 16, "growth_rate": 0.05, "lcoe_per_kwh": 0.072}
        },
        "storage_technologies": {
            "lithium_ion": {"cost_per_kwh": 132, "efficiency": 0.92, "cycle_life": 6000},
            "pumped_hydro": {"cost_per_kwh": 165, "efficiency": 0.85, "capacity_tw": 1.6},
            "compressed_air": {"cost_per_kwh": 140, "efficiency": 0.75, "duration_hours": 8}
        },
        "market_projections": {
            "2025": {"total_renewable_investment_billion": 380, "coal_retirement_gw": 85},
            "2030": {"total_renewable_investment_billion": 520, "coal_retirement_gw": 200},
            "2035": {"total_renewable_investment_billion": 680, "coal_retirement_gw": 320}
        }
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
                        {"kind": "text", "text": "Perform comprehensive analysis of renewable energy trends and create strategic recommendations"},
                        {"kind": "data", "data": energy_data}
                    ]
                },
                "metadata": {"skill_id": "minions_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        task = result["result"]
        
        completed_task = await wait_for_completion(client, task["id"], timeout=120)
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        
        artifact = completed_task["artifacts"][0]
        analysis = artifact["parts"][0]["text"].lower()
        
        # Should analyze multiple data dimensions in parallel
        assert len(analysis) > 300, f"Analysis too short: {len(analysis)}"
        has_capacity = "capacity" in analysis or "growth" in analysis
        has_strategy = "strategic" in analysis or "recommendation" in analysis
        assert has_capacity, f"Missing capacity analysis: {analysis[:200]}..."
        assert has_strategy, f"Missing strategic content: {analysis[:200]}..."
        
    print("✅ Structured data test passed")


async def test_minions_query_with_files():
    """Test minions_query with multiple file documents."""
    print("🧪 Testing minions_query with multiple files...")
    
    # Create multiple documents for parallel processing
    market_report = """
    ENERGY MARKET ANALYSIS Q4 2024
    
    Key Findings:
    - Solar installations grew 28% YoY globally
    - Wind energy costs decreased 12% in major markets  
    - Energy storage deployments doubled compared to 2023
    - Grid modernization investments reached $45B globally
    
    Regional Performance:
    Asia-Pacific leads renewable capacity additions with 65% of global installations.
    Europe focuses on offshore wind with 15GW of new capacity.
    North America emphasizes grid-scale storage integration.
    
    Technology Trends:
    Floating solar projects expand to water-constrained regions.
    Vertical axis wind turbines gain traction in urban environments.
    Green hydrogen production scales up with electrolysis efficiency improvements.
    """
    
    policy_brief = """
    RENEWABLE ENERGY POLICY UPDATE 2024
    
    Global Policy Framework:
    - 156 countries have net-zero commitments by 2050
    - Carbon pricing mechanisms cover 23% of global emissions
    - Renewable energy subsidies total $634B annually
    - Phase-out policies target 1,200GW of coal capacity by 2030
    
    Regulatory Changes:
    Updated grid codes accommodate 80% renewable penetration.
    Streamlined permitting processes reduce project timelines by 40%.
    International cooperation agreements facilitate technology transfer.
    
    Economic Incentives:
    Production tax credits extended through 2032 in major markets.
    Green bonds finance $2.3T in renewable infrastructure projects.
    Carbon border adjustments protect domestic clean energy industries.
    """
    
    # Encode documents as files
    market_bytes = market_report.encode('utf-8')
    market_b64 = base64.b64encode(market_bytes).decode('utf-8')
    
    policy_bytes = policy_brief.encode('utf-8') 
    policy_b64 = base64.b64encode(policy_bytes).decode('utf-8')
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "Synthesize market trends and policy impacts to recommend optimal renewable energy investment strategy"},
                        {
                            "kind": "file",
                            "file": {
                                "name": "market_analysis_q4.txt",
                                "mimeType": "text/plain",
                                "bytes": market_b64
                            }
                        },
                        {
                            "kind": "file", 
                            "file": {
                                "name": "policy_brief_2024.txt",
                                "mimeType": "text/plain",
                                "bytes": policy_b64
                            }
                        }
                    ]
                },
                "metadata": {"skill_id": "minions_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        task = result["result"]
        
        completed_task = await wait_for_completion(client, task["id"], timeout=150)
        assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
        
        artifact = completed_task["artifacts"][0]
        synthesis = artifact["parts"][0]["text"].lower()
        
        # Should synthesize information from multiple documents
        assert len(synthesis) > 400, f"Synthesis too short: {len(synthesis)}"
        has_market_policy = "market" in synthesis and "policy" in synthesis
        has_investment = "investment" in synthesis or "strategy" in synthesis
        assert has_market_policy, f"Missing market/policy synthesis: {synthesis[:200]}..."
        assert has_investment, f"Missing investment strategy: {synthesis[:200]}..."
        
    print("✅ Multiple files test passed")


async def test_skill_detection_parallel_keywords():
    """Test that parallel processing keywords trigger minions_query skill."""
    print("🧪 Testing automatic skill detection for parallel processing...")
    
    parallel_queries = [
        "Compare and analyze multiple renewable energy options",
        "Evaluate various investment strategies across different sectors", 
        "Assess multiple approaches to climate change mitigation"
    ]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for i, query in enumerate(parallel_queries):
            print(f"  Testing query {i+1}/{len(parallel_queries)}...")
            
            request = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "id": str(uuid.uuid4()),
                    "message": {
                        "role": "user",
                        "parts": [
                            {"kind": "text", "text": query}
                        ]
                    }
                    # No explicit skill_id - let system detect
                },
                "id": str(uuid.uuid4())
            }
            
            response = await client.post(f"{BASE_URL}/", json=request)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            result = response.json()
            task = result["result"]
            
            # Should auto-detect minions_query for parallel processing
            completed_task = await wait_for_completion(client, task["id"], timeout=90)
            assert completed_task["status"]["state"] == "completed", f"Task failed: {completed_task['status']}"
            
            # Should use parallel processing protocol
            artifact = completed_task["artifacts"][0]
            # Note: The artifact name might be "Minions Execution Result" if auto-detection uses minion_query
            # or "Parallel Minions Execution Result" if it uses minions_query
            artifact_name = artifact["name"]
            is_valid_name = artifact_name in ["Minions Execution Result", "Parallel Minions Execution Result"]
            assert is_valid_name, f"Unexpected artifact name: {artifact_name}"
            
    print("✅ Skill detection test passed")


async def test_error_handling():
    """Test error handling in minions_query parallel processing."""
    print("🧪 Testing error handling...")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Test with empty query
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": []
                },
                "metadata": {"skill_id": "minions_query"}
            },
            "id": str(uuid.uuid4())
        }
        
        response = await client.post(f"{BASE_URL}/", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        
        # Should handle error gracefully
        if "error" in result:
            assert result["error"]["code"] is not None, "Error should have code"
            print("✅ Error handling test passed (returned error)")
        else:
            task = result["result"]
            completed_task = await wait_for_completion(client, task["id"])
            final_state = completed_task["status"]["state"]
            assert final_state in ["completed", "failed"], f"Unexpected final state: {final_state}"
            print("✅ Error handling test passed (graceful handling)")


async def wait_for_completion(client: httpx.AsyncClient, task_id: str, timeout: int = 60) -> Dict[str, Any]:
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
    """Run all minions_query tests."""
    
    print("🚀 Running minions_query integration tests")
    print("=" * 50)
    
    # Check server health first
    if not await check_server_health():
        print("💡 Start the server with: python run_server.py")
        return False
    
    tests = [
        test_minions_query_multiple_questions,
        test_minions_query_with_large_document,
        test_minions_query_with_structured_data,
        test_minions_query_with_files,
        test_skill_detection_parallel_keywords,
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