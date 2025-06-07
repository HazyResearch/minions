#!/usr/bin/env python3
"""
Unit tests for A2A-Minions server.
Tests server endpoints, task management, and streaming functionality.
"""

import unittest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import HTTPException

from a2a_minions.server import (
    TaskState, TaskManager, A2AMinionsServer
)
from a2a_minions.models import (
    A2AMessage, MessagePart, TaskMetadata, TaskStatus, Task
)
from a2a_minions.auth import AuthConfig


class TestTaskState(unittest.TestCase):
    """Test TaskState constants."""
    
    def test_task_states(self):
        """Test task state constants."""
        self.assertEqual(TaskState.SUBMITTED, "submitted")
        self.assertEqual(TaskState.WORKING, "working")
        self.assertEqual(TaskState.COMPLETED, "completed")
        self.assertEqual(TaskState.FAILED, "failed")
        self.assertEqual(TaskState.CANCELED, "canceled")


class TestTaskManager(unittest.TestCase):
    """Test TaskManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.task_manager = TaskManager(max_tasks=5, retention_time=timedelta(hours=1))
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.loop.close()
    
    def test_create_task(self):
        """Test task creation."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test task")]
        )
        metadata = TaskMetadata(skill_id="test_skill")
        
        task = self.loop.run_until_complete(
            self.task_manager.create_task("task-123", message, metadata, "test-user")
        )
        
        self.assertEqual(task.id, "task-123")
        self.assertEqual(task.sessionId, "session_task-123")
        self.assertEqual(task.status.state, TaskState.SUBMITTED)
        self.assertEqual(len(task.history), 1)
        self.assertEqual(task.metadata["created_by"], "test-user")
        self.assertIn("task-123", self.task_manager.tasks)
    
    def test_task_limit_enforcement(self):
        """Test task limit enforcement with LRU eviction."""
        # Create max tasks
        for i in range(5):
            message = A2AMessage(
                role="user",
                parts=[MessagePart(kind="text", text=f"Task {i}")]
            )
            self.loop.run_until_complete(
                self.task_manager.create_task(f"task-{i}", message)
            )
        
        self.assertEqual(len(self.task_manager.tasks), 5)
        
        # Create one more task - should evict oldest
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="New task")]
        )
        self.loop.run_until_complete(
            self.task_manager.create_task("task-new", message)
        )
        
        # Should still have 5 tasks
        self.assertEqual(len(self.task_manager.tasks), 5)
        
        # Oldest task should be evicted
        self.assertNotIn("task-0", self.task_manager.tasks)
        self.assertIn("task-new", self.task_manager.tasks)
    
    def test_update_task_status(self):
        """Test updating task status."""
        # Create a task
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        self.loop.run_until_complete(
            self.task_manager.create_task("task-1", message)
        )
        
        # Update status
        self.loop.run_until_complete(
            self.task_manager.update_task_status("task-1", TaskState.WORKING, "Processing")
        )
        
        task = self.task_manager.tasks["task-1"]
        self.assertEqual(task["status"]["state"], TaskState.WORKING)
        self.assertEqual(task["status"]["message"], "Processing")
    
    def test_get_task_with_ownership(self):
        """Test getting task with ownership check."""
        # Create task owned by user1
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        self.loop.run_until_complete(
            self.task_manager.create_task("task-1", message, user="user1")
        )
        
        # Get task as owner
        task = self.loop.run_until_complete(
            self.task_manager.get_task("task-1", "user1")
        )
        self.assertIsNotNone(task)
        
        # Get task as different user - should return None
        task = self.loop.run_until_complete(
            self.task_manager.get_task("task-1", "user2")
        )
        self.assertIsNone(task)
        
        # Get task without user check
        task = self.loop.run_until_complete(
            self.task_manager.get_task("task-1")
        )
        self.assertIsNotNone(task)
    
    def test_setup_and_cleanup_streaming(self):
        """Test streaming setup and cleanup."""
        # Setup streaming
        queue = self.loop.run_until_complete(
            self.task_manager.setup_streaming("task-1")
        )
        
        self.assertIsNotNone(queue)
        self.assertIn("task-1", self.task_manager.task_streams)
        
        # Cleanup streaming
        self.loop.run_until_complete(
            self.task_manager.cleanup_streaming("task-1")
        )
        
        self.assertNotIn("task-1", self.task_manager.task_streams)
    
    def test_cancel_task(self):
        """Test task cancellation."""
        # Create and start a task
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Long task")]
        )
        self.loop.run_until_complete(
            self.task_manager.create_task("task-1", message)
        )
        
        # Mock active task
        mock_task = MagicMock()
        self.task_manager.active_tasks["task-1"] = mock_task
        
        # Cancel task
        result = self.loop.run_until_complete(
            self.task_manager.cancel_task("task-1")
        )
        
        self.assertTrue(result)
        mock_task.cancel.assert_called_once()
        
        # Cancel non-existent task
        result = self.loop.run_until_complete(
            self.task_manager.cancel_task("task-999")
        )
        self.assertFalse(result)


class TestTaskExecution(unittest.TestCase):
    """Test task execution functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.task_manager = TaskManager()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Mock converters and client factory
        self.mock_converter = MagicMock()
        self.task_manager.converter = self.mock_converter
        
        self.mock_config_manager = MagicMock()
        self.task_manager.config_manager = self.mock_config_manager
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.loop.close()
    
    @patch('a2a_minions.server.client_factory')
    def test_execute_task_success(self, mock_client_factory):
        """Test successful task execution."""
        # Create task
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test query")]
        )
        metadata = TaskMetadata(skill_id="minion_query", timeout=30)
        
        task = self.loop.run_until_complete(
            self.task_manager.create_task("task-1", message, metadata)
        )
        
        # Mock execution
        mock_protocol = MagicMock()
        mock_protocol.return_value = {
            "final_answer": "Test answer",
            "conversation": [],
            "usage": {},
            "timing": {}
        }
        mock_client_factory.create_minions_protocol.return_value = mock_protocol
        
        # Mock converter
        self.mock_converter.extract_query_and_document_from_parts = AsyncMock(
            return_value=("Test query", [], [])
        )
        self.mock_converter.convert_minions_result_to_a2a.return_value = {
            "name": "Result",
            "parts": [{"kind": "text", "text": "Test answer"}]
        }
        
        # Execute task
        self.loop.run_until_complete(
            self.task_manager.execute_task("task-1")
        )
        
        # Check task was updated
        updated_task = self.task_manager.tasks["task-1"]
        self.assertEqual(updated_task["status"]["state"], TaskState.COMPLETED)
        self.assertEqual(len(updated_task["artifacts"]), 1)
    
    @patch('a2a_minions.server.client_factory')
    def test_execute_task_timeout(self, mock_client_factory):
        """Test task execution timeout."""
        # Create task with short timeout
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test query")]
        )
        metadata = TaskMetadata(skill_id="minion_query", timeout=0.1)  # 100ms timeout
        
        task = self.loop.run_until_complete(
            self.task_manager.create_task("task-1", message, metadata)
        )
        
        # Mock slow execution
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(1)  # Sleep longer than timeout
            return {"final_answer": "Too slow"}
        
        # Mock setup
        self.mock_converter.extract_query_and_document_from_parts = AsyncMock(
            return_value=("Test query", [], [])
        )
        
        with patch.object(self.task_manager, '_execute_skill', slow_execution):
            # Execute task
            self.loop.run_until_complete(
                self.task_manager.execute_task("task-1")
            )
        
        # Check task failed due to timeout
        updated_task = self.task_manager.tasks["task-1"]
        self.assertEqual(updated_task["status"]["state"], TaskState.FAILED)
        self.assertIn("timed out", updated_task["status"]["message"])
    
    def test_execute_task_with_streaming(self):
        """Test task execution with streaming updates."""
        # Setup streaming
        queue = self.loop.run_until_complete(
            self.task_manager.setup_streaming("task-1")
        )
        
        # Create task
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        metadata = TaskMetadata(skill_id="minion_query")
        
        task = self.loop.run_until_complete(
            self.task_manager.create_task("task-1", message, metadata)
        )
        
        # Mock execution with streaming callback
        callback_called = []
        
        async def mock_execute_skill(skill_id, message, metadata, callback):
            # Test streaming callback
            callback("supervisor", "Starting task", False)
            callback_called.append(True)
            return {"final_answer": "Done", "conversation": [], "usage": {}, "timing": {}}
        
        # Mock converter
        self.mock_converter.extract_query_and_document_from_parts = AsyncMock(
            return_value=("Test", [], [])
        )
        self.mock_converter.convert_minions_result_to_a2a.return_value = {
            "name": "Result",
            "parts": [{"kind": "text", "text": "Done"}]
        }
        
        with patch.object(self.task_manager, '_execute_skill', mock_execute_skill):
            # Execute task
            self.loop.run_until_complete(
                self.task_manager.execute_task("task-1")
            )
        
        # Check callback was called
        self.assertTrue(callback_called)


class TestA2AMinionsServer(unittest.TestCase):
    """Test A2AMinionsServer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        auth_config = AuthConfig(require_auth=False)  # Disable auth for testing
        self.server = A2AMinionsServer(
            host="127.0.0.1",
            port=8001,
            base_url="http://localhost:8001",
            auth_config=auth_config
        )
        self.client = TestClient(self.server.app)
    
    def test_agent_card_endpoint(self):
        """Test agent card endpoint."""
        response = self.client.get("/.well-known/agent.json")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("name", data)
        self.assertIn("skills", data)
        self.assertIn("url", data)
        self.assertEqual(data["url"], "http://localhost:8001")
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("tasks_count", data)
    
    def test_send_task_endpoint(self):
        """Test tasks/send endpoint."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": "test-task-123",
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What is 2+2?"
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query"
                }
            },
            "id": "req-1"
        }
        
        # Mock task execution
        with patch.object(self.server.task_manager, 'execute_task', AsyncMock()):
            response = self.client.post("/", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["jsonrpc"], "2.0")
        self.assertEqual(data["id"], "req-1")
        self.assertIn("result", data)
        self.assertEqual(data["result"]["id"], "test-task-123")
        self.assertEqual(data["result"]["status"]["state"], "submitted")
    
    def test_get_task_endpoint(self):
        """Test tasks/get endpoint."""
        # First create a task
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        task = asyncio.run(
            self.server.task_manager.create_task("task-123", message)
        )
        
        # Get task
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "id": "task-123"
            },
            "id": "req-2"
        }
        
        response = self.client.post("/", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["id"], "req-2")
        self.assertIn("result", data)
        self.assertEqual(data["result"]["id"], "task-123")
    
    def test_cancel_task_endpoint(self):
        """Test tasks/cancel endpoint."""
        # First create a task
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        task = asyncio.run(
            self.server.task_manager.create_task("task-456", message)
        )
        
        # Cancel task
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {
                "id": "task-456"
            },
            "id": "req-3"
        }
        
        response = self.client.post("/", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["id"], "req-3")
        self.assertIn("result", data)
    
    def test_invalid_method(self):
        """Test invalid JSON-RPC method."""
        payload = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": "req-4"
        }
        
        response = self.client.post("/", json=payload)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32601)
        self.assertIn("Method not found", data["error"]["message"])
    
    def test_invalid_json(self):
        """Test invalid JSON request."""
        response = self.client.post(
            "/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32700)
    
    def test_missing_params(self):
        """Test missing required parameters."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {},  # Missing required params
            "id": "req-5"
        }
        
        response = self.client.post("/", json=payload)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32602)


class TestServerWithAuth(unittest.TestCase):
    """Test server with authentication enabled."""
    
    def setUp(self):
        """Set up test fixtures."""
        auth_config = AuthConfig(require_auth=True, jwt_secret="test-secret")
        self.server = A2AMinionsServer(
            host="127.0.0.1",
            port=8002,
            auth_config=auth_config
        )
        self.client = TestClient(self.server.app)
        
        # Get default API key
        self.api_key = self.server.auth_manager.default_api_key
    
    def test_authenticated_request(self):
        """Test request with valid authentication."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": "test"},
            "id": "req-1"
        }
        
        response = self.client.post(
            "/",
            json=payload,
            headers={"X-API-Key": self.api_key}
        )
        
        # Should work with valid API key
        self.assertNotEqual(response.status_code, 401)
    
    def test_unauthenticated_request(self):
        """Test request without authentication."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {"role": "user", "parts": [{"kind": "text", "text": "Test"}]}
            },
            "id": "req-2"
        }
        
        response = self.client.post("/", json=payload)
        
        self.assertEqual(response.status_code, 401)
        data = response.json()
        
        self.assertIn("error", data)
        self.assertIn("Authentication required", data["error"]["message"])
    
    def test_oauth_token_endpoint(self):
        """Test OAuth2 token endpoint."""
        # Get default client credentials
        oauth_client = list(self.server.auth_manager.oauth2_manager.clients.values())[0]
        client_id = oauth_client.client_id
        client_secret = oauth_client.client_secret
        
        response = self.client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "tasks:read tasks:write"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertIn("expires_in", data)


if __name__ == "__main__":
    unittest.main()