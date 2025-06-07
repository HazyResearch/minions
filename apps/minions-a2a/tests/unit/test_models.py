#!/usr/bin/env python3
"""
Unit tests for A2A-Minions Pydantic models.
Tests all data models, validation rules, and edge cases.
"""

import unittest
import json
from datetime import datetime

from a2a_minions.models import (
    FilePart, MessagePart, A2AMessage, TaskMetadata, SendTaskParams,
    GetTaskParams, CancelTaskParams, JSONRPCRequest, JSONRPCResponse,
    JSONRPCError, TaskStatus, Task,
    MAX_FILE_SIZE, MAX_TEXT_LENGTH, MAX_PARTS, ALLOWED_FILE_TYPES
)


class TestFilePart(unittest.TestCase):
    """Test FilePart model validation."""
    
    def test_valid_file_part(self):
        """Test creating a valid FilePart."""
        file_part = FilePart(
            name="test.pdf",
            mimeType="application/pdf",
            bytes="dGVzdCBjb250ZW50"  # base64 encoded "test content"
        )
        self.assertEqual(file_part.name, "test.pdf")
        self.assertEqual(file_part.mimeType, "application/pdf")
        
    def test_invalid_mime_type(self):
        """Test that invalid MIME types are rejected."""
        with self.assertRaises(ValueError) as context:
            FilePart(
                name="test.exe",
                mimeType="application/x-executable",
                bytes="dGVzdA=="
            )
        self.assertIn("not allowed", str(context.exception))
    
    def test_file_size_limit(self):
        """Test that oversized files are rejected."""
        # Create base64 content that exceeds the limit
        large_content = "A" * int(MAX_FILE_SIZE * 1.5)
        
        with self.assertRaises(ValueError) as context:
            FilePart(
                name="large.txt",
                mimeType="text/plain",
                bytes=large_content
            )
        self.assertIn("exceeds maximum", str(context.exception))
    
    def test_empty_name(self):
        """Test that empty file names are rejected."""
        with self.assertRaises(ValueError):
            FilePart(
                name="",
                mimeType="text/plain",
                bytes="dGVzdA=="
            )
    
    def test_with_uri(self):
        """Test FilePart with URI field."""
        file_part = FilePart(
            name="remote.pdf",
            mimeType="application/pdf",
            bytes="dGVzdA==",
            uri="https://example.com/file.pdf"
        )
        self.assertEqual(file_part.uri, "https://example.com/file.pdf")


class TestMessagePart(unittest.TestCase):
    """Test MessagePart model validation."""
    
    def test_text_part(self):
        """Test creating a text MessagePart."""
        part = MessagePart(
            kind="text",
            text="Hello, world!"
        )
        self.assertEqual(part.kind, "text")
        self.assertEqual(part.text, "Hello, world!")
    
    def test_file_part(self):
        """Test creating a file MessagePart."""
        file_data = FilePart(
            name="test.pdf",
            mimeType="application/pdf",
            bytes="dGVzdA=="
        )
        part = MessagePart(
            kind="file",
            file=file_data
        )
        self.assertEqual(part.kind, "file")
        self.assertEqual(part.file.name, "test.pdf")
    
    def test_data_part(self):
        """Test creating a data MessagePart."""
        data = {"key": "value", "number": 42}
        part = MessagePart(
            kind="data",
            data=data
        )
        self.assertEqual(part.kind, "data")
        self.assertEqual(part.data["key"], "value")
    
    def test_text_part_missing_content(self):
        """Test that text parts must have text content."""
        with self.assertRaises(ValueError) as context:
            MessagePart(kind="text")
        self.assertIn("must have text content", str(context.exception))
    
    def test_file_part_missing_content(self):
        """Test that file parts must have file content."""
        with self.assertRaises(ValueError) as context:
            MessagePart(kind="file")
        self.assertIn("must have file content", str(context.exception))
    
    def test_data_part_missing_content(self):
        """Test that data parts must have data content."""
        with self.assertRaises(ValueError) as context:
            MessagePart(kind="data")
        self.assertIn("must have data content", str(context.exception))
    
    def test_text_length_limit(self):
        """Test that text content respects length limit."""
        long_text = "A" * (MAX_TEXT_LENGTH + 1)
        with self.assertRaises(ValueError):
            MessagePart(
                kind="text",
                text=long_text
            )
    
    def test_with_metadata(self):
        """Test MessagePart with metadata."""
        part = MessagePart(
            kind="text",
            text="Test",
            metadata={"source": "test", "timestamp": "2024-01-01"}
        )
        self.assertEqual(part.metadata["source"], "test")


class TestA2AMessage(unittest.TestCase):
    """Test A2AMessage model validation."""
    
    def test_valid_message(self):
        """Test creating a valid A2A message."""
        part = MessagePart(kind="text", text="Test message")
        message = A2AMessage(
            role="user",
            parts=[part]
        )
        self.assertEqual(message.role, "user")
        self.assertEqual(len(message.parts), 1)
    
    def test_agent_role(self):
        """Test message with agent role."""
        part = MessagePart(kind="text", text="Response")
        message = A2AMessage(
            role="agent",
            parts=[part]
        )
        self.assertEqual(message.role, "agent")
    
    def test_invalid_role(self):
        """Test that invalid roles are rejected."""
        part = MessagePart(kind="text", text="Test")
        with self.assertRaises(ValueError):
            A2AMessage(
                role="invalid",
                parts=[part]
            )
    
    def test_empty_parts(self):
        """Test that messages must have at least one part."""
        with self.assertRaises(ValueError):
            A2AMessage(
                role="user",
                parts=[]
            )
    
    def test_too_many_parts(self):
        """Test that messages can't exceed max parts limit."""
        parts = [
            MessagePart(kind="text", text=f"Part {i}")
            for i in range(MAX_PARTS + 1)
        ]
        with self.assertRaises(ValueError):
            A2AMessage(
                role="user",
                parts=parts
            )
    
    def test_with_message_id_and_metadata(self):
        """Test message with ID and metadata."""
        part = MessagePart(kind="text", text="Test")
        message = A2AMessage(
            role="user",
            parts=[part],
            messageId="msg-123",
            metadata={"client": "test-client"}
        )
        self.assertEqual(message.messageId, "msg-123")
        self.assertEqual(message.metadata["client"], "test-client")


class TestTaskMetadata(unittest.TestCase):
    """Test TaskMetadata model validation."""
    
    def test_minimal_metadata(self):
        """Test creating metadata with only required fields."""
        metadata = TaskMetadata(skill_id="minion_query")
        self.assertEqual(metadata.skill_id, "minion_query")
        
    def test_full_metadata(self):
        """Test metadata with all fields."""
        metadata = TaskMetadata(
            skill_id="minions_query",
            local_provider="ollama",
            local_model="llama3.2",
            local_temperature=0.5,
            local_max_tokens=2048,
            remote_provider="openai",
            remote_model="gpt-4o",
            remote_temperature=0.3,
            remote_max_tokens=8192,
            max_rounds=5,
            timeout=600,
            privacy_mode=True,
            max_jobs_per_round=50,
            num_tasks_per_round=5,
            num_samples_per_task=3,
            chunking_strategy="chunk_by_paragraph",
            use_retrieval="bm25"
        )
        self.assertEqual(metadata.max_rounds, 5)
        self.assertEqual(metadata.timeout, 600)
        self.assertTrue(metadata.privacy_mode)
    
    def test_temperature_validation(self):
        """Test temperature range validation."""
        # Valid temperature
        metadata = TaskMetadata(
            skill_id="test",
            local_temperature=1.5
        )
        self.assertEqual(metadata.local_temperature, 1.5)
        
        # Invalid temperature (too high)
        with self.assertRaises(ValueError):
            TaskMetadata(
                skill_id="test",
                local_temperature=2.5
            )
    
    def test_timeout_validation(self):
        """Test timeout range validation."""
        # Minimum timeout
        metadata = TaskMetadata(skill_id="test", timeout=10)
        self.assertEqual(metadata.timeout, 10)
        
        # Maximum timeout
        metadata = TaskMetadata(skill_id="test", timeout=3600)
        self.assertEqual(metadata.timeout, 3600)
        
        # Too short
        with self.assertRaises(ValueError):
            TaskMetadata(skill_id="test", timeout=5)
        
        # Too long
        with self.assertRaises(ValueError):
            TaskMetadata(skill_id="test", timeout=3700)
    
    def test_retrieval_options(self):
        """Test retrieval method validation."""
        for method in ["bm25", "embedding", "multimodal-embedding"]:
            metadata = TaskMetadata(skill_id="test", use_retrieval=method)
            self.assertEqual(metadata.use_retrieval, method)
        
        # Invalid retrieval method
        with self.assertRaises(ValueError):
            TaskMetadata(skill_id="test", use_retrieval="invalid")


class TestSendTaskParams(unittest.TestCase):
    """Test SendTaskParams model validation."""
    
    def test_minimal_params(self):
        """Test minimal send task parameters."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        params = SendTaskParams(message=message)
        
        # Should have auto-generated ID
        self.assertIsNotNone(params.id)
        self.assertEqual(len(params.id), 36)  # UUID length
        
        # Should have default metadata with skill_id
        self.assertIsNotNone(params.metadata)
        self.assertEqual(params.metadata.skill_id, "minion_query")
    
    def test_with_explicit_id(self):
        """Test params with explicit task ID."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        params = SendTaskParams(
            id="custom-id-123",
            message=message
        )
        self.assertEqual(params.id, "custom-id-123")
    
    def test_skill_id_from_message_metadata(self):
        """Test skill_id extraction from message metadata."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")],
            metadata={"skill_id": "minions_query"}
        )
        params = SendTaskParams(message=message)
        self.assertEqual(params.metadata.skill_id, "minions_query")
    
    def test_skill_id_override(self):
        """Test explicit skill_id overrides message metadata."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")],
            metadata={"skill_id": "minions_query"}
        )
        metadata = TaskMetadata(skill_id="minion_query")
        params = SendTaskParams(message=message, metadata=metadata)
        self.assertEqual(params.metadata.skill_id, "minion_query")


class TestGetTaskParams(unittest.TestCase):
    """Test GetTaskParams model validation."""
    
    def test_valid_params(self):
        """Test valid get task parameters."""
        params = GetTaskParams(id="task-123")
        self.assertEqual(params.id, "task-123")
    
    def test_empty_id(self):
        """Test that empty task ID is rejected."""
        with self.assertRaises(ValueError):
            GetTaskParams(id="")


class TestCancelTaskParams(unittest.TestCase):
    """Test CancelTaskParams model validation."""
    
    def test_valid_params(self):
        """Test valid cancel task parameters."""
        params = CancelTaskParams(id="task-456")
        self.assertEqual(params.id, "task-456")
    
    def test_empty_id(self):
        """Test that empty task ID is rejected."""
        with self.assertRaises(ValueError):
            CancelTaskParams(id="")


class TestJSONRPCModels(unittest.TestCase):
    """Test JSON-RPC model validation."""
    
    def test_jsonrpc_request(self):
        """Test JSONRPCRequest model."""
        request = JSONRPCRequest(
            method="tasks/send",
            params={"id": "123"},
            id="req-1"
        )
        self.assertEqual(request.jsonrpc, "2.0")
        self.assertEqual(request.method, "tasks/send")
        self.assertEqual(request.params["id"], "123")
    
    def test_jsonrpc_request_no_params(self):
        """Test request without params."""
        request = JSONRPCRequest(
            method="health",
            id="req-2"
        )
        self.assertEqual(request.params, {})
    
    def test_jsonrpc_error(self):
        """Test JSONRPCError model."""
        error = JSONRPCError(
            code=-32600,
            message="Invalid Request",
            data={"details": "Missing method"}
        )
        self.assertEqual(error.code, -32600)
        self.assertEqual(error.message, "Invalid Request")
    
    def test_jsonrpc_response_success(self):
        """Test successful JSON-RPC response."""
        response = JSONRPCResponse(
            id="req-1",
            result={"status": "ok"}
        )
        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.result["status"], "ok")
        self.assertIsNone(response.error)
    
    def test_jsonrpc_response_error(self):
        """Test error JSON-RPC response."""
        error = JSONRPCError(code=-32603, message="Internal error")
        response = JSONRPCResponse(
            id="req-1",
            error=error
        )
        self.assertIsNone(response.result)
        self.assertEqual(response.error.code, -32603)


class TestTaskModels(unittest.TestCase):
    """Test Task-related models."""
    
    def test_task_status(self):
        """Test TaskStatus model."""
        status = TaskStatus(
            state="working",
            message="Processing request"
        )
        self.assertEqual(status.state, "working")
        self.assertEqual(status.message, "Processing request")
        # Timestamp should be auto-generated
        self.assertIsNotNone(status.timestamp)
        
        # Verify timestamp format
        datetime.fromisoformat(status.timestamp)  # Should not raise
    
    def test_task_model(self):
        """Test complete Task model."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test task")]
        )
        status = TaskStatus(state="submitted")
        
        task = Task(
            id="task-789",
            sessionId="session-123",
            status=status,
            history=[message],
            metadata={"created_by": "test-user"},
            artifacts=[]
        )
        
        self.assertEqual(task.id, "task-789")
        self.assertEqual(task.sessionId, "session-123")
        self.assertEqual(task.status.state, "submitted")
        self.assertEqual(len(task.history), 1)
        self.assertEqual(task.metadata["created_by"], "test-user")


if __name__ == "__main__":
    unittest.main()