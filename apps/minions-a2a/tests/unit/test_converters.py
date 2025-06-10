#!/usr/bin/env python3
"""
Unit tests for A2A-Minions converters.
Tests message conversion, file handling, and data transformation.
"""

import unittest
import base64
import json
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from a2a_minions.converters import A2AConverter, MinionsResult
from a2a_minions.models import A2AMessage, MessagePart, FilePart


class TestMinionsResult(unittest.TestCase):
    """Test MinionsResult class."""
    
    def test_minions_result_creation(self):
        """Test creating MinionsResult object."""
        result = MinionsResult(
            final_answer="Test answer",
            conversation=[{"role": "user", "content": "Test"}],
            usage={"total_tokens": 100},
            timing={"total_time": 1.5},
            metadata={"source": "test"}
        )
        
        self.assertEqual(result.final_answer, "Test answer")
        self.assertEqual(len(result.conversation), 1)
        self.assertEqual(result.usage["total_tokens"], 100)
        self.assertEqual(result.timing["total_time"], 1.5)
        self.assertEqual(result.metadata["source"], "test")


class TestA2AConverter(unittest.TestCase):
    """Test A2AConverter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = A2AConverter()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.converter.shutdown()
    
    async def test_extract_task_and_context_text_only(self):
        """Test extracting task and context from text-only message."""
        message = A2AMessage(
            role="user",
            parts=[
                MessagePart(kind="text", text="What is the capital of France?")
            ]
        )
        
        task, context, images = await self.converter.extract_task_and_context(message)
        
        self.assertEqual(task, "What is the capital of France?")
        self.assertEqual(context, [])
        self.assertEqual(images, [])
    
    async def test_extract_task_and_context_with_additional_text(self):
        """Test extracting task and context with multiple text parts."""
        message = A2AMessage(
            role="user",
            parts=[
                MessagePart(kind="text", text="Summarize this document"),
                MessagePart(kind="text", text="Document content here")
            ]
        )
        
        task, context, images = await self.converter.extract_task_and_context(message)
        
        self.assertEqual(task, "Summarize this document")
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0], "Document content here")
    
    async def test_extract_task_and_context_with_data(self):
        """Test extracting task and context with data part."""
        data = {"key": "value", "number": 42}
        message = A2AMessage(
            role="user",
            parts=[
                MessagePart(kind="text", text="Analyze this data"),
                MessagePart(kind="data", data=data)
            ]
        )
        
        task, context, images = await self.converter.extract_task_and_context(message)
        
        self.assertEqual(task, "Analyze this data")
        self.assertEqual(len(context), 1)
        self.assertIn("JSON DATA", context[0])
        self.assertIn('"key": "value"', context[0])
    
    async def test_extract_query_and_document_query_only(self):
        """Test extracting query without document."""
        parts = [
            {"kind": "text", "text": "What is machine learning?"}
        ]
        
        query, context, images = await self.converter.extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "What is machine learning?")
        self.assertEqual(context, [])
        self.assertEqual(images, [])
    
    async def test_extract_query_and_document_with_text(self):
        """Test extracting query with text document."""
        parts = [
            {"kind": "text", "text": "Summarize this"},
            {"kind": "text", "text": "Important document content"}
        ]
        
        query, context, images = await self.converter.extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "Summarize this")
        self.assertEqual(len(context), 1)
        self.assertIn("DOCUMENT CONTENT", context[0])
        self.assertIn("Important document content", context[0])
    
    async def test_extract_query_and_document_with_json_data(self):
        """Test extracting query with JSON data document."""
        parts = [
            {"kind": "text", "text": "Analyze sales data"},
            {"kind": "data", "data": {"sales": 1000, "region": "North"}}
        ]
        
        query, context, images = await self.converter.extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "Analyze sales data")
        self.assertEqual(len(context), 1)
        self.assertIn("JSON DATA", context[0])
        self.assertIn('"sales": 1000', context[0])
    
    async def test_extract_query_and_document_empty_parts(self):
        """Test extraction with empty parts raises error."""
        with self.assertRaises(ValueError) as context:
            await self.converter.extract_query_and_document_from_parts([])
        self.assertIn("No parts provided", str(context.exception))
    
    async def test_extract_query_and_document_non_text_first(self):
        """Test extraction with non-text first part raises error."""
        parts = [{"kind": "file", "file": {"name": "test.pdf"}}]
        
        with self.assertRaises(ValueError) as context:
            await self.converter.extract_query_and_document_from_parts(parts)
        self.assertIn("First part must be text", str(context.exception))
    
    async def test_extract_query_and_document_empty_query(self):
        """Test extraction with empty query text raises error."""
        parts = [{"kind": "text", "text": ""}]
        
        with self.assertRaises(ValueError) as context:
            await self.converter.extract_query_and_document_from_parts(parts)
        self.assertIn("Query text is empty", str(context.exception))
    
    async def test_extract_file_content_base64(self):
        """Test extracting content from base64 encoded file."""
        text_content = "Hello, world!"
        base64_content = base64.b64encode(text_content.encode()).decode()
        
        file_info = {
            "name": "test.txt",
            "mimeType": "text/plain",
            "bytes": base64_content
        }
        
        content = await self.converter._extract_file_content(file_info)
        self.assertEqual(content, text_content)
    
    async def test_extract_file_content_pdf(self):
        """Test extracting content from PDF file."""
        # Create a simple test PDF content (not a real PDF)
        pdf_content = b"Mock PDF content"
        base64_content = base64.b64encode(pdf_content).decode()
        
        file_info = {
            "name": "document.pdf",
            "mimeType": "application/pdf",
            "bytes": base64_content
        }
        
        # Mock PDF extraction
        with patch.object(self.converter, '_extract_pdf_text', 
                         return_value="Extracted PDF text") as mock_extract:
            content = await self.converter._extract_file_content(file_info)
        
        mock_extract.assert_called_once()
        self.assertEqual(content, "Extracted PDF text")
    
    async def test_extract_file_content_image(self):
        """Test extracting content from image file."""
        image_content = b"fake image data"
        base64_content = base64.b64encode(image_content).decode()
        
        file_info = {
            "name": "image.jpg",
            "mimeType": "image/jpeg",
            "bytes": base64_content
        }
        
        content = await self.converter._extract_file_content(file_info)
        self.assertEqual(content, "[Image file: image.jpg]")
    
    async def test_extract_file_content_uri(self):
        """Test extracting content from URI reference."""
        file_info = {
            "name": "remote.pdf",
            "uri": "https://example.com/file.pdf"
        }
        
        content = await self.converter._extract_file_content(file_info)
        self.assertEqual(content, "[External file: https://example.com/file.pdf]")
    
    async def test_extract_file_content_error(self):
        """Test error handling in file extraction."""
        file_info = {
            "name": "bad.txt",
            "bytes": "invalid base64"
        }
        
        content = await self.converter._extract_file_content(file_info)
        self.assertIn("[Error reading file:", content)
    
    def test_is_image_file(self):
        """Test image file detection."""
        # Image files
        self.assertTrue(self.converter._is_image_file({"mimeType": "image/jpeg"}))
        self.assertTrue(self.converter._is_image_file({"mimeType": "image/png"}))
        self.assertTrue(self.converter._is_image_file({"mimeType": "image/gif"}))
        
        # Non-image files
        self.assertFalse(self.converter._is_image_file({"mimeType": "text/plain"}))
        self.assertFalse(self.converter._is_image_file({"mimeType": "application/pdf"}))
        self.assertFalse(self.converter._is_image_file({}))
    
    async def test_save_temp_file(self):
        """Test saving file to temporary location."""
        content = "Test file content"
        base64_content = base64.b64encode(content.encode()).decode()
        
        file_info = {
            "name": "test.txt",
            "bytes": base64_content
        }
        
        path = await self.converter._save_temp_file(file_info)
        
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())
        
        # Read back content
        with open(path, 'rb') as f:
            saved_content = f.read()
        self.assertEqual(saved_content.decode(), content)
    
    async def test_save_temp_file_sanitization(self):
        """Test filename sanitization in temp file saving."""
        content = b"test"
        base64_content = base64.b64encode(content).decode()
        
        # Dangerous filename
        file_info = {
            "name": "../../../etc/passwd",
            "bytes": base64_content
        }
        
        path = await self.converter._save_temp_file(file_info)
        
        # Should not contain path traversal
        self.assertNotIn("..", path)
        self.assertIn("etcpasswd", path)  # Sanitized name
    
    def test_convert_minions_result_to_a2a_dict(self):
        """Test converting Minions result dict to A2A format."""
        result = {
            "final_answer": "The answer is 42",
            "conversation": [
                {"role": "user", "content": "What is the answer?"},
                {"role": "assistant", "content": "The answer is 42"}
            ],
            "usage": {"total_tokens": 50},
            "timing": {"total_time": 2.5}
        }
        
        artifact = self.converter.convert_minions_result_to_a2a(result)
        
        # Check structure
        self.assertIn("name", artifact)
        self.assertIn("parts", artifact)
        self.assertEqual(len(artifact["parts"]), 2)
        
        # Check answer part
        answer_part = artifact["parts"][0]
        self.assertEqual(answer_part["kind"], "text")
        self.assertEqual(answer_part["text"], "The answer is 42")
        
        # Check conversation part
        conv_part = artifact["parts"][1]
        self.assertEqual(conv_part["kind"], "data")
        self.assertEqual(len(conv_part["data"]["conversation"]), 2)
    
    def test_convert_minions_result_to_a2a_object(self):
        """Test converting MinionsResult object to A2A format."""
        result = MinionsResult(
            final_answer="Test answer",
            conversation=[],
            usage={"tokens": 100},
            timing={"time": 1.0},
            metadata={"test": True}
        )
        
        artifact = self.converter.convert_minions_result_to_a2a(result)
        
        self.assertEqual(artifact["parts"][0]["text"], "Test answer")
        self.assertEqual(artifact["metadata"]["test"], True)
    
    def test_serialize_usage(self):
        """Test usage object serialization."""
        # Simple dict
        usage = {"total_tokens": 100, "prompt_tokens": 50}
        serialized = self.converter._serialize_usage(usage)
        self.assertEqual(serialized, usage)
        
        # Object with to_dict method
        mock_usage = MagicMock()
        mock_usage.to_dict.return_value = {"tokens": 200}
        serialized = self.converter._serialize_usage(mock_usage)
        self.assertEqual(serialized, {"tokens": 200})
        
        # Object with __dict__
        class MockUsage:
            def __init__(self):
                self.tokens = 300
                self.time = 2.5
        
        mock_usage = MockUsage()
        serialized = self.converter._serialize_usage(mock_usage)
        self.assertEqual(serialized["tokens"], 300)
        self.assertEqual(serialized["time"], 2.5)
    
    def test_create_streaming_event(self):
        """Test creating A2A streaming event."""
        # Supervisor message
        event = self.converter.create_streaming_event(
            "supervisor",
            "Processing task",
            is_final=False
        )
        
        self.assertEqual(event["kind"], "message")
        self.assertEqual(event["role"], "agent")
        self.assertEqual(event["parts"][0]["text"], "Processing task")
        self.assertFalse(event["metadata"]["is_final"])
        
        # Worker message with dict
        event = self.converter.create_streaming_event(
            "worker",
            {"content": "Working on it"},
            is_final=True
        )
        
        self.assertEqual(event["parts"][0]["text"], "Working on it")
        self.assertTrue(event["metadata"]["is_final"])
    
    def test_format_jobs_list(self):
        """Test formatting jobs list for display."""
        # Empty list
        formatted = self.converter._format_jobs_list([])
        self.assertEqual(formatted, "No jobs completed.")
        
        # Jobs with answers
        job1 = MagicMock()
        job1.output.answer = "First answer"
        
        job2 = MagicMock()
        job2.output.answer = "Second answer"
        
        job3 = MagicMock()
        job3.output.answer = "none"  # Should be filtered
        
        formatted = self.converter._format_jobs_list([job1, job2, job3])
        self.assertIn("Job 1:", formatted)
        self.assertIn("First answer", formatted)
        self.assertIn("Job 2:", formatted)
        self.assertIn("Second answer", formatted)
        self.assertNotIn("Job 3:", formatted)
    
    def test_extract_skill_id(self):
        """Test skill ID extraction."""
        message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="Test")]
        )
        
        # From metadata parameter
        skill_id = self.converter.extract_skill_id(message, {"skill_id": "test_skill"})
        self.assertEqual(skill_id, "test_skill")
        
        # From message metadata
        message.metadata = {"skill_id": "message_skill"}
        skill_id = self.converter.extract_skill_id(message, {})
        self.assertEqual(skill_id, "message_skill")
        
        # Priority: metadata parameter over message metadata
        skill_id = self.converter.extract_skill_id(message, {"skill_id": "override"})
        self.assertEqual(skill_id, "override")
        
        # Missing skill_id
        message.metadata = {}
        with self.assertRaises(ValueError) as context:
            self.converter.extract_skill_id(message, {})
        self.assertIn("No skill_id found", str(context.exception))
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        # Create a temp file
        test_file = self.converter.temp_dir / "test.txt"
        test_file.write_text("test")
        
        self.assertTrue(test_file.exists())
        
        # Cleanup
        self.converter.cleanup_temp_files()
        
        # Temp dir should be recreated but empty
        self.assertTrue(self.converter.temp_dir.exists())
        self.assertFalse(test_file.exists())


class TestPDFExtraction(unittest.TestCase):
    """Test PDF extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = A2AConverter()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.converter.shutdown()
    
    @patch('a2a_minions.converters.PDF_AVAILABLE', True)
    @patch('PyPDF2.PdfReader')
    async def test_extract_pdf_text_success(self, mock_pdf_reader):
        """Test successful PDF text extraction."""
        # Mock PDF pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        pdf_bytes = b"fake pdf content"
        result = await self.converter._extract_pdf_text(pdf_bytes, "test.pdf")
        
        self.assertIn("Page 1", result)
        self.assertIn("Page 1 content", result)
        self.assertIn("Page 2", result)
        self.assertIn("Page 2 content", result)
    
    @patch('a2a_minions.converters.PDF_AVAILABLE', True)
    @patch('PyPDF2.PdfReader')
    async def test_extract_pdf_text_error(self, mock_pdf_reader):
        """Test PDF extraction with error."""
        mock_pdf_reader.side_effect = Exception("PDF error")
        
        pdf_bytes = b"fake pdf content"
        result = await self.converter._extract_pdf_text(pdf_bytes, "test.pdf")
        
        self.assertIn("[PDF file: test.pdf", result)
        self.assertIn("Error extracting text", result)
    
    @patch('a2a_minions.converters.PDF_AVAILABLE', False)
    async def test_extract_pdf_text_no_pypdf2(self):
        """Test PDF extraction when PyPDF2 is not available."""
        pdf_bytes = b"fake pdf content"
        result = await self.converter._extract_pdf_text(pdf_bytes, "test.pdf")
        
        self.assertIn("[PDF file: test.pdf", result)
        self.assertIn("PyPDF2 not installed", result)


if __name__ == "__main__":
    unittest.main()