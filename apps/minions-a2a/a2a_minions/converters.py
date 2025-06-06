"""
Message conversion utilities for A2A-Minions integration.
Handles translation between A2A protocol messages and Minions format.
Supports PDF extraction, multi-modal processing, and parts-based A2A format.
"""

import base64
import json
import tempfile
import aiofiles
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import mimetypes

from .models import A2AMessage, MessagePart

# PDF processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class MinionsResult:
    """Minions execution result."""
    def __init__(self, final_answer: str, conversation: List[Dict[str, Any]], 
                 usage: Dict[str, Any], timing: Dict[str, Any], 
                 metadata: Optional[Dict[str, Any]] = None):
        self.final_answer = final_answer
        self.conversation = conversation
        self.usage = usage
        self.timing = timing
        self.metadata = metadata or {}


class A2AConverter:
    """Converts between A2A and Minions message formats."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "a2a_minions"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def extract_task_and_context(self, message: A2AMessage) -> Tuple[str, List[str], List[str]]:
        """
        Extract task, context, and file paths from A2A message.
        Legacy method for backwards compatibility.
        
        Returns:
            Tuple of (task, context_list, image_paths)
        """
        task = ""
        context = []
        image_paths = []
        
        for part in message.parts:
            if part.kind == "text" and part.text:
                if not task:
                    # First text part is the main task
                    task = part.text
                else:
                    # Additional text parts are context
                    context.append(part.text)
            
            elif part.kind == "file" and part.file:
                # Handle file parts
                file_content = await self._extract_file_content(part.file.dict())
                if file_content and not file_content.startswith("[Error"):
                    # Add clear document formatting for legacy file content
                    file_name = part.file.name
                    formatted_content = f"=== FILE CONTENT: {file_name} ===\n{file_content.strip()}\n=== END FILE ==="
                    context.append(formatted_content)
                
                # Check if it's an image file
                if self._is_image_file(part.file.dict()):
                    image_path = await self._save_temp_file(part.file.dict())
                    if image_path:
                        image_paths.append(image_path)
            
            elif part.kind == "data" and part.data:
                # Convert data parts to context
                data_str = json.dumps(part.data, indent=2)
                formatted_content = f"=== JSON DATA ===\n{data_str}\n=== END DATA ==="
                context.append(formatted_content)
        
        # If no explicit task found, use first context as task
        if not task and context:
            task = context.pop(0)
        
        return task, context, image_paths
    
    async def extract_query_and_document_from_parts(self, parts: List[Dict[str, Any]]) -> Tuple[str, List[str], List[str]]:
        """
        Extract query and document from A2A parts following the new format.
        Expected: First part = query (text), Second part = document (text/file/data).
        
        Args:
            parts: List of A2A parts
            
        Returns:
            Tuple of (task, context_list, image_paths)
        """
        if not parts:
            raise ValueError("No parts provided")
        
        # First part must be text containing the query
        if parts[0].get("kind") != "text":
            raise ValueError("First part must be text containing the query")
        
        task = parts[0].get("text", "")
        if not task:
            raise ValueError("Query text is empty")
        
        context = []
        image_paths = []
        
        # Process second part if it exists (the document)
        if len(parts) > 1:
            second_part = parts[1]
            kind = second_part.get("kind")
            
            if kind == "text":
                # Text content
                content = second_part.get("text", "")
                if content and content.strip():  # Check for non-empty content
                    # Add clear document formatting to help the worker understand it has access to this content
                    formatted_content = f"=== DOCUMENT CONTENT ===\n{content.strip()}\n=== END DOCUMENT ==="
                    context.append(formatted_content)
            
            elif kind == "file":
                # File content
                file_data = second_part.get("file", {})
                
                # Check if it's an image first
                if self._is_image_file(file_data):
                    image_path = await self._save_temp_file(file_data)
                    if image_path:
                        image_paths.append(image_path)
                else:
                    # Extract text content from file
                    content = await self._extract_file_content(file_data)
                    if content and content.strip() and not content.startswith("[Error"):
                        # Add clear document formatting for file content
                        file_name = file_data.get("name", "document")
                        formatted_content = f"=== FILE CONTENT: {file_name} ===\n{content.strip()}\n=== END FILE ==="
                        context.append(formatted_content)
            
            elif kind == "data":
                # Structured data
                data = second_part.get("data", {})
                if data:  # Only add non-empty data
                    formatted_data = json.dumps(data, indent=2)
                    # Add clear data formatting
                    formatted_content = f"=== JSON DATA ===\n{formatted_data}\n=== END DATA ==="
                    context.append(formatted_content)
        
        # If we still have no context, check if this was intentional (query-only) or an error
        if not context:
            logger.warning("No meaningful context extracted from parts")
            # For single-part messages (query only), this is normal
            if len(parts) == 1:
                logger.info("Single part message detected - query without context")
            else:
                logger.info(f"{len(parts)} parts provided but no extractable context found")
        
        return task, context, image_paths
    
    async def _extract_file_content(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Extract text content from file information."""
        try:
            if "bytes" in file_info and file_info["bytes"]:
                # Decode base64 content
                content_bytes = base64.b64decode(file_info["bytes"])
                mime_type = file_info.get("mimeType", "")
                
                # Handle PDF files specifically
                if mime_type == "application/pdf" or file_info.get("name", "").lower().endswith(".pdf"):
                    return await self._extract_pdf_text(content_bytes, file_info.get("name", "document.pdf"))
                
                # Try to decode as text
                try:
                    return content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # If not text, handle based on mime type
                    if mime_type.startswith("image/"):
                        # For images, return metadata
                        return f"[Image file: {file_info.get('name', 'unknown')}]"
                    else:
                        # For other binary files, return metadata
                        return f"[Binary file: {file_info.get('name', 'unknown')} ({mime_type})]"
            
            elif "uri" in file_info and file_info["uri"]:
                # Handle URI-based files
                return f"[External file: {file_info['uri']}]"
            
        except Exception as e:
            logger.error(f"Error extracting file content: {e}")
            return f"[Error reading file: {file_info.get('name', 'unknown')}]"
        
        return None
    
    async def _extract_pdf_text(self, pdf_bytes: bytes, filename: str) -> str:
        """Extract text from PDF bytes."""
        if not PDF_AVAILABLE:
            return f"[PDF file: {filename} - PyPDF2 not installed for text extraction]"
        
        try:
            # Save to temporary file for PyPDF2
            temp_path = self.temp_dir / f"temp_{filename}"
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(pdf_bytes)
            
            # Extract text using PyPDF2
            with open(temp_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"\n--- Page {page_num + 1} ---\n")
                            text_content.append(page_text)
                    except Exception as e:
                        text_content.append(f"\n--- Page {page_num + 1} (error: {e}) ---\n")
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
                
                if text_content:
                    return "".join(text_content)
                else:
                    return f"[PDF file: {filename} - No extractable text found]"
                    
        except Exception as e:
            return f"[PDF file: {filename} - Error extracting text: {e}]"
    
    def _is_image_file(self, file_info: Dict[str, Any]) -> bool:
        """Check if file is an image."""
        mime_type = file_info.get("mimeType", "")
        return mime_type.startswith("image/")
    
    async def _save_temp_file(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Save file content to temporary file and return path."""
        try:
            if "bytes" in file_info and file_info["bytes"]:
                content_bytes = base64.b64decode(file_info["bytes"])
                
                # Sanitize filename to prevent path traversal
                file_name = file_info.get("name", "temp_file")
                # Remove path separators and sanitize
                safe_name = os.path.basename(file_name)
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")
                
                # Generate unique filename to prevent overwrites
                unique_name = f"{uuid.uuid4()}_{safe_name}"
                temp_path = self.temp_dir / unique_name
                
                async with aiofiles.open(temp_path, "wb") as f:
                    await f.write(content_bytes)
                
                return str(temp_path)
                
        except Exception as e:
            logger.error(f"Error saving temp file: {e}")
            
        return None
    
    def convert_minions_result_to_a2a(self, result: Union[Dict[str, Any], MinionsResult]) -> Dict[str, Any]:
        """Convert Minions result to A2A artifact format."""
        
        # Handle both dict and MinionsResult objects
        if isinstance(result, dict):
            final_answer = result.get("final_answer", "")
            conversation = result.get("conversation", [])
            usage = result.get("usage", {})
            timing = result.get("timing", {})
            metadata = result.get("metadata", {})
        else:
            final_answer = result.final_answer
            conversation = result.conversation
            usage = result.usage
            timing = result.timing
            metadata = result.metadata or {}
        
        # Convert usage objects to serializable format
        serializable_usage = self._serialize_usage(usage)
        
        # Create main response part
        parts = [
            {
                "kind": "text",
                "text": final_answer,
                "metadata": {
                    "type": "final_answer"
                }
            }
        ]
        
        # Add conversation data if available
        if conversation:
            parts.append({
                "kind": "data", 
                "data": {
                    "conversation": conversation,
                    "usage": serializable_usage,
                    "timing": timing
                },
                "metadata": {
                    "type": "execution_details"
                }
            })
        
        return {
            "name": "Minions Execution Result",
            "description": "Result from Minions protocol execution",
            "parts": parts,
            "metadata": {
                **metadata,
                "usage": serializable_usage,
                "timing": timing,
                "protocol": "minions"
            }
        }
    
    def _serialize_usage(self, usage: Any) -> Dict[str, Any]:
        """Convert usage objects to JSON-serializable format."""
        # Check if usage object has to_dict method (for minions.usage.Usage objects)
        if hasattr(usage, 'to_dict') and callable(getattr(usage, 'to_dict')):
            return usage.to_dict()
        elif hasattr(usage, '__dict__'):
            # If it's an object with attributes, convert to dict
            usage_dict = {}
            for key, value in usage.__dict__.items():
                if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                    # Use to_dict for nested Usage objects
                    usage_dict[key] = value.to_dict()
                elif hasattr(value, '__dict__'):
                    # Recursively serialize nested objects
                    usage_dict[key] = self._serialize_usage(value)
                else:
                    usage_dict[key] = value
            return usage_dict
        elif isinstance(usage, dict):
            # If it's already a dict, recursively serialize values
            return {k: self._serialize_usage(v) for k, v in usage.items()}
        else:
            # If it's a primitive type, return as-is
            return usage
    
    def create_streaming_event(self, role: str, message: Any, is_final: bool = True) -> Dict[str, Any]:
        """Create A2A streaming event from Minions callback."""
        
        # Map Minions roles to A2A message roles
        a2a_role = "agent"  # Both supervisor and worker are "agent" from A2A perspective
        
        # Determine event type based on role and content
        if role == "supervisor":
            event_type = "supervisor_message"
        elif role == "worker":
            event_type = "worker_message"  
        else:
            event_type = "system_message"
        
        # Format message content
        if isinstance(message, dict):
            if "content" in message:
                content = message["content"]
            else:
                content = json.dumps(message, indent=2)
        elif isinstance(message, list):
            # Handle job lists from Minions protocol
            content = self._format_jobs_list(message)
        else:
            content = str(message)
        
        # Create message part
        parts = [{
            "kind": "text",
            "text": content,
            "metadata": {
                "role": role,
                "event_type": event_type,
                "is_final": is_final
            }
        }]
        
        return {
            "kind": "message",
            "role": a2a_role,
            "parts": parts,
            "messageId": f"minions_{role}_{hash(content) % 10000}",
            "metadata": {
                "source": "minions",
                "original_role": role,
                "event_type": event_type,
                "is_final": is_final
            }
        }
    
    def _format_jobs_list(self, jobs: List[Any]) -> str:
        """Format jobs list for display."""
        if not jobs:
            return "No jobs completed."
        
        lines = ["### Job Results:"]
        for i, job in enumerate(jobs):
            if hasattr(job, 'output') and hasattr(job.output, 'answer'):
                if job.output.answer and job.output.answer.lower().strip() != "none":
                    lines.append(f"**Job {i+1}:** {job.output.answer}")
        
        if len(lines) == 1:
            lines.append("No relevant information found in processed jobs.")
            
        return "\n".join(lines)
    
    def extract_skill_id(self, message: A2AMessage, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Extract skill ID from metadata or raise error if missing."""
        
        # Check metadata for explicit skill_id (this is how A2A clients should specify which skill to use)
        if metadata and "skill_id" in metadata:
            return metadata["skill_id"]
        
        # Check message metadata for skill_id
        if message.metadata and "skill_id" in message.metadata:
            return message.metadata["skill_id"]
        
        # If no skill_id found, raise an error
        raise ValueError("No skill_id found in metadata or message metadata. A skill_id is required.")
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}") 