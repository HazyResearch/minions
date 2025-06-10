"""
Pydantic models for A2A-Minions request/response validation.
"""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, validator, model_validator
import uuid
from datetime import datetime


# Constants for validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TEXT_LENGTH = 100_000  # 100k characters
MAX_PARTS = 10  # Maximum number of parts in a message
ALLOWED_FILE_TYPES = {
    "application/pdf",
    "text/plain", 
    "text/markdown",
    "text/csv",
    "application/json",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp"
}


class FilePart(BaseModel):
    """File information in a message part."""
    name: str = Field(..., min_length=1, max_length=255)
    mimeType: str
    bytes: str  # Base64 encoded
    uri: Optional[str] = None
    
    @validator('mimeType')
    def validate_mime_type(cls, v):
        if v not in ALLOWED_FILE_TYPES:
            raise ValueError(f"File type {v} not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}")
        return v
    
    @validator('bytes')
    def validate_file_size(cls, v):
        # Base64 increases size by ~33%, so adjust limit
        if len(v) > MAX_FILE_SIZE * 1.34:
            raise ValueError(f"File size exceeds maximum of {MAX_FILE_SIZE/1024/1024}MB")
        return v


class MessagePart(BaseModel):
    """A part of an A2A message."""
    kind: Literal["text", "file", "data"]
    text: Optional[str] = Field(None, max_length=MAX_TEXT_LENGTH)
    file: Optional[FilePart] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def validate_content_matches_kind(self):
        """Validate that the appropriate content field is set based on kind."""
        if self.kind == 'text':
            if not self.text:
                raise ValueError("Text part must have text content")
        elif self.kind == 'file':
            if not self.file:
                raise ValueError("File part must have file content")
        elif self.kind == 'data':
            if not self.data:
                raise ValueError("Data part must have data content")
        
        return self


class A2AMessage(BaseModel):
    """A2A message format."""
    role: Literal["user", "agent"]
    parts: List[MessagePart] = Field(..., min_items=1, max_items=MAX_PARTS)
    messageId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskMetadata(BaseModel):
    """Metadata for task execution."""
    skill_id: str = Field(..., description="Required skill ID")
    
    # Model configuration
    local_provider: Optional[str] = "ollama"
    local_model: Optional[str] = "llama3.2"
    local_temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0)
    local_max_tokens: Optional[int] = Field(4096, gt=0)
    
    remote_provider: Optional[str] = "openai"
    remote_model: Optional[str] = "gpt-4o"
    remote_temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0)
    remote_max_tokens: Optional[int] = Field(4096, gt=0)
    
    # Protocol parameters
    max_rounds: Optional[int] = Field(3, ge=1, le=10)
    timeout: Optional[int] = Field(300, ge=10, le=3600)  # 10s to 1 hour
    
    # Privacy
    privacy_mode: Optional[bool] = False
    
    # Minions-specific
    max_jobs_per_round: Optional[int] = Field(None, ge=1, le=100)
    num_tasks_per_round: Optional[int] = Field(3, ge=1, le=10)
    num_samples_per_task: Optional[int] = Field(1, ge=1, le=5)
    chunking_strategy: Optional[str] = "chunk_by_section"
    use_retrieval: Optional[Literal["bm25", "embedding", "multimodal-embedding"]] = None


class SendTaskParams(BaseModel):
    """Parameters for tasks/send and tasks/sendSubscribe."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    message: A2AMessage
    metadata: Optional[TaskMetadata] = None
    
    @validator('metadata', pre=True, always=True)
    def ensure_metadata(cls, v, values):
        """Ensure metadata exists and has skill_id."""
        # Convert TaskMetadata instance to dict if needed
        if isinstance(v, TaskMetadata):
            metadata_dict = v.dict()
        elif v is None:
            metadata_dict = {}
        else:
            metadata_dict = v
        
        # Check if skill_id is in the message metadata
        message = values.get('message')
        if message and hasattr(message, 'metadata') and message.metadata:
            if 'skill_id' in message.metadata and 'skill_id' not in metadata_dict:
                metadata_dict['skill_id'] = message.metadata['skill_id']
        
        # Validate we have a skill_id somewhere
        if not metadata_dict.get('skill_id'):
            # Default to minion_query if not specified
            metadata_dict['skill_id'] = 'minion_query'
        
        # Return as TaskMetadata instance
        return TaskMetadata(**metadata_dict)


class GetTaskParams(BaseModel):
    """Parameters for tasks/get."""
    id: str = Field(..., min_length=1)


class CancelTaskParams(BaseModel):
    """Parameters for tasks/cancel."""
    id: str = Field(..., min_length=1)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request format."""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str = Field(..., min_length=1)
    params: Optional[Dict[str, Any]] = {}
    id: Optional[str] = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error format."""
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response format."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


class TaskStatus(BaseModel):
    """Task status information."""
    state: str
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Task(BaseModel):
    """Task representation."""
    id: str
    sessionId: str
    status: TaskStatus
    history: List[A2AMessage]
    metadata: Optional[Dict[str, Any]] = {}
    artifacts: List[Dict[str, Any]] = []