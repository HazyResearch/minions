"""
A2A Server implementation for Minions protocol.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from .config import ConfigManager, MinionsConfig
from .agent_cards import get_default_agent_card, get_extended_agent_card
from .converters import A2AConverter, A2AMessage, MessagePart
from .client_factory import client_factory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState:
    """Task state management."""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskManager:
    """Manages A2A tasks and their execution."""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_streams: Dict[str, asyncio.Queue] = {}
        self.converter = A2AConverter()
        self.config_manager = ConfigManager()
    
    async def create_task(self, task_id: str, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new task."""
        
        task = {
            "id": task_id,
            "sessionId": str(uuid.uuid4()),
            "status": {
                "state": TaskState.SUBMITTED,
                "timestamp": datetime.now().isoformat()
            },
            "history": [message],
            "metadata": metadata or {},
            "artifacts": []
        }
        
        self.tasks[task_id] = task
        return task
    
    async def execute_task(self, task_id: str) -> None:
        """Execute a Minions task asynchronously."""
        
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return
        
        task = self.tasks[task_id]
        
        try:
            # Update task status
            await self.update_task_status(task_id, TaskState.WORKING, "Starting Minions execution")
            
            # Get last message from history
            last_message = task["history"][-1]
            
            # Handle both old A2AMessage format and new parts-based format
            if "parts" in last_message:
                parts = last_message["parts"]
                if not parts or len(parts) == 0:
                    # Empty parts array - treat as error
                    raise ValueError("Empty parts array provided")
                
                # New parts-based format with valid parts
                minions_task, context, image_paths = await self.converter.extract_query_and_document_from_parts(parts)
                # Create a minimal A2AMessage for skill detection
                a2a_message = A2AMessage(role="user", parts=[{"kind": "text", "text": minions_task}])
            else:
                # Legacy A2AMessage format or no parts key
                if not isinstance(last_message, dict) or not last_message:
                    raise ValueError("Invalid message format - empty or malformed message")
                
                # Try to create A2AMessage from legacy format
                try:
                    a2a_message = A2AMessage(**last_message)
                    minions_task, context, image_paths = await self.converter.extract_task_and_context(a2a_message)
                except Exception as e:
                    raise ValueError(f"Failed to parse legacy A2A message format: {e}")
            
            # Parse configuration from metadata
            config = self.config_manager.parse_a2a_metadata(task.get("metadata"))
            
            # Determine skill to use
            skill_id = self.converter.extract_skill_id(a2a_message, task.get("metadata"))
            logger.info(f"Executing skill: {skill_id} for task: {task_id}")
            
            # Set up streaming callback (sync, stores events for async processing)
            def streaming_callback(role: str, message: Any, is_final: bool = True):
                if task_id in self.task_streams:
                    event = self.converter.create_streaming_event(role, message, is_final)
                    # Store event in a thread-safe way for async processing
                    try:
                        # Use asyncio's thread-safe method to put item in queue
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self.task_streams[task_id].put(event))
                        )
                    except Exception as e:
                        logger.warning(f"Failed to enqueue streaming event: {e}")
            
            # Execute the appropriate skill
            result = await self._execute_skill(skill_id, minions_task, context, image_paths, config, streaming_callback)
            
            # Convert result to A2A artifact
            artifact = self.converter.convert_minions_result_to_a2a(result)
            
            # Update task with final result
            task["artifacts"].append(artifact)
            await self.update_task_status(task_id, TaskState.COMPLETED, "Task completed successfully")
            
            # Send final streaming event
            if task_id in self.task_streams:
                final_event = {
                    "kind": "taskStatusUpdate",
                    "state": TaskState.COMPLETED,
                    "final": True,
                    "timestamp": datetime.now().isoformat()
                }
                await self.task_streams[task_id].put(final_event)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            await self.update_task_status(task_id, TaskState.FAILED, f"Task failed: {str(e)}")
            
            # Send error streaming event
            if task_id in self.task_streams:
                error_event = {
                    "kind": "error",
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    },
                    "final": True
                }
                await self.task_streams[task_id].put(error_event)
    
    async def _execute_skill(self, skill_id: str, task: str, context: List[str], 
                           image_paths: List[str], config: MinionsConfig, 
                           callback) -> Dict[str, Any]:
        """Execute a specific Minions skill."""
        
        if skill_id == "minion_query":
            return await self._execute_minion_query(task, context, image_paths, config, callback)
        
        elif skill_id == "minions_query":
            return await self._execute_minions_query(task, context, config, callback)
        
        else:
            # Default to minion query
            return await self._execute_minion_query(task, context, image_paths, config, callback)
    
    async def _execute_minion_query(self, task: str, context: List[str], 
                                   image_paths: List[str], config: MinionsConfig, 
                                   callback) -> Dict[str, Any]:
        """Execute Minion (singular) query."""
        
        # Set protocol to MINION for this specific execution
        config.protocol = "minion"
        
        # Create Minion instance with callback
        minion = client_factory.create_minions_protocol(config)
        minion.callback = callback  # Set the callback after creation

        print(f"context length: {len(context)}, items: {len(context) if isinstance(context, list) else 'not a list'}")
        
        # Validate context before proceeding
        if not context or (isinstance(context, list) and all(not c.strip() for c in context)):
            logger.warning("No meaningful context provided for task")
            context = ["No specific context was provided for this task."]
        else:
            # Add a clear header to help the worker understand it has access to this information
            context_header = """IMPORTANT: You have been provided with documents and information below. When asked about papers, research, or specific statistics, you MUST refer to the content provided below. Do NOT say you don't have access - the information is available in the context below:"""
            context = [context_header] + context
        
        # Log context details for debugging
        if isinstance(context, list):
            total_chars = sum(len(c) for c in context)
            logger.info(f"Context validation: {len(context)} items, {total_chars} total characters")
            for i, c in enumerate(context):
                logger.info(f"Context item {i+1}: {len(c)} chars - {c[:100]}...")
            
            # Debug: Log what will actually be sent to minion
            joined_context = "\n\n".join(context)
            logger.info(f"JOINED CONTEXT LENGTH: {len(joined_context)} chars")
            logger.info(f"JOINED CONTEXT PREVIEW: {joined_context[:500]}...")
        
        
        # Execute Minion protocol
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: minion(
                task=task,
                context=context,
                max_rounds=config.max_rounds,
                images=image_paths if image_paths else None,
                is_privacy=config.privacy_mode
            )
        )
        
        return result
    
    async def _execute_minions_query(self, task: str, context: List[str], 
                                   config: MinionsConfig, callback) -> Dict[str, Any]:
        """Execute Minions (parallel) query."""
        
        # Set protocol to MINIONS for this specific execution
        config.protocol = "minions"
        
        # Create Minions instance with callback
        minions = client_factory.create_minions_protocol(config)
        minions.callback = callback  # Set the callback after creation
        
        # Generate document metadata
        doc_metadata = f"Documents provided. Total extracted text length: {sum(len(c) for c in context)} characters"
        
        # Execute Minions protocol
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: minions(
                task=task,
                doc_metadata=doc_metadata,
                context=context,
                max_rounds=config.max_rounds,
                max_jobs_per_round=config.max_jobs_per_round,
                num_tasks_per_round=config.num_tasks_per_round,
                num_samples_per_task=config.num_samples_per_task,
                use_retrieval=config.use_retrieval,
                chunk_fn=config.chunking_strategy
            )
        )
        
        return result
    
    async def update_task_status(self, task_id: str, state: str, message: Optional[str] = None):
        """Update task status."""
        
        if task_id not in self.tasks:
            return
        
        self.tasks[task_id]["status"] = {
            "state": state,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    async def setup_streaming(self, task_id: str) -> asyncio.Queue:
        """Set up streaming for a task."""
        queue = asyncio.Queue()
        self.task_streams[task_id] = queue
        return queue
    
    async def cleanup_streaming(self, task_id: str):
        """Clean up streaming for a task."""
        if task_id in self.task_streams:
            del self.task_streams[task_id]


class A2AMinionsServer:
    """A2A Server for Minions protocol."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, base_url: Optional[str] = None):
        self.host = host
        self.port = port
        self.base_url = base_url or f"http://{host}:{port}"
        self.app = FastAPI(title="A2A Minions Server", version="0.1.0")
        self.task_manager = TaskManager()
        
        # Set up routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Get the public agent card."""
            card = get_default_agent_card(self.base_url)
            return card.dict(exclude_none=True)
        
        @self.app.get("/agent/authenticatedExtendedCard")
        async def get_extended_card():
            """Get the extended agent card for authenticated users."""
            card = get_extended_agent_card(self.base_url)
            return card.dict(exclude_none=True)
        
        @self.app.post("/")
        async def handle_a2a_request(request: Request, background_tasks: BackgroundTasks):
            """Handle A2A JSON-RPC requests."""
            
            request_id = None
            try:
                # Parse request body
                try:
                    body = await request.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON body: {e}")
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                            "data": "Invalid JSON in request body"
                        }
                    }, status_code=400)
                
                # Validate JSON-RPC structure
                if not isinstance(body, dict):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": "Request body must be a JSON object"
                        }
                    }, status_code=400)
                
                # Extract JSON-RPC components
                method = body.get("method")
                params = body.get("params", {})
                request_id = body.get("id")
                
                # Validate method
                if not method:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": "Missing 'method' field"
                        }
                    }, status_code=400)
                
                # Route to appropriate handler
                if method == "tasks/send":
                    return await self._handle_send_task(params, request_id, background_tasks)
                
                elif method == "tasks/sendSubscribe":
                    return await self._handle_send_task_streaming(params, request_id, background_tasks)
                
                elif method == "tasks/get":
                    return await self._handle_get_task(params, request_id)
                
                elif method == "tasks/cancel":
                    return await self._handle_cancel_task(params, request_id)
                
                else:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": "Method not found",
                            "data": f"Unknown method: {method}"
                        }
                    }, status_code=400)
                    
            except ValueError as e:
                # Handle validation errors from task processing
                logger.error(f"Validation error: {e}")
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": str(e)
                    }
                }, status_code=400)
                
            except Exception as e:
                logger.error(f"Unexpected error handling request: {e}")
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    }
                }, status_code=500)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "a2a-minions"}
    
    async def _handle_send_task(self, params: Dict[str, Any], request_id: str, 
                              background_tasks: BackgroundTasks) -> JSONResponse:
        """Handle tasks/send requests."""
        
        # Validate required parameters
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary")
        
        message = params.get("message")
        if not message:
            raise ValueError("Missing required 'message' parameter")
        
        # Validate message structure
        if not isinstance(message, dict):
            raise ValueError("Message parameter must be a dictionary")
        
        # Validate parts if present
        if "parts" in message:
            parts = message["parts"]
            if not parts or len(parts) == 0:
                raise ValueError("Empty parts array provided")
        
        task_id = params.get("id", str(uuid.uuid4()))
        metadata = params.get("metadata")
        
        # Create task
        task = await self.task_manager.create_task(task_id, message, metadata)
        
        # Execute task in background
        background_tasks.add_task(self.task_manager.execute_task, task_id)
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task
        })
    
    async def _handle_send_task_streaming(self, params: Dict[str, Any], request_id: str,
                                        background_tasks: BackgroundTasks) -> EventSourceResponse:
        """Handle tasks/sendSubscribe requests with streaming."""
        
        # Validate required parameters
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary")
        
        message = params.get("message")
        if not message:
            raise ValueError("Missing required 'message' parameter")
        
        # Validate message structure
        if not isinstance(message, dict):
            raise ValueError("Message parameter must be a dictionary")
        
        # Validate parts if present
        if "parts" in message:
            parts = message["parts"]
            if not parts or len(parts) == 0:
                raise ValueError("Empty parts array provided")
        
        task_id = params.get("id", str(uuid.uuid4()))
        metadata = params.get("metadata")
        
        # Set up streaming
        stream_queue = await self.task_manager.setup_streaming(task_id)
        
        # Create task
        await self.task_manager.create_task(task_id, message, metadata)
        
        # Execute task in background
        background_tasks.add_task(self.task_manager.execute_task, task_id)
        
        async def event_generator():
            """Generate SSE events."""
            try:
                while True:
                    event = await stream_queue.get()
                    
                    # Format as JSON-RPC response
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": event
                    }
                    
                    yield {"data": json.dumps(response)}
                    
                    # Check if this is the final event
                    if event.get("final", False):
                        break
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_response = {
                    "jsonrpc": "2.0", 
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Streaming error",
                        "data": str(e)
                    }
                }
                yield {"data": json.dumps(error_response)}
            
            finally:
                await self.task_manager.cleanup_streaming(task_id)
        
        return EventSourceResponse(event_generator())
    
    async def _handle_get_task(self, params: Dict[str, Any], request_id: str) -> JSONResponse:
        """Handle tasks/get requests."""
        
        task_id = params.get("id")
        task = await self.task_manager.get_task(task_id)
        
        if task is None:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Task not found"
                }
            }, status_code=404)
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task
        })
    
    async def _handle_cancel_task(self, params: Dict[str, Any], request_id: str) -> JSONResponse:
        """Handle tasks/cancel requests."""
        
        task_id = params.get("id")
        task = await self.task_manager.get_task(task_id)
        
        if task is None:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Task not found"
                }
            }, status_code=404)
        
        # Update task status to canceled
        await self.task_manager.update_task_status(task_id, TaskState.CANCELED, "Task canceled by user")
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task
        })
    
    def run(self):
        """Run the server."""
        logger.info(f"Starting A2A Minions Server at {self.base_url}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="A2A Minions Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--base-url", help="Base URL for agent card")
    
    args = parser.parse_args()
    
    server = A2AMinionsServer(
        host=args.host,
        port=args.port,
        base_url=args.base_url
    )
    
    server.run()


if __name__ == "__main__":
    main() 