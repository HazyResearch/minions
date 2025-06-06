"""
A2A Server implementation for Minions protocol.
"""

import asyncio
import json
import logging
import uuid
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import ValidationError
import uvicorn
from collections import OrderedDict

from .config import ConfigManager, MinionsConfig
from .agent_cards import get_default_agent_card, get_extended_agent_card
from .converters import A2AConverter
from .client_factory import client_factory
from .models import (
    A2AMessage, MessagePart, SendTaskParams, GetTaskParams, 
    CancelTaskParams, JSONRPCRequest, JSONRPCResponse, 
    JSONRPCError, Task, TaskStatus, TaskMetadata
)
from .auth import init_auth, get_auth_manager, AuthConfig, TokenData
from .metrics import metrics_manager, create_metrics_endpoint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    
    def __init__(self, max_tasks: int = 1000, retention_time: timedelta = timedelta(hours=24)):
        self.tasks: OrderedDict[str, Dict[str, Any]] = OrderedDict()  # Use OrderedDict for LRU
        self.task_streams: Dict[str, asyncio.Queue] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.converter = A2AConverter()
        self.config_manager = ConfigManager()
        self.max_tasks = max_tasks
        self.retention_time = retention_time
        self._cleanup_task = None
        self._metrics_task = None
        self._shutdown_event = asyncio.Event()
        self._start_cleanup_task()
        self._start_metrics_task()
    
    def _start_cleanup_task(self):
        """Start background task for periodic cleanup."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    await self._cleanup_old_tasks()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def _start_metrics_task(self):
        """Start background task for periodic metrics updates."""
        async def metrics_loop():
            while True:
                try:
                    await asyncio.sleep(30)  # Update every 30 seconds
                    # Update task counts
                    metrics_manager.update_stored_tasks(len(self.tasks))
                    metrics_manager.update_streaming_sessions(len(self.task_streams))
                    
                    # Update client pool stats
                    pool_stats = client_factory.get_pool_stats()
                    provider_stats = {}
                    # Parse pool stats to get per-provider counts
                    if pool_stats.get("total_clients", 0) > 0:
                        # Simple approximation - in real implementation, 
                        # client_factory should provide detailed stats
                        provider_stats["total"] = pool_stats["total_clients"]
                    metrics_manager.update_client_pool_stats(provider_stats)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in metrics task: {e}")
        
        self._metrics_task = asyncio.create_task(metrics_loop())
    
    async def _cleanup_old_tasks(self):
        """Remove old completed tasks based on retention policy."""
        now = datetime.now()
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            # Skip active tasks
            if task_id in self.active_tasks:
                continue
                
            # Check if task is old enough
            created_at_str = task.get("metadata", {}).get("created_at")
            if created_at_str:
                created_at = datetime.fromisoformat(created_at_str)
                if now - created_at > self.retention_time:
                    tasks_to_remove.append(task_id)
            
            # Also check task completion time
            status = task.get("status", {})
            if status.get("state") in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                timestamp = status.get("timestamp")
                if timestamp:
                    completed_at = datetime.fromisoformat(timestamp)
                    if now - completed_at > timedelta(hours=1):  # Keep completed tasks for 1 hour
                        tasks_to_remove.append(task_id)
        
        # Remove old tasks
        for task_id in tasks_to_remove:
            logger.info(f"Removing old task: {task_id}")
            del self.tasks[task_id]
            metrics_manager.track_task_eviction()
    
    def _enforce_task_limit(self):
        """Enforce maximum task limit using LRU eviction."""
        while len(self.tasks) > self.max_tasks:
            # Remove oldest task that is not active
            for task_id in list(self.tasks.keys()):
                if task_id not in self.active_tasks:
                    logger.info(f"Evicting task due to limit: {task_id}")
                    del self.tasks[task_id]
                    metrics_manager.track_task_eviction()
                    break
    
    async def create_task(self, task_id: str, message: A2AMessage, metadata: Optional[TaskMetadata] = None, 
                         user: Optional[str] = None) -> Task:
        """Create a new task."""
        
        # Check task limit before creating
        self._enforce_task_limit()
        
        # Generate a session ID for this task
        session_id = f"session_{task_id}"
        
        # Create initial history with the incoming message
        history = [message]
        
        # Create task status object
        task_status = TaskStatus(
            state=TaskState.SUBMITTED,
            message=None,
            timestamp=datetime.now().isoformat()
        )
        
        # Build metadata including user ownership
        task_metadata = metadata.dict() if metadata else {}
        if user:
            task_metadata["created_by"] = user
        task_metadata["created_at"] = datetime.now().isoformat()
        
        task = Task(
            id=task_id,
            sessionId=session_id,
            status=task_status,
            history=history,
            metadata=task_metadata,
            artifacts=[]
        )
        
        self.tasks[task_id] = task.dict()
        
        return task
    
    async def execute_task(self, task_id: str) -> None:
        """Execute a task using the Minions protocol."""
        
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return
        
        task = self.tasks[task_id]
        skill_id = None
        
        try:
            # Update task status
            await self.update_task_status(task_id, TaskState.WORKING)
            
            # Extract message from history and metadata
            # The first message in history is the original user message
            message = A2AMessage(**task["history"][0])
            metadata = TaskMetadata(**task.get("metadata", {}))
            
            # Get timeout from metadata
            timeout = metadata.timeout
            skill_id = metadata.skill_id
            
            logger.info(f"Executing skill: {skill_id} for task: {task_id} with timeout: {timeout}s")
            
            with metrics_manager.track_task(skill_id):
                # Define streaming callback for real-time updates
                def streaming_callback(role: str, message: Any, is_final: bool = True):
                    """Callback to send streaming updates."""
                    if task_id not in self.task_streams:
                        return
                    
                    # Format the message properly
                    if isinstance(message, str):
                        text = message
                    else:
                        text = str(message)
                    
                    # Create streaming event
                    event = {
                        "kind": "message",
                        "parts": [{
                            "kind": "text",
                            "text": text
                        }],
                        "metadata": {
                            "original_role": role,
                            "timestamp": datetime.now().isoformat()
                        },
                        "final": is_final
                    }
                    
                    # Track streaming event
                    metrics_manager.track_streaming_event()
                    
                    # Thread-safe queue update using run_coroutine_threadsafe
                    async def put_event():
                        queue = self.task_streams.get(task_id)
                        if queue:
                            await queue.put(event)
                    
                    # Get the running event loop
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # If no loop is running, try to get the main loop
                        loop = asyncio.get_event_loop()
                    
                    # Schedule the coroutine in a thread-safe manner
                    asyncio.run_coroutine_threadsafe(put_event(), loop)
                
                # Execute the appropriate skill with timeout
                result = await asyncio.wait_for(
                    self._execute_skill(skill_id, message, metadata, streaming_callback),
                    timeout=timeout
                )
            
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
        
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after {timeout} seconds")
            await self.update_task_status(task_id, TaskState.FAILED, f"Task timed out after {timeout} seconds")
            
            # Send timeout error streaming event
            if task_id in self.task_streams:
                error_event = {
                    "kind": "error",
                    "error": {
                        "code": -32603,
                        "message": "Task timeout",
                        "data": f"Task exceeded timeout of {timeout} seconds"
                    },
                    "final": True
                }
                await self.task_streams[task_id].put(error_event)
        
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
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
        
        finally:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_skill(self, skill_id: str, message: A2AMessage, 
                           metadata: TaskMetadata, callback) -> Dict[str, Any]:
        """Execute a specific Minions skill."""
        
        # Extract task and context from message
        task, context, image_paths = await self.converter.extract_query_and_document_from_parts(
            [part.dict() for part in message.parts]
        )
        
        # Parse configuration from metadata
        config = self.config_manager.parse_a2a_metadata(metadata.dict())
        
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

        logger.debug(f"Context length: {len(context)} items")
        
        # Validate context before proceeding
        valid_context = [c for c in context if c and c.strip()]
        if valid_context:
            # Add a clear header to help the worker understand it has access to this information
            context_header = """IMPORTANT: You have been provided with documents and information below. When asked about papers, research, or specific statistics, you MUST refer to the content provided below. Do NOT say you don't have access - the information is available in the context below:"""
            context = [context_header] + valid_context
        else:
            logger.warning("No meaningful context provided for task")
            context = ["No specific context was provided for this task."]
        
        # Log context details for debugging
        if isinstance(context, list):
            total_chars = sum(len(c) for c in context)
            logger.info(f"Context validation: {len(context)} items, {total_chars} total characters")
            for i, c in enumerate(context[:3]):  # Log first 3 items only
                logger.debug(f"Context item {i+1}: {len(c)} chars - {c[:100]}...")
        
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
        
        # Create new TaskStatus object
        new_status = TaskStatus(
            state=state,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        
        self.tasks[task_id]["status"] = new_status.dict()
    
    async def get_task(self, task_id: str, user: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get task by ID, optionally checking ownership."""
        task = self.tasks.get(task_id)
        
        # If user is provided, check ownership
        if task and user and task.get("metadata", {}).get("created_by") != user:
            return None  # Don't reveal task exists to unauthorized users
            
        return task
    
    async def setup_streaming(self, task_id: str) -> asyncio.Queue:
        """Set up streaming for a task."""
        queue = asyncio.Queue()
        self.task_streams[task_id] = queue
        return queue
    
    async def cleanup_streaming(self, task_id: str):
        """Clean up streaming for a task."""
        if task_id in self.task_streams:
            del self.task_streams[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            await self.update_task_status(task_id, TaskState.CANCELED, "Task canceled by user")
            return True
        return False
    
    async def shutdown(self):
        """Gracefully shutdown task manager."""
        logger.info("Shutting down task manager...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel metrics task
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active tasks
        for task_id, task in list(self.active_tasks.items()):
            logger.info(f"Canceling active task: {task_id}")
            task.cancel()
        
        # Wait for tasks to complete (with timeout)
        if self.active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                    timeout=10
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within shutdown timeout")
        
        # Shutdown converter (includes temp file cleanup and thread pool shutdown)
        self.converter.shutdown()
        
        logger.info("Task manager shutdown complete")


class A2AMinionsServer:
    """A2A Server for Minions protocol."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, base_url: Optional[str] = None, 
                 auth_config: Optional[AuthConfig] = None):
        self.host = host
        self.port = port
        self.base_url = base_url or f"http://{host}:{port}"
        self.app = FastAPI(title="A2A Minions Server", version="1.0.0")
        self.task_manager = TaskManager()
        self._shutdown_event = asyncio.Event()
        
        # Initialize authentication
        self.auth_config = auth_config or AuthConfig()
        self.auth_manager = init_auth(self.auth_config)
        
        # Set up routes
        self._setup_routes()
        
        # Set up shutdown handlers
        self._setup_shutdown_handlers()
    
    def _setup_shutdown_handlers(self):
        """Set up graceful shutdown handlers."""
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.task_manager.shutdown()
        
        # Handle signals for graceful shutdown
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Get the public agent card."""
            card = get_default_agent_card(self.base_url)
            return card.dict(exclude_none=True)
        
        @self.app.get("/agent/authenticatedExtendedCard")
        async def get_extended_card(token_data: TokenData = Depends(self.auth_manager.authenticate)):
            """Get the extended agent card for authenticated users."""
            card = get_extended_agent_card(self.base_url)
            return card.dict(exclude_none=True)
        
        @self.app.post("/oauth/token")
        async def oauth_token(
            grant_type: str = Form(...),
            client_id: str = Form(...),
            client_secret: str = Form(...),
            scope: str = Form(default="")
        ):
            """Handle OAuth2 token requests."""
            return await self.auth_manager.handle_oauth_token(
                grant_type, client_id, client_secret, scope
            )
        
        # Add metrics endpoint
        self.app.get("/metrics")(create_metrics_endpoint())
        
        @self.app.post("/")
        async def handle_a2a_request(
            request: Request, 
            background_tasks: BackgroundTasks,
            token_data: Optional[TokenData] = Depends(
                self.auth_manager.authenticate if self.auth_config.require_auth else lambda: None
            )
        ):
            """Handle A2A JSON-RPC requests."""
            
            request_id = None
            method = None
            
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
                
                # Parse and validate JSON-RPC request
                try:
                    rpc_request = JSONRPCRequest(**body)
                    request_id = rpc_request.id
                    method = rpc_request.method
                except ValidationError as e:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": body.get("id") if isinstance(body, dict) else None,
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": str(e)
                        }
                    }, status_code=400)
                
                # Track request
                with metrics_manager.track_request(method):
                    # Check authorization for method
                    if self.auth_config.require_auth and token_data:
                        method_scopes = self.auth_config.allowed_scopes.get(
                            rpc_request.method, 
                            ["tasks:read", "tasks:write"]
                        )
                        if not self.auth_manager.check_scopes(token_data, method_scopes):
                            return JSONResponse({
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": "Forbidden",
                                    "data": f"Insufficient permissions. Required scopes: {method_scopes}"
                                }
                            }, status_code=403)
                    
                    # Route to appropriate handler
                    user = token_data.sub if token_data else None
                    
                    if rpc_request.method == "tasks/send":
                        return await self._handle_send_task(rpc_request.params, request_id, background_tasks, user)
                    
                    elif rpc_request.method == "tasks/sendSubscribe":
                        return await self._handle_send_task_streaming(rpc_request.params, request_id, background_tasks, user)
                    
                    elif rpc_request.method == "tasks/get":
                        return await self._handle_get_task(rpc_request.params, request_id, user)
                    
                    elif rpc_request.method == "tasks/cancel":
                        return await self._handle_cancel_task(rpc_request.params, request_id, user)
                    
                    else:
                        return JSONResponse({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Method not found",
                                "data": f"Unknown method: {rpc_request.method}"
                            }
                        }, status_code=400)
                    
            except ValidationError as e:
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
                logger.error(f"Unexpected error handling request: {e}", exc_info=True)
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
                              background_tasks: BackgroundTasks, user: Optional[str] = None) -> JSONResponse:
        """Handle tasks/send requests."""
        
        # Validate and parse parameters
        try:
            send_params = SendTaskParams(**params)
        except ValidationError as e:
            # Re-raise the pydantic ValidationError directly
            raise e
        
        # Create task
        task = await self.task_manager.create_task(
            send_params.id,
            send_params.message,
            send_params.metadata,
            user
        )
        
        # Execute task in background
        async_task = asyncio.create_task(self.task_manager.execute_task(send_params.id))
        self.task_manager.active_tasks[send_params.id] = async_task
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task.dict()
        })
    
    async def _handle_send_task_streaming(self, params: Dict[str, Any], request_id: str,
                                        background_tasks: BackgroundTasks, user: Optional[str] = None) -> EventSourceResponse:
        """Handle tasks/sendSubscribe requests with streaming."""
        
        # Validate and parse parameters
        try:
            send_params = SendTaskParams(**params)
        except ValidationError as e:
            # For streaming, we need to return error in SSE format
            error_msg = str(e)  # Capture the error message
            async def error_generator():
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": error_msg
                    }
                }
                yield {"data": json.dumps(error_response)}
            
            return EventSourceResponse(error_generator())
        
        # Set up streaming
        stream_queue = await self.task_manager.setup_streaming(send_params.id)
        
        # Create task
        await self.task_manager.create_task(
            send_params.id,
            send_params.message,
            send_params.metadata,
            user
        )
        
        # Execute task in background
        async_task = asyncio.create_task(self.task_manager.execute_task(send_params.id))
        self.task_manager.active_tasks[send_params.id] = async_task
        
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
                await self.task_manager.cleanup_streaming(send_params.id)
        
        return EventSourceResponse(event_generator())
    
    async def _handle_get_task(self, params: Dict[str, Any], request_id: str, user: Optional[str] = None) -> JSONResponse:
        """Handle tasks/get requests."""
        
        # Validate parameters
        try:
            get_params = GetTaskParams(**params)
        except ValidationError as e:
            # Re-raise the pydantic ValidationError directly
            raise e
        
        task = await self.task_manager.get_task(get_params.id, user)
        
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
    
    async def _handle_cancel_task(self, params: Dict[str, Any], request_id: str, user: Optional[str] = None) -> JSONResponse:
        """Handle tasks/cancel requests."""
        
        # Validate parameters
        try:
            cancel_params = CancelTaskParams(**params)
        except ValidationError as e:
            # Re-raise the pydantic ValidationError directly
            raise e
        
        task = await self.task_manager.get_task(cancel_params.id, user)
        
        if task is None:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Task not found"
                }
            }, status_code=404)
        
        # Cancel the task
        canceled = await self.task_manager.cancel_task(cancel_params.id)
        
        if canceled:
            # Get updated task
            task = await self.task_manager.get_task(cancel_params.id, user)
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task
        })
    
    def run(self):
        """Run the server."""
        logger.info(f"Starting A2A Minions Server at {self.base_url}")
        
        async def serve():
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Run server with shutdown event
            serve_task = asyncio.create_task(server.serve())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            
            done, pending = await asyncio.wait(
                {serve_task, shutdown_task},
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Shutdown server if it's still running
            if serve_task in pending:
                server.should_exit = True
                await server.shutdown()
        
        asyncio.run(serve())


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