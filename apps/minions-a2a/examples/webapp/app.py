#!/usr/bin/env python3
"""
Web interface for A2A-Minions server.
Provides a user-friendly HTML interface for sending queries and viewing results.
"""

import asyncio
import json
import uuid
import base64
import time
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, stream_template
from flask_socketio import SocketIO, emit
import httpx
import threading
import queue

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a2a-minions-webapp-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
A2A_SERVER_URL = "http://localhost:8001"
API_KEY = "abcd"  # Default API key for testing

class A2AClient:
    """Async client for A2A-Minions server."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health", headers=self.headers)
            response.raise_for_status()
            return response.json()
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/.well-known/agent.json")
            response.raise_for_status()
            return response.json()
    
    async def send_task(self, message: Dict[str, Any], metadata: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Send a task to the A2A server."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": message,
                "metadata": metadata
            },
            "id": str(uuid.uuid4())
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
    
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": str(uuid.uuid4())
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
    
    async def send_task_streaming(self, message: Dict[str, Any], metadata: Dict[str, Any], task_id: str, callback):
        """Send task with streaming response."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/sendSubscribe", 
            "params": {
                "id": task_id,
                "message": message,
                "metadata": metadata
            },
            "id": str(uuid.uuid4())
        }
        
        stream_headers = self.headers.copy()
        stream_headers["Accept"] = "text/event-stream"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self.base_url, json=payload, headers=stream_headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            event = json.loads(data)
                            callback(event)
                            
                            if event.get("result", {}).get("final"):
                                break
                        except json.JSONDecodeError:
                            continue

# Context calculation is now handled by the A2A server

# Global client instance
a2a_client = A2AClient(A2A_SERVER_URL, API_KEY)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Check A2A server health."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        health_data = loop.run_until_complete(a2a_client.health_check())
        loop.close()
        return jsonify({"status": "ok", "a2a_server": health_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/agent-card')
def agent_card():
    """Get agent card."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        card_data = loop.run_until_complete(a2a_client.get_agent_card())
        loop.close()
        return jsonify(card_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/providers')
def get_providers():
    """Get available model providers."""
    providers = {
        "local": ["ollama", "distributed_inference", "openai", "anthropic"],
        "remote": ["openai", "anthropic", "ollama"],
        "ollama_available": False
    }
    
    # Check if Ollama is available
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            providers["ollama_available"] = True
    except:
        pass
    
    return jsonify(providers)

@app.route('/api-keys', methods=['GET', 'POST'])
def manage_api_keys():
    """Manage API keys for different providers."""
    if request.method == 'GET':
        # Return current API key status (without exposing actual keys)
        return jsonify({
            "openai": bool(app.config.get('OPENAI_API_KEY')),
            "anthropic": bool(app.config.get('ANTHROPIC_API_KEY')),
            "together": bool(app.config.get('TOGETHER_API_KEY'))
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        # Update API keys in app config
        if 'openai' in data and data['openai']:
            app.config['OPENAI_API_KEY'] = data['openai']
        if 'anthropic' in data and data['anthropic']:
            app.config['ANTHROPIC_API_KEY'] = data['anthropic']
        if 'together' in data and data['together']:
            app.config['TOGETHER_API_KEY'] = data['together']
        
        return jsonify({"success": True, "message": "API keys updated successfully"})

@app.route('/context-size', methods=['POST'])
def calculate_context():
    """Calculate required context size for given text."""
    try:
        data = request.get_json()
        text_length = data.get('text_length', 0)
        
        if text_length > 0:
            # Simple context size estimation since context calculation is now handled by the A2A server
            estimated_tokens = int(text_length / 4)
            context_size = max(4096, estimated_tokens + 2048)  # Add buffer for response
            return jsonify({
                "text_length": text_length,
                "estimated_tokens": estimated_tokens,
                "recommended_context": context_size
            })
        else:
            return jsonify({"error": "No text length provided"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send-task', methods=['POST'])
def send_task():
    """Send a task to the A2A server."""
    try:
        data = request.get_json()
        
        # Parse form data
        query_text = data.get('query', '').strip()
        context_text = data.get('context', '').strip()
        file_content = data.get('file_content')
        file_name = data.get('file_name')
        file_type = data.get('file_type')
        json_data = data.get('json_data')
        skill_id = data.get('skill_id', 'minion_query')
        max_rounds = int(data.get('max_rounds', 2))
        streaming = data.get('streaming', False)
        
        # Model provider configuration
        local_provider = data.get('local_provider', 'ollama')
        local_model = data.get('local_model', 'llama3.2')
        remote_provider = data.get('remote_provider', 'openai')
        remote_model = data.get('remote_model', 'gpt-4o-mini')
        
        # Dynamic context will be calculated by the A2A server
        
        # Build message parts
        parts = []
        
        if query_text:
            parts.append({
                "kind": "text",
                "text": query_text
            })
        
        if context_text:
            parts.append({
                "kind": "text", 
                "text": context_text
            })
        
        if file_content and file_name:
            parts.append({
                "kind": "file",
                "file": {
                    "name": file_name,
                    "mimeType": file_type or "text/plain",
                    "bytes": file_content  # Should be base64 encoded
                }
            })
        
        if json_data:
            try:
                parsed_json = json.loads(json_data)
                parts.append({
                    "kind": "data",
                    "data": parsed_json
                })
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON data"}), 400
        
        if not parts:
            return jsonify({"error": "No content provided"}), 400
        
        # Build message
        message = {
            "role": "user",
            "parts": parts
        }
        
        # Build metadata
        metadata = {
            "skill_id": skill_id,
            "max_rounds": max_rounds,
            "local_provider": local_provider,
            "local_model": local_model,
            "remote_provider": remote_provider, 
            "remote_model": remote_model
        }
        
        # Add API keys to metadata if available
        if app.config.get('OPENAI_API_KEY'):
            metadata['openai_api_key'] = app.config['OPENAI_API_KEY']
        if app.config.get('ANTHROPIC_API_KEY'):
            metadata['anthropic_api_key'] = app.config['ANTHROPIC_API_KEY']
        if app.config.get('TOGETHER_API_KEY'):
            metadata['together_api_key'] = app.config['TOGETHER_API_KEY']
        
        # Context sizes will be calculated by the A2A server based on actual processed content
        
        # Add skill-specific metadata
        if skill_id == "minions_query":
            metadata.update({
                "max_jobs_per_round": int(data.get('max_jobs_per_round', 5)),
                "num_tasks_per_round": int(data.get('num_tasks_per_round', 3)),
                "num_samples_per_task": int(data.get('num_samples_per_task', 1))
            })
        
        task_id = str(uuid.uuid4())
        
        if streaming:
            # Handle streaming in background
            def stream_handler():
                def callback(event):
                    socketio.emit('task_update', {
                        'task_id': task_id,
                        'event': event
                    })
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        a2a_client.send_task_streaming(message, metadata, task_id, callback)
                    )
                except Exception as e:
                    socketio.emit('task_error', {
                        'task_id': task_id,
                        'error': str(e)
                    })
                finally:
                    loop.close()
            
            thread = threading.Thread(target=stream_handler)
            thread.daemon = True
            thread.start()
            
            return jsonify({"task_id": task_id, "streaming": True})
        else:
            # Non-streaming
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    a2a_client.send_task(message, metadata, task_id)
                )
                return jsonify({"task_id": task_id, "result": result})
            finally:
                loop.close()
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/task-status/<task_id>')
def task_status(task_id):
    """Get task status."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(a2a_client.get_task(task_id))
        loop.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="A2A-Minions Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to") 
    parser.add_argument("--a2a-url", default="http://localhost:8001", help="A2A server URL")
    parser.add_argument("--api-key", default="abcd", help="API key for A2A server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Update global configuration
    A2A_SERVER_URL = args.a2a_url
    API_KEY = args.api_key
    a2a_client = A2AClient(A2A_SERVER_URL, API_KEY)
    
    print(f"Starting A2A-Minions Web Interface")
    print(f"   Web Server: http://{args.host}:{args.port}")
    print(f"   A2A Server: {args.a2a_url}")
    print(f"   API Key: {args.api_key}")
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug) 