#!/usr/bin/env python3
"""
Claude Code Token Tracker Server
Monitors and serves token usage data from Claude Code JSONL files
"""

import os
import json
import glob
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import threading
import time
from pathlib import Path
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Global storage for token data
token_data = {
    "sessions": [],
    "total_usage": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0
    },
    "by_model": defaultdict(lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "message_count": 0
    }),
    "last_updated": None
}

# Lock for thread-safe access
data_lock = threading.Lock()

def get_claude_projects_directory():
    """Get the Claude Code projects directory"""
    return os.path.expanduser("~/.claude/projects")

def parse_jsonl_file(filepath):
    """Parse a Claude Code JSONL file and extract token usage data"""
    sessions = []
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0
    }
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Only process assistant messages with usage data
                    if entry.get("type") == "assistant" and "message" in entry:
                        message = entry["message"]
                        if "usage" in message:
                            usage = message["usage"]
                            timestamp = entry.get("timestamp", "Unknown")
                            model = message.get("model", "Unknown")
                            
                            # Extract token counts
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                            cache_creation = usage.get("cache_creation_input_tokens", 0)
                            cache_read = usage.get("cache_read_input_tokens", 0)
                            
                            # Create session info
                            session_info = {
                                "timestamp": timestamp,
                                "model": model,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                                "cache_creation_tokens": cache_creation,
                                "cache_read_tokens": cache_read,
                                "request_id": entry.get("requestId", ""),
                                "uuid": entry.get("uuid", ""),
                                "content_preview": ""
                            }
                            
                            # Get a preview of the content
                            if message.get("content") and len(message["content"]) > 0:
                                first_content = message["content"][0]
                                if first_content.get("type") == "text":
                                    preview = first_content.get("text", "")[:100]
                                    session_info["content_preview"] = preview + "..." if len(preview) == 100 else preview
                            
                            sessions.append(session_info)
                            
                            # Accumulate totals
                            total_usage["input_tokens"] += input_tokens
                            total_usage["output_tokens"] += output_tokens
                            total_usage["cache_creation_tokens"] += cache_creation
                            total_usage["cache_read_tokens"] += cache_read
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
        
        return sessions, total_usage
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return [], total_usage

def scan_claude_logs():
    """Scan all Claude Code JSONL files and update token data"""
    projects_dir = get_claude_projects_directory()
    
    if not os.path.exists(projects_dir):
        print(f"Claude Code projects directory not found: {projects_dir}")
        return
    
    all_sessions = []
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0
    }
    by_model = defaultdict(lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "message_count": 0
    })
    
    # Scan all project directories
    for project_dir in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project_dir)
        if os.path.isdir(project_path):
            # Find all JSONL files in this project
            jsonl_files = glob.glob(os.path.join(project_path, "*.jsonl"))
            
            for jsonl_file in jsonl_files:
                sessions, file_usage = parse_jsonl_file(jsonl_file)
                
                # Add project name to sessions
                for session in sessions:
                    session["project"] = project_dir.replace("-", "/")
                
                all_sessions.extend(sessions)
                
                # Accumulate totals
                total_usage["input_tokens"] += file_usage["input_tokens"]
                total_usage["output_tokens"] += file_usage["output_tokens"]
                total_usage["cache_creation_tokens"] += file_usage["cache_creation_tokens"]
                total_usage["cache_read_tokens"] += file_usage["cache_read_tokens"]
                
                # Track by model
                for session in sessions:
                    model = session["model"]
                    by_model[model]["input_tokens"] += session["input_tokens"]
                    by_model[model]["output_tokens"] += session["output_tokens"]
                    by_model[model]["total_tokens"] += session["total_tokens"]
                    by_model[model]["message_count"] += 1
    
    # Calculate total tokens
    total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
    
    # Sort sessions by timestamp (most recent first)
    all_sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    
    with data_lock:
        token_data["sessions"] = all_sessions[:100]  # Keep last 100 messages
        token_data["total_usage"] = total_usage
        token_data["by_model"] = dict(by_model)
        token_data["last_updated"] = datetime.now().isoformat()

def background_scanner():
    """Background thread to periodically scan for new logs"""
    while True:
        scan_claude_logs()
        time.sleep(5)  # Scan every 5 seconds

@app.route('/api/usage')
def get_usage():
    """API endpoint to get current token usage data"""
    with data_lock:
        return jsonify(token_data)

@app.route('/api/usage/latest')
def get_latest():
    """Get the most recent session's usage"""
    with data_lock:
        if token_data["sessions"]:
            return jsonify(token_data["sessions"][0])
        else:
            return jsonify({"error": "No sessions found"})

@app.route('/')
def dashboard():
    """Serve the Claude Code dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'claude_code_dashboard.html')
    # If claude-specific dashboard doesn't exist, use the regular one
    if not os.path.exists(dashboard_path):
        dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    
    try:
        with open(dashboard_path, 'r') as f:
            return f.read()
    except:
        return "Dashboard file not found", 404

if __name__ == "__main__":
    # Start background scanner thread
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    
    # Initial scan
    scan_claude_logs()
    
    print("Claude Code Token Tracker Server starting on http://localhost:5555")
    print(f"Monitoring Claude Code projects directory: {get_claude_projects_directory()}")
    app.run(host='0.0.0.0', port=5555, debug=False)