#!/usr/bin/env python3
"""
Token Tracker Server for Minions
Monitors and serves token usage data from minions log files
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

app = Flask(__name__)
CORS(app)

# Global storage for token data
token_data = {
    "sessions": [],
    "total_usage": {
        "remote": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "local": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    },
    "last_updated": None
}

# Lock for thread-safe access
data_lock = threading.Lock()

def get_logs_directory():
    """Find the logs directory relative to the minions root"""
    # Navigate up to minions root directory
    current_dir = Path(__file__).resolve().parent
    minions_root = current_dir.parent.parent  # Go up from apps/token-tracker to minions/
    logs_dir = minions_root / "logs"
    return str(logs_dir)

def parse_log_file(filepath):
    """Parse a minions log file and extract token usage data"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract relevant information
        session_info = {
            "timestamp": data.get("timestamp", "Unknown"),
            "task": data.get("task", "Unknown task"),
            "protocol": data.get("protocol", "unknown"),
            "model": {
                "local": data.get("local_model", "Unknown"),
                "remote": data.get("remote_model", "Unknown")
            },
            "usage": data.get("usage", {
                "remote": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "local": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }),
            "rounds": data.get("rounds", 0),
            "filename": os.path.basename(filepath)
        }
        
        return session_info
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def scan_logs():
    """Scan all log files and update token data"""
    logs_dir = get_logs_directory()
    
    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return
    
    log_files = glob.glob(os.path.join(logs_dir, "*.json"))
    sessions = []
    total_remote = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_local = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for log_file in sorted(log_files, reverse=True):  # Most recent first
        session = parse_log_file(log_file)
        if session:
            sessions.append(session)
            
            # Accumulate totals
            if "remote" in session["usage"]:
                remote = session["usage"]["remote"]
                total_remote["prompt_tokens"] += remote.get("prompt_tokens", 0)
                total_remote["completion_tokens"] += remote.get("completion_tokens", 0)
                total_remote["total_tokens"] += remote.get("total_tokens", 0)
            
            if "local" in session["usage"]:
                local = session["usage"]["local"]
                total_local["prompt_tokens"] += local.get("prompt_tokens", 0)
                total_local["completion_tokens"] += local.get("completion_tokens", 0)
                total_local["total_tokens"] += local.get("total_tokens", 0)
    
    with data_lock:
        token_data["sessions"] = sessions[:50]  # Keep last 50 sessions
        token_data["total_usage"]["remote"] = total_remote
        token_data["total_usage"]["local"] = total_local
        token_data["last_updated"] = datetime.now().isoformat()

def background_scanner():
    """Background thread to periodically scan for new logs"""
    while True:
        scan_logs()
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
    """Serve the main dashboard"""
    with open(os.path.join(os.path.dirname(__file__), 'dashboard.html'), 'r') as f:
        return f.read()

if __name__ == "__main__":
    # Start background scanner thread
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    
    # Initial scan
    scan_logs()
    
    print("Token Tracker Server starting on http://localhost:5555")
    print(f"Monitoring logs directory: {get_logs_directory()}")
    app.run(host='0.0.0.0', port=5555, debug=False)