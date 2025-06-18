#!/usr/bin/env python3
"""
Simple Token Tracker Server for Minions
Uses Python's built-in HTTP server - no Flask required
"""

import os
import json
import glob
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

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
    current_dir = Path(__file__).resolve().parent
    minions_root = current_dir.parent.parent
    logs_dir = minions_root / "logs"
    return str(logs_dir)

def parse_log_file(filepath):
    """Parse a minions log file and extract token usage data"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
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
    
    for log_file in sorted(log_files, reverse=True):
        session = parse_log_file(log_file)
        if session:
            sessions.append(session)
            
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
        token_data["sessions"] = sessions[:50]
        token_data["total_usage"]["remote"] = total_remote
        token_data["total_usage"]["local"] = total_local
        token_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

def background_scanner():
    """Background thread to periodically scan for new logs"""
    while True:
        scan_logs()
        time.sleep(5)

class TokenTrackerHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the token tracker"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            # Serve the dashboard
            self.serve_dashboard()
        elif parsed_path.path == '/api/usage':
            # Serve the API data
            self.serve_api_data()
        else:
            # Try to serve static files
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the dashboard HTML"""
        dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
        try:
            with open(dashboard_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except:
            self.send_error(404, "Dashboard not found")
    
    def serve_api_data(self):
        """Serve the token usage data as JSON"""
        with data_lock:
            data = json.dumps(token_data)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data.encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def run_server(port=5555):
    """Run the HTTP server"""
    handler = TokenTrackerHandler
    
    # Start background scanner thread
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    
    # Initial scan
    scan_logs()
    
    print(f"Token Tracker Server starting on http://localhost:{port}")
    print(f"Monitoring logs directory: {get_logs_directory()}")
    print("\nPress Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()