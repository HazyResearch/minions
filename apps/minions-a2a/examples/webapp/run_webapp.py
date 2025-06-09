#!/usr/bin/env python3
"""
Launcher script for A2A-Minions webapp.
Starts both the A2A server and the web interface.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path

class ServerManager:
    """Manages A2A server, web server, and Ollama processes."""
    
    def __init__(self):
        self.a2a_process = None
        self.web_process = None
        self.ollama_process = None
        self.running = True
        self.ollama_enabled = False
        
    def start_a2a_server(self, port=8001, api_key="abcd"):
        """Start the A2A server."""
        print(f"Starting A2A server on port {port}...")
        
        # Path to the A2A server script
        a2a_script = Path(__file__).parent.parent.parent / "run_server.py"
        
        if not a2a_script.exists():
            print(f"❌ A2A server script not found at {a2a_script}")
            return False
        
        # Build command
        cmd = [
            sys.executable, str(a2a_script),
            "--host", "0.0.0.0",
            "--port", str(port),
            "--api-key", api_key,
            "--skip-checks"  # Skip environment checks for demo
        ]
        
        # Set environment variables to handle Unicode encoding on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        try:
            self.a2a_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            # Monitor A2A server output
            def monitor_a2a():
                while self.running and self.a2a_process:
                    line = self.a2a_process.stdout.readline()
                    if line:
                        print(f"[A2A] {line.strip()}")
                    elif self.a2a_process.poll() is not None:
                        break
            
            thread = threading.Thread(target=monitor_a2a, daemon=True)
            thread.start()
            
            # Wait a moment for server to start
            time.sleep(3)
            
            if self.a2a_process.poll() is None:
                print(f"A2A server started successfully")
                return True
            else:
                print(f"A2A server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start A2A server: {e}")
            return False
    
    def start_ollama_server(self):
        """Start the Ollama server if not already running."""
        print("Checking Ollama server status...")
        
        # First check if Ollama is already running
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is already running")
                self.ollama_enabled = True
                return True
        except:
            pass  # Ollama not running, we'll start it
        
        print("Starting Ollama server...")
        
        # Check if ollama command is available
        try:
            result = subprocess.run(["ollama", "--version"], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("❌ Ollama command not found. Please install Ollama first.")
                print("   Visit: https://ollama.ai/download")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Ollama command not found. Please install Ollama first.")
            print("   Visit: https://ollama.ai/download")
            return False
        
        try:
            # Start Ollama server
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            # Monitor Ollama server output
            def monitor_ollama():
                while self.running and self.ollama_process:
                    line = self.ollama_process.stdout.readline()
                    if line:
                        print(f"[OLLAMA] {line.strip()}")
                    elif self.ollama_process.poll() is not None:
                        break
            
            thread = threading.Thread(target=monitor_ollama, daemon=True)
            thread.start()
            
            # Wait for Ollama to start and verify it's working
            max_retries = 10
            for i in range(max_retries):
                time.sleep(2)
                try:
                    import httpx
                    response = httpx.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        print("✅ Ollama server started successfully")
                        self.ollama_enabled = True
                        return True
                except:
                    if i == max_retries - 1:
                        print("❌ Ollama server failed to respond after startup")
                        return False
                    continue
            
            return False
                
        except Exception as e:
            print(f"❌ Failed to start Ollama server: {e}")
            return False
    
    def start_web_server(self, host="127.0.0.1", port=5000, a2a_url="http://localhost:8001", api_key="abcd"):
        """Start the web server."""
        print(f"🌐 Starting web server on {host}:{port}...")
        
        # Path to the web app script
        web_script = Path(__file__).parent / "app.py"
        
        if not web_script.exists():
            print(f"❌ Web app script not found at {web_script}")
            return False
        
        # Build command
        cmd = [
            sys.executable, str(web_script),
            "--host", host,
            "--port", str(port),
            "--a2a-url", a2a_url,
            "--api-key", api_key
        ]
        
        try:
            # Set environment for UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            self.web_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            # Monitor web server output
            def monitor_web():
                while self.running and self.web_process:
                    line = self.web_process.stdout.readline()
                    if line:
                        print(f"[WEB] {line.strip()}")
                    elif self.web_process.poll() is not None:
                        break
            
            thread = threading.Thread(target=monitor_web, daemon=True)
            thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
            if self.web_process.poll() is None:
                print(f"✅ Web server started successfully")
                return True
            else:
                print(f"❌ Web server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start web server: {e}")
            return False
    
    def check_health(self, a2a_url="http://localhost:8001", web_url="http://localhost:5000"):
        """Check if all servers are healthy."""
        import httpx
        
        print("🔍 Checking server health...")
        
        # Check A2A server
        try:
            response = httpx.get(f"{a2a_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ A2A server is healthy")
            else:
                print(f"⚠️  A2A server returned status {response.status_code}")
        except Exception as e:
            print(f"❌ A2A server health check failed: {e}")
        
        # Check web server
        try:
            response = httpx.get(f"{web_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Web server is healthy")
            else:
                print(f"⚠️  Web server returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Web server health check failed: {e}")
        
        # Check Ollama server if enabled
        if self.ollama_enabled:
            try:
                response = httpx.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama server is healthy")
                else:
                    print(f"⚠️  Ollama server returned status {response.status_code}")
            except Exception as e:
                print(f"❌ Ollama server health check failed: {e}")
    
    def stop_servers(self):
        """Stop all servers."""
        print("\n🛑 Stopping servers...")
        self.running = False
        
        if self.web_process:
            print("   Stopping web server...")
            self.web_process.terminate()
            try:
                self.web_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.web_process.kill()
            print("   ✅ Web server stopped")
        
        if self.a2a_process:
            print("   Stopping A2A server...")
            self.a2a_process.terminate()
            try:
                self.a2a_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.a2a_process.kill()
            print("   ✅ A2A server stopped")
        
        if self.ollama_process:
            print("   Stopping Ollama server...")
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
            print("   ✅ Ollama server stopped")
    
    def run(self, a2a_port=8001, web_host="127.0.0.1", web_port=5000, api_key="abcd", auto_ollama=True):
        """Run all servers."""
        
        def signal_handler(signum, frame):
            print(f"\n📶 Received signal {signum}")
            self.stop_servers()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start Ollama server if requested
            if auto_ollama:
                print("🦙 Auto-starting Ollama server...")
                if not self.start_ollama_server():
                    print("⚠️  Ollama server failed to start, but continuing...")
                    print("   You can still use other model providers (OpenAI, Anthropic)")
            
            # Start A2A server
            if not self.start_a2a_server(port=a2a_port, api_key=api_key):
                print("❌ Failed to start A2A server")
                return False
            
            # Start web server
            a2a_url = f"http://localhost:{a2a_port}"
            if not self.start_web_server(
                host=web_host,
                port=web_port,
                a2a_url=a2a_url,
                api_key=api_key
            ):
                print("❌ Failed to start web server")
                self.stop_servers()
                return False
            
            # Wait a moment then check health
            time.sleep(3)
            self.check_health(
                a2a_url=a2a_url,
                web_url=f"http://{web_host}:{web_port}"
            )
            
            print("\n" + "="*60)
            print("🎉 A2A-Minions webapp is running!")
            print("="*60)
            print(f"📊 A2A Server:    http://localhost:{a2a_port}")
            print(f"🌐 Web Interface: http://{web_host}:{web_port}")
            print(f"🔑 API Key:       {api_key}")
            if self.ollama_enabled:
                print(f"🦙 Ollama Server: http://localhost:11434")
            print("="*60)
            print("\n💡 Usage:")
            print("   1. Open the web interface in your browser")
            print("   2. Select a processing type (Focused or Parallel)")
            print("   3. Choose your model providers (Local/Remote)")
            print("   4. Enter your question and any context")
            print("   5. Click 'Send Query' to get results")
            print("\n📖 Features:")
            print("   • Configurable model providers (Ollama, OpenAI, Anthropic)")
            print("   • Automatic Ollama server management")
            print("   • Real-time streaming responses")
            print("   • File upload support (PDF, TXT, JSON, etc.)")
            print("   • JSON data processing")
            print("   • Task status monitoring")
            print("\n⌨️  Press Ctrl+C to stop all servers")
            print("-"*60)
            
            # Keep running until interrupted
            try:
                while self.running:
                    if self.a2a_process and self.a2a_process.poll() is not None:
                        print("❌ A2A server process died")
                        break
                    if self.web_process and self.web_process.poll() is not None:
                        print("❌ Web server process died")
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ Error running servers: {e}")
            return False
        finally:
            self.stop_servers()


def check_dependencies():
    """Check if required dependencies are installed."""
    
    print("📦 Checking dependencies...")
    
    # Check if we're in the right directory
    if not Path("run_server.py").parent.parent.parent.exists():
        print("❌ Please run this script from the apps/minions-a2a/examples/webapp directory")
        return False
    
    # Check Python modules
    required_modules = ['flask', 'httpx', 'flask_socketio']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing Python modules: {', '.join(missing_modules)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available")
    return True


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="A2A-Minions Webapp Launcher")
    parser.add_argument("--a2a-port", type=int, default=8001, help="A2A server port")
    parser.add_argument("--web-host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--web-port", type=int, default=5000, help="Web server port")
    parser.add_argument("--api-key", default="abcd", help="API key for authentication")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency checks")
    parser.add_argument("--no-ollama", action="store_true", help="Don't auto-start Ollama server")
    
    args = parser.parse_args()
    
    print("🚀 A2A-Minions Webapp Launcher")
    print("=" * 50)
    
    # Check dependencies unless skipped
    if not args.skip_deps:
        if not check_dependencies():
            sys.exit(1)
    
    # Create and run server manager
    manager = ServerManager()
    success = manager.run(
        a2a_port=args.a2a_port,
        web_host=args.web_host,
        web_port=args.web_port,
        api_key=args.api_key,
        auto_ollama=not args.no_ollama
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 