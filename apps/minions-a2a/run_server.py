#!/usr/bin/env python3
"""
Startup script for A2A-Minions server.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_environment():
    """Check if required environment variables are set."""
    
    print("🔍 Checking environment...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API access",
        "OLLAMA_HOST": "Ollama server (optional - defaults to localhost:11434)"
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        if var == "OLLAMA_HOST":
            # OLLAMA_HOST is optional
            continue
            
        if not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")
    
    if missing_vars:
        print("⚠️  Missing environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease set these environment variables before running the server.")
        print("Example:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print("✅ Environment looks good!")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    
    print("📦 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import httpx
        import pydantic
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check if Minions is available
    try:
        from minions.minion import Minion
        from minions.clients.ollama import OllamaClient
        from minions.clients.openai import OpenAIClient
        print("✅ Minions protocol available")
    except ImportError as e:
        print(f"❌ Minions not available: {e}")
        print("Make sure you're running from the Minions project directory")
        return False
    
    return True


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="A2A-Minions Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--base-url", help="Base URL for agent card")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    
    args = parser.parse_args()
    
    print("🚀 A2A-Minions Server Startup")
    print("=" * 50)
    
    # Run checks unless skipped
    if not args.skip_checks:
        if not check_environment():
            sys.exit(1)
        
        if not check_dependencies():
            sys.exit(1)
    
    # Import and start server
    try:
        from a2a_minions.server import A2AMinionsServer
        
        server = A2AMinionsServer(
            host=args.host,
            port=args.port,
            base_url=args.base_url
        )
        
        print(f"🌟 Starting A2A-Minions server...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   URL: http://{args.host}:{args.port}")
        print(f"   Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
        print()
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        server.run()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n💥 Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 