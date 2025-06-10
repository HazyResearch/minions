#!/usr/bin/env python3
"""
Startup script for A2A-Minions server.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if required environment variables are set."""
    
    logger.info("🔍 Checking environment...")
    
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
        logger.warning("⚠️  Missing environment variables:")
        for var in missing_vars:
            logger.warning(var)
        logger.info("Please set these environment variables before running the server.")
        logger.info("Example:")
        logger.info("  export OPENAI_API_KEY=your_api_key_here")
        return False
    
    logger.info("✅ Environment looks good!")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    
    logger.info("📦 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import httpx
        import pydantic
        import jwt
        logger.info("✅ Core dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    # Check if Minions is available
    try:
        from minions.minion import Minion
        from minions.clients.ollama import OllamaClient
        from minions.clients.openai import OpenAIClient
        logger.info("✅ Minions protocol available")
    except ImportError as e:
        logger.error(f"❌ Minions not available: {e}")
        logger.error("Make sure you're running from the Minions project directory")
        logger.error("and have installed minions with: pip install -e .")
        return False
    
    return True


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="A2A-Minions Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--base-url", help="Base URL for agent card")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    
    # Authentication options
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication (for testing only)")
    parser.add_argument("--api-key", help="Set a specific API key instead of generating one")
    parser.add_argument("--jwt-secret", help="JWT secret for token generation")
    parser.add_argument("--api-keys-file", default="api_keys.json", help="Path to API keys file")
    
    args = parser.parse_args()
    
    logger.info("🚀 A2A-Minions Server Startup")
    logger.info("=" * 50)
    
    # Run checks unless skipped
    if not args.skip_checks:
        if not check_environment():
            sys.exit(1)
        
        if not check_dependencies():
            sys.exit(1)
    
    # Import and start server
    try:
        from a2a_minions.server import A2AMinionsServer
        from a2a_minions.auth import AuthConfig
        
        # Configure authentication
        auth_config = AuthConfig(
            require_auth=not args.no_auth,
            api_keys_file=args.api_keys_file,
            jwt_secret=args.jwt_secret or os.getenv("A2A_JWT_SECRET")
        )
        
        # Create server instance
        server = A2AMinionsServer(
            host=args.host,
            port=args.port,
            base_url=args.base_url,
            auth_config=auth_config
        )
        
        # If API key was provided, add it
        if args.api_key and not args.no_auth:
            from a2a_minions.auth import get_auth_manager
            auth_manager = get_auth_manager()
            auth_manager.api_key_manager.api_keys[args.api_key] = {
                "name": "cli_provided",
                "created_at": datetime.now().isoformat(),
                "scopes": ["minion:query", "minions:query", "tasks:read", "tasks:write"],
                "active": True
            }
            auth_manager.api_key_manager._save_api_keys()
            logger.info(f"Added API key: {args.api_key}")
        
        logger.info(f"🌟 Starting A2A-Minions server...")
        logger.info(f"   Host: {args.host}")
        logger.info(f"   Port: {args.port}")
        logger.info(f"   URL: http://{args.host}:{args.port}")
        logger.info(f"   Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
        logger.info(f"   Auth: {'ENABLED' if not args.no_auth else 'DISABLED'}")
        
        if not args.no_auth:
            logger.info("")
            logger.info("🔐 Authentication is enabled. Use one of:")
            logger.info("   - API Key header: X-API-Key: <your-key>")
            logger.info("   - Bearer token: Authorization: Bearer <token>")
            logger.info("   - OAuth2: POST /oauth/token (client credentials flow)")
            
            # Show default API key if it was generated
            if auth_config.require_auth and not args.api_key:
                # Check if a default key was generated
                if hasattr(server, 'auth_manager') and server.auth_manager.default_api_key:
                    logger.info("")
                    logger.info("🔑 Generated default API key:")
                    logger.info(f"   {server.auth_manager.default_api_key}")
                    logger.info("   ⚠️  Save this key - it won't be shown again!")
        
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("-" * 50)
        
        server.run()
        
    except KeyboardInterrupt:
        logger.info("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"\n💥 Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 