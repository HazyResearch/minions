"""
Authentication and authorization for A2A-Minions server.
Implements A2A-compatible security schemes.
"""

import os
import json
import secrets
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import jwt
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TokenData(BaseModel):
    """Token payload data."""
    sub: str  # Subject (client_id or user_id)
    scopes: List[str] = []
    exp: Optional[int] = None
    iat: Optional[int] = None
    
    
class AuthConfig(BaseModel):
    """Authentication configuration."""
    # API Key settings
    api_keys_file: str = "api_keys.json"
    api_key_header_name: str = "X-API-Key"
    
    # JWT settings
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    
    # OAuth2 settings
    oauth2_enabled: bool = False
    oauth2_token_url: str = "/oauth/token"
    
    # Security settings
    require_auth: bool = True
    allowed_scopes: Dict[str, List[str]] = {
        "minion_query": ["minion:query"],
        "minions_query": ["minions:query"],
        "tasks/send": ["tasks:write"],
        "tasks/sendSubscribe": ["tasks:write"],
        "tasks/get": ["tasks:read"],
        "tasks/cancel": ["tasks:write"]
    }


class APIKeyManager:
    """Manages API keys for local deployments."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.api_keys_file = Path(config.api_keys_file)
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from file."""
        if self.api_keys_file.exists():
            try:
                with open(self.api_keys_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
        return {}
    
    def _save_api_keys(self):
        """Save API keys to file."""
        try:
            with open(self.api_keys_file, 'w') as f:
                json.dump(self.api_keys, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def generate_api_key(self, name: str, scopes: List[str] = None) -> str:
        """Generate a new API key."""
        api_key = f"a2a_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "scopes": scopes or ["minion:query", "minions:query", "tasks:read", "tasks:write"],
            "active": True
        }
        
        self._save_api_keys()
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        key_data = self.api_keys.get(api_key)
        if key_data and key_data.get("active", True):
            return key_data
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            self.api_keys[api_key]["revoked_at"] = datetime.now().isoformat()
            self._save_api_keys()
            return True
        return False


class JWTManager:
    """Manages JWT tokens for bearer authentication."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.secret = config.jwt_secret or os.getenv("A2A_JWT_SECRET", secrets.token_urlsafe(32))
        self.algorithm = config.jwt_algorithm
        self.expiry_hours = config.jwt_expiry_hours
    
    def create_token(self, subject: str, scopes: List[str] = None) -> str:
        """Create a JWT token."""
        now = datetime.utcnow()
        payload = {
            "sub": subject,
            "scopes": scopes or [],
            "iat": now,
            "exp": now + timedelta(hours=self.expiry_hours)
        }
        
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return TokenData(**payload)
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


class AuthenticationManager:
    """Main authentication manager supporting multiple schemes."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.api_key_manager = APIKeyManager(self.config)
        self.jwt_manager = JWTManager(self.config)
        self.default_api_key = None  # Store the default key
        
        # Security dependencies
        self.api_key_header = APIKeyHeader(
            name=self.config.api_key_header_name,
            auto_error=False
        )
        self.bearer_scheme = HTTPBearer(auto_error=False)
        
        # Initialize with a default API key if none exist
        if not self.api_key_manager.api_keys and self.config.require_auth:
            self.default_api_key = self.api_key_manager.generate_api_key(
                "default",
                ["minion:query", "minions:query", "tasks:read", "tasks:write"]
            )
            logger.info(f"Generated default API key: {self.default_api_key}")
            logger.info(f"Save this key - it won't be shown again!")
    
    async def authenticate(
        self,
        request: Request,
        api_key: Optional[str] = Security(lambda: None),
        bearer_token: Optional[str] = Security(lambda: None)
    ) -> Optional[TokenData]:
        """Authenticate request using available methods."""
        
        # Check if authentication is required
        if not self.config.require_auth:
            return TokenData(sub="anonymous", scopes=["*"])
        
        # Try API key authentication
        api_key = request.headers.get(self.config.api_key_header_name)
        if api_key:
            key_data = self.api_key_manager.validate_api_key(api_key)
            if key_data:
                return TokenData(
                    sub=key_data.get("name", "api_key_user"),
                    scopes=key_data.get("scopes", [])
                )
        
        # Try Bearer token authentication
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            token_data = self.jwt_manager.verify_token(token)
            if token_data:
                return token_data
        
        # No valid authentication found
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer, ApiKey"}
        )
    
    def check_scopes(self, token_data: TokenData, required_scopes: List[str]) -> bool:
        """Check if token has required scopes."""
        if "*" in token_data.scopes:  # Admin scope
            return True
        
        return any(scope in token_data.scopes for scope in required_scopes)
    
    def require_scopes(self, *scopes: str):
        """Dependency to require specific scopes."""
        async def _check_scopes(
            request: Request,
            token_data: TokenData = Depends(self.authenticate)
        ):
            # Get method-specific scopes if configured
            method = request.scope.get("path", "").lstrip("/")
            required_scopes = self.config.allowed_scopes.get(method, list(scopes))
            
            if not self.check_scopes(token_data, required_scopes):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required scopes: {required_scopes}"
                )
            
            return token_data
        
        return _check_scopes
    
    async def handle_oauth_token(self, grant_type: str, client_id: str, 
                                client_secret: str, scope: str = "") -> Dict[str, Any]:
        """Handle OAuth2 token requests (client credentials flow)."""
        
        if grant_type != "client_credentials":
            raise HTTPException(
                status_code=400,
                detail="Unsupported grant type. Only 'client_credentials' is supported."
            )
        
        # In a real implementation, validate client_id and client_secret
        # For now, we'll create a token with requested scopes
        requested_scopes = scope.split() if scope else []
        
        # Validate scopes
        valid_scopes = ["minion:query", "minions:query", "tasks:read", "tasks:write"]
        scopes = [s for s in requested_scopes if s in valid_scopes]
        
        if not scopes:
            scopes = valid_scopes  # Grant all scopes if none specified
        
        # Create access token
        access_token = self.jwt_manager.create_token(client_id, scopes)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.config.jwt_expiry_hours * 3600,
            "scope": " ".join(scopes)
        }


# Global auth manager instance
auth_manager = None


def init_auth(config: Optional[AuthConfig] = None) -> AuthenticationManager:
    """Initialize the global auth manager."""
    global auth_manager
    auth_manager = AuthenticationManager(config)
    return auth_manager


def get_auth_manager() -> AuthenticationManager:
    """Get the global auth manager."""
    if auth_manager is None:
        init_auth()
    return auth_manager