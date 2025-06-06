"""
Authentication and authorization for A2A-Minions server.
Implements A2A-compatible security schemes.
"""

import os
import json
import secrets
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import jwt
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenData(BaseModel):
    """Token payload data."""
    sub: str  # Subject (client_id or user_id)
    scopes: List[str] = []
    exp: Optional[int] = None
    iat: Optional[int] = None
    
    
class AuthConfig(BaseModel):
    """Authentication configuration."""
    require_auth: bool = True
    api_key_header_name: str = "X-API-Key"
    api_keys_path: Path = Path("api_keys.json")
    oauth2_clients_path: Path = Path("oauth2_clients.json")
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expire_seconds: int = 3600
    allowed_scopes: Dict[str, List[str]] = {
        "tasks/send": ["tasks:write"],
        "tasks/sendSubscribe": ["tasks:write"],
        "tasks/get": ["tasks:read"],
        "tasks/cancel": ["tasks:write"],
        "agent/authenticatedExtendedCard": ["tasks:read"]
    }


class APIKeyManager:
    """Manages API keys with persistence."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.storage_path = config.api_keys_path
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
        return {}
    
    def _save_api_keys(self):
        """Save API keys to file."""
        try:
            with open(self.storage_path, 'w') as f:
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
    """Manages JWT token creation and validation."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        
        # Use provided secret or generate one
        if config.jwt_secret:
            self.secret = config.jwt_secret
        else:
            self.secret = secrets.token_urlsafe(32)
            logger.warning("No JWT secret provided. Generated random secret (not suitable for production)")
        self.algorithm = config.jwt_algorithm
    
    def create_token(self, token_data: TokenData) -> str:
        """Create a JWT token from token data."""
        payload = {
            "sub": token_data.sub,
            "exp": token_data.exp,
            "scopes": token_data.scopes
        }
        
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify a JWT token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            
            return TokenData(
                sub=payload.get("sub"),
                exp=payload.get("exp"),
                scopes=payload.get("scopes", [])
            )
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


class OAuth2Client(BaseModel):
    """OAuth2 client registration."""
    client_id: str
    client_secret: str
    name: str
    scopes: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    active: bool = True


class OAuth2ClientManager:
    """Manages OAuth2 client registrations."""
    
    def __init__(self, storage_path: Path = Path("oauth2_clients.json")):
        self.storage_path = storage_path
        self.clients: Dict[str, OAuth2Client] = self._load_clients()
    
    def _load_clients(self) -> Dict[str, OAuth2Client]:
        """Load OAuth2 clients from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    return {
                        client_id: OAuth2Client(**client_data)
                        for client_id, client_data in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load OAuth2 clients: {e}")
        return {}
    
    def _save_clients(self):
        """Save OAuth2 clients to storage."""
        try:
            with open(self.storage_path, "w") as f:
                data = {
                    client_id: client.dict()
                    for client_id, client in self.clients.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save OAuth2 clients: {e}")
    
    def register_client(self, name: str, scopes: List[str]) -> Tuple[str, str]:
        """Register a new OAuth2 client."""
        client_id = f"oauth2_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)
        
        client = OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            name=name,
            scopes=scopes
        )
        
        self.clients[client_id] = client
        self._save_clients()
        
        logger.info(f"Registered OAuth2 client: {name} (ID: {client_id})")
        return client_id, client_secret
    
    def validate_client(self, client_id: str, client_secret: str) -> Optional[OAuth2Client]:
        """Validate OAuth2 client credentials."""
        client = self.clients.get(client_id)
        if client and client.active and client.client_secret == client_secret:
            return client
        return None
    
    def revoke_client(self, client_id: str) -> bool:
        """Revoke an OAuth2 client."""
        if client_id in self.clients:
            self.clients[client_id].active = False
            self._save_clients()
            logger.info(f"Revoked OAuth2 client: {client_id}")
            return True
        return False
    
    def list_clients(self) -> List[Dict[str, Any]]:
        """List all OAuth2 clients (without secrets)."""
        return [
            {
                "client_id": client.client_id,
                "name": client.name,
                "scopes": client.scopes,
                "created_at": client.created_at,
                "active": client.active
            }
            for client in self.clients.values()
        ]


class AuthenticationManager:
    """Main authentication manager supporting multiple schemes."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.api_key_manager = APIKeyManager(self.config)
        self.jwt_manager = JWTManager(self.config)
        self.oauth2_manager = OAuth2ClientManager(self.config.oauth2_clients_path)
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
        
        # Initialize with a default OAuth2 client if none exist
        if not self.oauth2_manager.clients and self.config.require_auth:
            client_id, client_secret = self.oauth2_manager.register_client(
                "default_oauth_client",
                ["minion:query", "minions:query", "tasks:read", "tasks:write"]
            )
            logger.info(f"Generated default OAuth2 client:")
            logger.info(f"  Client ID: {client_id}")
            logger.info(f"  Client Secret: {client_secret}")
            logger.info(f"Save these credentials - they won't be shown again!")
    
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
        """Handle OAuth2 token requests."""
        if grant_type != "client_credentials":
            raise HTTPException(
                status_code=400,
                detail="Unsupported grant type. Only 'client_credentials' is supported."
            )
        
        # Validate client credentials
        client = self.oauth2_manager.validate_client(client_id, client_secret)
        if not client:
            raise HTTPException(
                status_code=401,
                detail="Invalid client credentials"
            )
        
        # Parse requested scopes
        requested_scopes = scope.split() if scope else []
        
        # Check if client has access to requested scopes
        granted_scopes = []
        for requested_scope in requested_scopes:
            if requested_scope in client.scopes:
                granted_scopes.append(requested_scope)
        
        # If no scopes requested, grant all client scopes
        if not requested_scopes:
            granted_scopes = client.scopes
        
        # Generate access token
        token_data = TokenData(
            sub=client_id,
            exp=datetime.utcnow() + timedelta(seconds=self.config.jwt_expire_seconds),
            scopes=granted_scopes
        )
        
        access_token = self.jwt_manager.create_token(token_data)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.config.jwt_expire_seconds,
            "scope": " ".join(granted_scopes)
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