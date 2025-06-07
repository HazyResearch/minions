#!/usr/bin/env python3
"""
Unit tests for A2A-Minions authentication and authorization.
Tests API key management, JWT tokens, OAuth2 flows, and security enforcement.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from a2a_minions.auth import (
    TokenData, AuthConfig, APIKeyManager, JWTManager, OAuth2Client,
    OAuth2ClientManager, AuthenticationManager, init_auth, get_auth_manager
)


class TestTokenData(unittest.TestCase):
    """Test TokenData model."""
    
    def test_minimal_token_data(self):
        """Test minimal token data."""
        token = TokenData(sub="user123")
        self.assertEqual(token.sub, "user123")
        self.assertEqual(token.scopes, [])
        self.assertIsNone(token.exp)
        self.assertIsNone(token.iat)
    
    def test_full_token_data(self):
        """Test full token data."""
        exp_time = int((datetime.utcnow() + timedelta(hours=1)).timestamp())
        iat_time = int(datetime.utcnow().timestamp())
        
        token = TokenData(
            sub="client_abc",
            scopes=["tasks:read", "tasks:write"],
            exp=exp_time,
            iat=iat_time
        )
        self.assertEqual(token.sub, "client_abc")
        self.assertEqual(len(token.scopes), 2)
        self.assertEqual(token.exp, exp_time)
        self.assertEqual(token.iat, iat_time)


class TestAuthConfig(unittest.TestCase):
    """Test AuthConfig model."""
    
    def test_default_config(self):
        """Test default authentication configuration."""
        config = AuthConfig()
        self.assertTrue(config.require_auth)
        self.assertEqual(config.api_key_header_name, "X-API-Key")
        self.assertEqual(config.jwt_algorithm, "HS256")
        self.assertEqual(config.jwt_expire_seconds, 3600)
        self.assertIsNotNone(config.allowed_scopes)
    
    def test_custom_config(self):
        """Test custom authentication configuration."""
        config = AuthConfig(
            require_auth=False,
            api_key_header_name="X-Custom-Key",
            jwt_secret="test-secret",
            jwt_expire_seconds=7200,
            allowed_scopes={
                "custom/method": ["custom:scope"]
            }
        )
        self.assertFalse(config.require_auth)
        self.assertEqual(config.api_key_header_name, "X-Custom-Key")
        self.assertEqual(config.jwt_secret, "test-secret")
        self.assertEqual(config.jwt_expire_seconds, 7200)
        self.assertEqual(config.allowed_scopes["custom/method"], ["custom:scope"])


class TestAPIKeyManager(unittest.TestCase):
    """Test API key management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_keys_path = Path(self.temp_dir) / "test_api_keys.json"
        self.config = AuthConfig(api_keys_path=self.api_keys_path)
        self.manager = APIKeyManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_generate_api_key(self):
        """Test API key generation."""
        key = self.manager.generate_api_key("test-app")
        
        self.assertTrue(key.startswith("a2a_"))
        self.assertIn(key, self.manager.api_keys)
        self.assertEqual(self.manager.api_keys[key]["name"], "test-app")
        self.assertTrue(self.manager.api_keys[key]["active"])
        
        # Check file was saved
        self.assertTrue(self.api_keys_path.exists())
    
    def test_generate_api_key_with_scopes(self):
        """Test API key generation with custom scopes."""
        scopes = ["minion:query", "tasks:read"]
        key = self.manager.generate_api_key("limited-app", scopes)
        
        self.assertEqual(self.manager.api_keys[key]["scopes"], scopes)
    
    def test_validate_api_key(self):
        """Test API key validation."""
        key = self.manager.generate_api_key("test-app")
        
        # Valid key
        result = self.manager.validate_api_key(key)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "test-app")
        
        # Invalid key
        result = self.manager.validate_api_key("invalid_key")
        self.assertIsNone(result)
    
    def test_revoke_api_key(self):
        """Test API key revocation."""
        key = self.manager.generate_api_key("test-app")
        
        # Revoke key
        success = self.manager.revoke_api_key(key)
        self.assertTrue(success)
        
        # Validate revoked key
        result = self.manager.validate_api_key(key)
        self.assertIsNone(result)
        
        # Check revocation was saved
        self.assertFalse(self.manager.api_keys[key]["active"])
        self.assertIn("revoked_at", self.manager.api_keys[key])
    
    def test_load_existing_api_keys(self):
        """Test loading existing API keys from file."""
        # Create test data
        test_keys = {
            "test_key_1": {
                "name": "app1",
                "created_at": "2024-01-01",
                "scopes": ["tasks:read"],
                "active": True
            },
            "test_key_2": {
                "name": "app2",
                "created_at": "2024-01-02",
                "scopes": ["tasks:write"],
                "active": False
            }
        }
        
        with open(self.api_keys_path, 'w') as f:
            json.dump(test_keys, f)
        
        # Create new manager to load from file
        new_manager = APIKeyManager(self.config)
        
        self.assertEqual(len(new_manager.api_keys), 2)
        self.assertIn("test_key_1", new_manager.api_keys)
        self.assertIn("test_key_2", new_manager.api_keys)
        
        # Validate active key
        result = new_manager.validate_api_key("test_key_1")
        self.assertIsNotNone(result)
        
        # Validate inactive key
        result = new_manager.validate_api_key("test_key_2")
        self.assertIsNone(result)


class TestJWTManager(unittest.TestCase):
    """Test JWT token management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AuthConfig(jwt_secret="test-secret-key")
        self.manager = JWTManager(self.config)
    
    def test_create_and_verify_token(self):
        """Test JWT token creation and verification."""
        # Create token data
        exp_time = int((datetime.utcnow() + timedelta(hours=1)).timestamp())
        token_data = TokenData(
            sub="user123",
            scopes=["tasks:read", "tasks:write"],
            exp=exp_time
        )
        
        # Create token
        token = self.manager.create_token(token_data)
        self.assertIsInstance(token, str)
        
        # Verify token
        verified = self.manager.verify_token(token)
        self.assertIsNotNone(verified)
        self.assertEqual(verified.sub, "user123")
        self.assertEqual(verified.scopes, ["tasks:read", "tasks:write"])
    
    def test_expired_token(self):
        """Test expired token handling."""
        # Create expired token
        exp_time = int((datetime.utcnow() - timedelta(hours=1)).timestamp())
        print(f"Expired time: {exp_time}")
        print(f"Current time: {int(datetime.utcnow().timestamp())}")
        
        token_data = TokenData(sub="user123", exp=exp_time, scopes=[])  # Add scopes
        
        token = self.manager.create_token(token_data)
        print(f"Created token: {token}")
        
        # Try to decode manually to see what happens
        try:
            import jwt as jwt_lib
            decoded = jwt_lib.decode(token, self.manager.secret, 
                                   algorithms=[self.manager.algorithm], 
                                   options={"verify_exp": False})
            print(f"Token payload: {decoded}")
        except Exception as e:
            print(f"Manual decode error: {e}")
        
        # Verify should return None for expired token
        verified = self.manager.verify_token(token)
        print(f"Verification result: {verified}")
        self.assertIsNone(verified)
    
    def test_invalid_token(self):
        """Test invalid token handling."""
        # Verify invalid token
        verified = self.manager.verify_token("invalid.token.here")
        self.assertIsNone(verified)
    
    def test_token_without_secret(self):
        """Test JWT manager without provided secret."""
        config = AuthConfig()  # No jwt_secret
        manager = JWTManager(config)
        
        # Should generate a random secret
        self.assertIsNotNone(manager.secret)
        self.assertNotEqual(manager.secret, "")


class TestOAuth2ClientManager(unittest.TestCase):
    """Test OAuth2 client management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.clients_path = Path(self.temp_dir) / "test_oauth2_clients.json"
        self.manager = OAuth2ClientManager(self.clients_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_register_client(self):
        """Test OAuth2 client registration."""
        client_id, client_secret = self.manager.register_client(
            "test-client",
            ["minion:query", "tasks:read"]
        )
        
        self.assertTrue(client_id.startswith("oauth2_"))
        self.assertTrue(len(client_secret) > 20)
        self.assertIn(client_id, self.manager.clients)
        
        # Check file was saved
        self.assertTrue(self.clients_path.exists())
    
    def test_validate_client(self):
        """Test OAuth2 client validation."""
        client_id, client_secret = self.manager.register_client("test-client", [])
        
        # Valid credentials
        client = self.manager.validate_client(client_id, client_secret)
        self.assertIsNotNone(client)
        self.assertEqual(client.name, "test-client")
        
        # Invalid client ID
        client = self.manager.validate_client("invalid_id", client_secret)
        self.assertIsNone(client)
        
        # Invalid secret
        client = self.manager.validate_client(client_id, "wrong_secret")
        self.assertIsNone(client)
    
    def test_revoke_client(self):
        """Test OAuth2 client revocation."""
        client_id, client_secret = self.manager.register_client("test-client", [])
        
        # Revoke client
        success = self.manager.revoke_client(client_id)
        self.assertTrue(success)
        
        # Validate revoked client
        client = self.manager.validate_client(client_id, client_secret)
        self.assertIsNone(client)
    
    def test_list_clients(self):
        """Test listing OAuth2 clients."""
        # Register multiple clients
        id1, _ = self.manager.register_client("client1", ["scope1"])
        id2, _ = self.manager.register_client("client2", ["scope2"])
        
        clients = self.manager.list_clients()
        self.assertEqual(len(clients), 2)
        
        # Check that secrets are not exposed
        for client in clients:
            self.assertNotIn("client_secret", client)
            self.assertIn("client_id", client)
            self.assertIn("name", client)
            self.assertIn("scopes", client)


class TestAuthenticationManager(unittest.TestCase):
    """Test main authentication manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AuthConfig(
            api_keys_path=Path(self.temp_dir) / "api_keys.json",
            oauth2_clients_path=Path(self.temp_dir) / "oauth2_clients.json",
            jwt_secret="test-secret"
        )
        self.manager = AuthenticationManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_creates_defaults(self):
        """Test that initialization creates default credentials."""
        # Should have created default API key
        self.assertIsNotNone(self.manager.default_api_key)
        self.assertTrue(self.manager.default_api_key.startswith("a2a_"))
        
        # Should have created default OAuth2 client
        self.assertEqual(len(self.manager.oauth2_manager.clients), 1)
    
    async def test_authenticate_with_api_key(self):
        """Test authentication with API key."""
        # Create a test API key
        api_key = self.manager.api_key_manager.generate_api_key("test", ["tasks:read"])
        
        # Mock request and dependencies
        request = MagicMock(spec=Request)
        
        # Call authenticate function
        auth_func = self.manager.authenticate
        token_data = await auth_func(request, api_key=api_key, credentials=None)
        
        self.assertIsNotNone(token_data)
        self.assertEqual(token_data.sub, "test")
        self.assertEqual(token_data.scopes, ["tasks:read"])
    
    async def test_authenticate_with_jwt(self):
        """Test authentication with JWT bearer token."""
        # Create a JWT token
        token_data = TokenData(
            sub="jwt-user",
            scopes=["tasks:write"],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp())
        )
        jwt_token = self.manager.jwt_manager.create_token(token_data)
        
        # Mock credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=jwt_token
        )
        
        # Mock request
        request = MagicMock(spec=Request)
        
        # Call authenticate function
        auth_func = self.manager.authenticate
        result = await auth_func(request, api_key=None, credentials=credentials)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.sub, "jwt-user")
        self.assertEqual(result.scopes, ["tasks:write"])
    
    async def test_authenticate_fails_without_credentials(self):
        """Test authentication fails when no credentials provided."""
        request = MagicMock(spec=Request)
        
        auth_func = self.manager.authenticate
        
        with self.assertRaises(HTTPException) as context:
            await auth_func(request, api_key=None, credentials=None)
        
        self.assertEqual(context.exception.status_code, 401)
        self.assertIn("Authentication required", context.exception.detail)
    
    async def test_authenticate_optional(self):
        """Test optional authentication (require_auth=False)."""
        config = AuthConfig(require_auth=False)
        manager = AuthenticationManager(config)
        
        request = MagicMock(spec=Request)
        
        auth_func = manager.authenticate
        result = await auth_func(request, api_key=None, credentials=None)
        
        # Should return None instead of raising exception
        self.assertIsNone(result)
    
    def test_check_scopes(self):
        """Test scope checking."""
        # User with specific scopes
        token_data = TokenData(sub="user", scopes=["tasks:read", "minion:query"])
        
        # Check allowed scopes
        self.assertTrue(self.manager.check_scopes(token_data, ["tasks:read"]))
        self.assertTrue(self.manager.check_scopes(token_data, ["minion:query"]))
        self.assertTrue(self.manager.check_scopes(token_data, ["tasks:read", "minion:query"]))
        
        # Check disallowed scopes
        self.assertFalse(self.manager.check_scopes(token_data, ["tasks:write"]))
        self.assertFalse(self.manager.check_scopes(token_data, ["admin"]))
        
        # Admin scope allows everything
        admin_token = TokenData(sub="admin", scopes=["*"])
        self.assertTrue(self.manager.check_scopes(admin_token, ["anything"]))
    
    async def test_require_scopes_success(self):
        """Test require_scopes dependency with valid scopes."""
        token_data = TokenData(sub="user", scopes=["tasks:write"])
        request = MagicMock(spec=Request)
        request.scope = {"path": "/tasks/send"}
        
        # Get the dependency function
        scope_check = self.manager.require_scopes("tasks:write")
        
        # Should not raise
        result = await scope_check(request, token_data)
        self.assertEqual(result, token_data)
    
    async def test_require_scopes_failure(self):
        """Test require_scopes dependency with insufficient scopes."""
        token_data = TokenData(sub="user", scopes=["tasks:read"])
        request = MagicMock(spec=Request)
        request.scope = {"path": "/tasks/send"}
        
        # Get the dependency function
        scope_check = self.manager.require_scopes("tasks:write")
        
        # Should raise 403
        with self.assertRaises(HTTPException) as context:
            await scope_check(request, token_data)
        
        self.assertEqual(context.exception.status_code, 403)
        self.assertIn("Insufficient permissions", context.exception.detail)
    
    async def test_handle_oauth_token_success(self):
        """Test OAuth2 token endpoint with valid credentials."""
        # Register a client
        client_id, client_secret = self.manager.oauth2_manager.register_client(
            "test-oauth-client",
            ["tasks:read", "tasks:write"]
        )
        
        # Request token
        result = await self.manager.handle_oauth_token(
            grant_type="client_credentials",
            client_id=client_id,
            client_secret=client_secret,
            scope="tasks:read tasks:write"
        )
        
        self.assertIn("access_token", result)
        self.assertEqual(result["token_type"], "bearer")
        self.assertEqual(result["expires_in"], 3600)
        self.assertEqual(result["scope"], "tasks:read tasks:write")
        
        # Verify the token is valid
        token = result["access_token"]
        verified = self.manager.jwt_manager.verify_token(token)
        self.assertIsNotNone(verified)
        self.assertEqual(verified.sub, client_id)
    
    async def test_handle_oauth_token_invalid_grant(self):
        """Test OAuth2 token endpoint with invalid grant type."""
        with self.assertRaises(HTTPException) as context:
            await self.manager.handle_oauth_token(
                grant_type="password",  # Not supported
                client_id="test",
                client_secret="test",
                scope=""
            )
        
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("Unsupported grant type", context.exception.detail)
    
    async def test_handle_oauth_token_invalid_credentials(self):
        """Test OAuth2 token endpoint with invalid credentials."""
        with self.assertRaises(HTTPException) as context:
            await self.manager.handle_oauth_token(
                grant_type="client_credentials",
                client_id="invalid",
                client_secret="wrong",
                scope=""
            )
        
        self.assertEqual(context.exception.status_code, 401)
        self.assertIn("Invalid client credentials", context.exception.detail)


class TestAuthGlobals(unittest.TestCase):
    """Test global auth functions."""
    
    def test_init_and_get_auth_manager(self):
        """Test global auth manager initialization."""
        # Clear any existing manager
        import a2a_minions.auth
        a2a_minions.auth.auth_manager = None
        
        # Get should initialize
        manager1 = get_auth_manager()
        self.assertIsNotNone(manager1)
        
        # Get again should return same instance
        manager2 = get_auth_manager()
        self.assertIs(manager1, manager2)
        
        # Init with custom config
        config = AuthConfig(jwt_secret="custom-secret")
        manager3 = init_auth(config)
        self.assertEqual(manager3.config.jwt_secret, "custom-secret")
        
        # Get should now return the new instance
        manager4 = get_auth_manager()
        self.assertIs(manager3, manager4)


if __name__ == "__main__":
    unittest.main()