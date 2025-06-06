# Authentication Fix for A2A-Minions Server

## Issue
The server was encountering a TypeError when trying to authenticate requests:
```
TypeError: APIKeyHeader.__call__() missing 1 required positional argument: 'request'
```

This occurred because the `authenticate` method was trying to call the security dependencies directly:
```python
api_key = await self.api_key_header() if not api_key else api_key
```

## Root Cause
FastAPI security dependencies (like `APIKeyHeader` and `HTTPBearer`) are designed to be used with the `Depends()` system, not called directly. They need the request object to be injected by FastAPI.

## Fix Applied
Refactored the `authenticate` method to be a property that returns a dependency function:

1. Changed `authenticate` from a regular method to a `@property` decorator
2. Returns an inner `_authenticate` function that properly uses `Depends()`
3. Fixed the return type to always return a proper `TokenData` object

The new structure:
```python
@property
def authenticate(self):
    """Return authentication dependency function."""
    async def _authenticate(
        request: Request,
        api_key: Optional[str] = Depends(self.api_key_header),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.bearer_scheme)
    ) -> Optional[TokenData]:
        # Authentication logic here
```

## Result
The authentication system now properly integrates with FastAPI's dependency injection system, allowing the server to handle API key and bearer token authentication correctly.