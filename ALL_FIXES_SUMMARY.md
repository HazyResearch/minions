# A2A-Minions Server - All Fixes Applied

## Summary
Fixed multiple issues preventing the A2A-Minions server from starting and handling requests properly.

## Issue 1: Validation Errors (400 Bad Request)
**Problem**: Task model mismatch - server was creating tasks with incorrect fields
**Fix**: Updated `create_task` method to match the Task model:
- Added required `sessionId` field
- Added required `history` field containing the message
- Use proper `TaskStatus` object instead of dict
- Moved `created_at` and `created_by` into metadata

## Issue 2: TypeError with ValidationError (500 Internal Server Error) 
**Problem**: Incorrect ValidationError instantiation
**Fix**: Changed all occurrences from:
```python
raise ValidationError(f"Invalid parameters: {e}")
```
To:
```python
raise e  # Re-raise the original pydantic ValidationError
```

## Issue 3: Event Loop Error on Startup
**Problem**: "no running event loop" - TaskManager trying to create async tasks during `__init__`
**Fix**: 
- Deferred background task creation to a `start()` method
- Added FastAPI startup event handler to call `await self.task_manager.start()`
- Fixed `asyncio.Event()` creation to happen when event loop is running
- Simplified server run method to use `uvicorn.run()` directly

## Issue 4: Authentication Error
**Problem**: `APIKeyHeader.__call__() missing 1 required positional argument: 'request'`
**Fix**: 
- Refactored `authenticate` from a regular method to a `@property`
- Returns a dependency function that properly uses `Depends()`
- Fixed return type to always return proper `TokenData` objects

## Issue 5: JSON Serialization Error
**Problem**: `Object of type ModelMetaclass is not JSON serializable` when caching client configs
**Fix**:
- Modified `_get_client_key` method in `client_factory.py` to exclude non-serializable fields
- Filters out `structured_output_schema` which contains Pydantic model classes

## Files Modified
1. `apps/minions-a2a/a2a_minions/server.py` - Task creation, validation handling, event loop fixes
2. `apps/minions-a2a/a2a_minions/auth.py` - Authentication dependency fixes
3. `apps/minions-a2a/a2a_minions/client_factory.py` - JSON serialization fix for client caching

## Testing
To verify all fixes work:
1. Start the server: `python apps/minions-a2a/run_server.py --port 8001 --api-key "abcd"`
2. Run tests: `python apps/minions-a2a/tests/test_client_minions.py`

The server should now:
- Start without errors ✅
- Handle authentication properly ✅  
- Process requests without validation errors ✅
- Create and cache Minions protocol clients successfully ✅