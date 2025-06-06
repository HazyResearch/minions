# Event Loop Fix for A2A-Minions Server

## Issue
The server failed to start with the error: "no running event loop"

This occurred because the `TaskManager` was trying to create async tasks during initialization when no event loop was running yet. Specifically:
- `asyncio.create_task()` was called in `__init__`
- `asyncio.Event()` was created before the event loop started

## Fixes Applied

### 1. Deferred TaskManager Background Tasks
- Removed `self._start_cleanup_task()` and `self._start_metrics_task()` from `__init__`
- Created an async `start()` method that initializes background tasks
- Added a startup event handler that calls `await self.task_manager.start()`

### 2. Fixed asyncio.Event Creation
- Changed `self._shutdown_event = asyncio.Event()` to `self._shutdown_event = None` in init
- Create the event in the startup handler when the event loop is running

### 3. Updated Shutdown Handler
- Added check `if self._shutdown_event:` before calling `set()` to handle signals before startup

### 4. Simplified Server Run Method
- Replaced custom async serve logic with direct `uvicorn.run()`
- This avoids issues with shutdown_event being accessed before creation

## Result
The server should now start successfully without event loop errors. The background tasks for cleanup and metrics will start when the FastAPI app starts up.