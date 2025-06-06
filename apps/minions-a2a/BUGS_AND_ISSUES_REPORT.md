# A2A-Minions: Fixed Issues and Remaining Work

## Fixed Issues (Completed)

### 1. ✅ Missing PyPDF2 Dependency
- **Fix Applied**: Added `PyPDF2>=3.0.0` to dependencies in `pyproject.toml`
- **Impact**: PDF processing will now work correctly in production

### 2. ✅ Missing Skill ID Error Handling  
- **Fix Applied**: Added try/catch block in `server.py` to handle ValueError from `extract_skill_id()`
- **Impact**: Now returns proper 400 error with clear message instead of 500 error

### 3. ✅ Path Traversal Vulnerability
- **Fix Applied**: Sanitized filenames in `converters.py` `_save_temp_file()` method
- **Impact**: Prevents arbitrary file writes through malicious filenames

### 4. ✅ Context Validation Logic Issue
- **Fix Applied**: Fixed logic in `server.py` to only add context header when valid context exists
- **Impact**: No more misleading headers sent to workers for empty context

### 5. ✅ Generic Exception Catching
- **Fix Applied**: Improved exception handling in `client_factory.py` with specific exceptions and logging
- **Impact**: Better error messages and easier debugging

## Remaining Issues (To Be Done)

### Critical Issues
1. **Thread Safety Issue in Streaming Callback** - Requires replacing `call_soon_threadsafe` with `run_coroutine_threadsafe`
2. **Unbounded Task Storage Memory Leak** - Need task cleanup mechanism and retention policy
3. **Missing Timeout Configuration** - Tasks can run indefinitely

### Security Issues  
4. **No Authentication/Authorization** - Service is completely open
5. **No Rate Limiting** - Can be abused

### Error Handling Issues
6. **Missing Request Validation** - Need comprehensive Pydantic models for inputs
7. **No Graceful Shutdown** - In-flight requests lost on shutdown

### Performance Issues
8. **Synchronous File Operations** - PDF extraction blocks event loop
9. **No Connection Pooling** - Creates new clients for each request

### Missing Features
10. **No Metrics or Monitoring** - Can't track service health
11. **No Request Correlation** - Hard to trace requests
12. **No Health Check with Dependencies** - Basic health check doesn't verify dependencies

### Data Validation
13. **File Size Limits** - No limits on uploaded file sizes
14. **Content Type Validation** - Limited mime type validation

### Code Quality
15. **Missing Type Hints** - Several methods lack proper type annotations
16. **Inconsistent Logging** - Mix of print statements and logger calls
17. **No Structured Logging** - Logs lack correlation IDs and structured format

## Next Steps

1. **Immediate Priority**: 
   - Implement timeout configuration
   - Add task cleanup mechanism
   - Fix thread safety in streaming

2. **Short Term**:
   - Add authentication
   - Implement request validation
   - Add connection pooling

3. **Long Term**:
   - Add comprehensive monitoring
   - Implement distributed task storage
   - Performance optimization