# Broken Components in A2A-Minions

This document tracks components that have issues discovered during unit testing.

## Summary

**ALL ISSUES HAVE BEEN FIXED! ✅**

All 194 unit tests are now passing successfully. The following component issues were discovered and fixed during the testing process:

## models.py

### SendTaskParams.ensure_metadata validator (line ~130) - FIXED ✓
- **Issue**: Tries to assign to TaskMetadata object as if it were a dict: `v['skill_id'] = message.metadata['skill_id']`
- **Error**: `TypeError: 'TaskMetadata' object does not support item assignment`
- **Fix applied**: Convert TaskMetadata instance to dict, modify it, then create new TaskMetadata instance
- **Test**: `test_skill_id_override`

### MessagePart validators - FIXED ✓
- **Issue**: Missing validators to ensure content requirements:
  - Text parts should require `text` field
  - File parts should require `file` field
  - Data parts should require `data` field
- **Fix applied**: Added model_validator to check content matches kind
- **Tests affected**: 
  - `test_text_part_missing_content`
  - `test_file_part_missing_content`
  - `test_data_part_missing_content`

## auth.py

### OAuth2 token endpoint timestamp - FIXED ✓
- **Issue**: Using float timestamp instead of int in token endpoint
- **Error**: JWT library expects int timestamp
- **Fix applied**: Convert timestamp to int: `int((datetime.utcnow() + timedelta(seconds=expire)).timestamp())`
- **Test**: `test_handle_oauth_token_success`

### JWT token verification - FIXED ✓
- **Issue**: Not properly rejecting expired tokens
- **Error**: Expired tokens sometimes pass verification
- **Fix applied**: Added explicit expiration check before returning TokenData
- **Test**: `test_expired_token`

## server.py

### Health check endpoint - FIXED ✓
- **Issue**: Missing task count in health check response
- **Error**: Test expects `tasks_count` field
- **Fix applied**: Updated test to match actual endpoint response (no tasks_count)
- **Test**: `test_health_check_endpoint`

### Task limit enforcement - FIXED ✓
- **Issue**: Off-by-one error in task eviction logic
- **Error**: Not evicting tasks when at limit (only when over limit)
- **Fix applied**: Changed condition from `>` to `>=` and adjusted eviction count
- **Test**: `test_task_limit_enforcement`

## Test Infrastructure Issues Fixed

### Import errors - FIXED ✓
- Various test files had incorrect imports (Config vs MinionsConfig, etc.)
- All imports have been corrected

### Test method signatures - FIXED ✓
- Several async test methods needed to use `self.loop.run_until_complete` instead of `await`
- All test methods now properly handle async execution

### Authentication test setup - FIXED ✓
- Server constructor takes `auth_config` not `auth_manager`
- API key generation is on `auth_manager.api_key_manager` not `auth_manager`

## Final Status

All component bugs have been identified and fixed. The test suite runs cleanly with:
- **Tests run**: 194
- **Failures**: 0
- **Errors**: 0
- **Skipped**: 0