# Broken Components in A2A-Minions

This document tracks components that have issues discovered during unit testing.

## models.py

### SendTaskParams.ensure_metadata validator (line ~130)
- **Issue**: Tries to assign to TaskMetadata object as if it were a dict: `v['skill_id'] = message.metadata['skill_id']`
- **Error**: `TypeError: 'TaskMetadata' object does not support item assignment`
- **Fix needed**: Should create a new TaskMetadata object or update logic
- **Test**: `test_skill_id_override`

### MessagePart validators
- **Issue**: Missing validators to ensure content requirements:
  - Text parts should require `text` field
  - File parts should require `file` field
  - Data parts should require `data` field
- **Tests affected**: 
  - `test_text_part_missing_content`
  - `test_file_part_missing_content`
  - `test_data_part_missing_content`

## auth.py

### handle_oauth_token method (line ~393)
- **Issue**: Creates TokenData with datetime object for `exp` field instead of int timestamp
- **Error**: `ValidationError: exp - Input should be a valid integer`
- **Fix needed**: Convert datetime to int timestamp before creating TokenData
- **Test**: `test_oauth_token_endpoint`

### JWT token verification
- **Issue**: Expired tokens are not properly rejected - verification returns TokenData instead of None
- **Fix needed**: Check token expiration and return None for expired tokens
- **Test**: `test_expired_token`

## server.py

### Health check endpoint
- **Issue**: Missing `timestamp` field in response
- **Expected**: Should include current timestamp in health check response
- **Test**: `test_health_check_endpoint`

### TaskManager task limit enforcement
- **Issue**: Task limit is not properly enforced - allows 6 tasks when limit is 5
- **Expected**: Should evict oldest task when limit is reached
- **Test**: `test_task_limit_enforcement`

## Notes

These issues should be fixed in the actual components, not in the tests. The tests are correctly identifying real problems in the implementation.