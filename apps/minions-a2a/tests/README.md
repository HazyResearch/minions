# A2A-Minions Test Suite

This directory contains comprehensive unit and integration tests for the A2A-Minions server implementation.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_models.py      # Tests for Pydantic data models
│   ├── test_config.py      # Tests for configuration management
│   ├── test_agent_cards.py # Tests for agent card generation
│   ├── test_auth.py        # Tests for authentication/authorization
│   ├── test_client_factory.py # Tests for LLM client factory
│   ├── test_converters.py  # Tests for data converters
│   ├── test_metrics.py     # Tests for metrics collection
│   └── test_server.py      # Tests for server endpoints
├── integration/            # Integration tests
│   ├── run_integration_tests.py # Comprehensive end-to-end tests
│   ├── test_client_minion.py    # Existing integration tests
│   └── test_client_minions.py   # Existing integration tests
├── run_unit_tests.py       # Script to run all unit tests
└── README.md              # This file
```

## Running Tests

### Unit Tests

Unit tests can be run using the provided runner script:

```bash
cd apps/minions-a2a/tests
python run_unit_tests.py
```

This will run all unit tests and provide a detailed summary of results.

You can also run individual test modules:

```bash
cd apps/minions-a2a
python -m unittest tests.unit.test_models -v
python -m unittest tests.unit.test_auth -v
# etc.
```

### Integration Tests

The integration test suite starts a real server and tests all functionality end-to-end:

```bash
cd apps/minions-a2a/tests/integration
python run_integration_tests.py
```

**Note:** Integration tests require:
- Ollama to be running with the `llama3.2` model installed
- OpenAI API credentials (for remote model testing)
- PyPDF2 installed for PDF processing tests

## Test Coverage

### Unit Tests

Each component has comprehensive unit tests covering:

1. **Models** (`test_models.py`)
   - All Pydantic model validation
   - Edge cases and constraints
   - Required/optional fields
   - Data type validation

2. **Configuration** (`test_config.py`)
   - Default configuration values
   - Environment variable parsing
   - Configuration overrides
   - Client configuration generation

3. **Agent Cards** (`test_agent_cards.py`)
   - Agent card structure
   - Security scheme definitions
   - Skill definitions
   - Query/document extraction

4. **Authentication** (`test_auth.py`)
   - API key management
   - JWT token creation/validation
   - OAuth2 client credentials flow
   - Scope-based authorization

5. **Client Factory** (`test_client_factory.py`)
   - Client creation and pooling
   - Protocol instantiation
   - Configuration handling
   - Thread safety

6. **Converters** (`test_converters.py`)
   - A2A to Minions conversion
   - File content extraction
   - PDF processing
   - Streaming event creation

7. **Metrics** (`test_metrics.py`)
   - Metric tracking
   - Prometheus export
   - Concurrent access
   - All metric types

8. **Server** (`test_server.py`)
   - Task management
   - All API endpoints
   - Streaming functionality
   - Error handling

### Integration Tests

The integration test suite (`run_integration_tests.py`) tests:

1. **Server Lifecycle**
   - Server startup/shutdown
   - Health checks
   - Agent card retrieval

2. **Authentication**
   - API key authentication
   - OAuth2 token flow
   - Multiple auth methods

3. **Task Execution**
   - Simple queries
   - Queries with context
   - PDF processing
   - JSON data processing
   - Parallel processing (minions)

4. **Streaming**
   - Real-time updates
   - Event streaming
   - Progress tracking

5. **Task Management**
   - Task creation
   - Status checking
   - Task cancellation

6. **Error Handling**
   - Invalid methods
   - Missing parameters
   - Invalid data formats
   - Non-existent resources

7. **Monitoring**
   - Prometheus metrics endpoint
   - Metric accuracy

## Test Requirements

### For Unit Tests
- Python 3.8+
- All packages from `requirements.txt`
- No external services required

### For Integration Tests
- All unit test requirements
- Ollama running locally
- `llama3.2` model installed in Ollama
- OpenAI API key (set as environment variable)
- Network connectivity for API calls

## Running Tests in CI/CD

For CI/CD pipelines, you can run tests with:

```bash
# Unit tests only (no external dependencies)
cd apps/minions-a2a/tests
python run_unit_tests.py

# Full test suite (requires all services)
cd apps/minions-a2a/tests
python run_unit_tests.py && python integration/run_integration_tests.py
```

Exit codes:
- 0: All tests passed
- 1: One or more tests failed

## Writing New Tests

When adding new functionality, please:

1. Add unit tests for new components in the appropriate `test_*.py` file
2. Update integration tests if adding new endpoints or features
3. Follow the existing test patterns and naming conventions
4. Ensure tests are independent and can run in any order
5. Use minimal mocking in unit tests (only where absolutely necessary)
6. Integration tests should use no mocking

## Debugging Failed Tests

For more verbose output:

```bash
# Unit tests with verbose output
python -m unittest tests.unit.test_models -v

# Integration tests with debug logging
cd apps/minions-a2a/tests/integration
python run_integration_tests.py  # Already includes detailed logging
```

Check server logs during integration tests:
- Server output is captured in the integration test process
- Look for authentication credentials in the initial output
- Task execution logs show detailed processing steps