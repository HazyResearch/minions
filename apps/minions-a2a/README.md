# A2A-Minions: Agent-to-Agent Integration with Minions Protocol

A comprehensive A2A (Agent-to-Agent) server implementation that provides seamless integration with the Minions protocol, enabling both focused single-document analysis and parallel processing capabilities for complex multi-document tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Skills and Protocol Selection](#skills-and-protocol-selection)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Monitoring](#monitoring)
- [Production Deployment](#production-deployment)

## Overview

A2A-Minions bridges the gap between A2A protocol agents and the Minions framework, providing:

- **Dual Protocol Support**: Skills for both Minion and Minion*S* protocols
- **Document Processing**: Full support for text, files (including PDFs), and structured data
- **Streaming Responses**: Real-time task execution updates
- **Cost-Efficient Architecture**: Leverages both local and cloud models strategically

## Features

- **A2A Protocol Compliance**: Full implementation of the Agent-to-Agent protocol specification
- **Dual Query Skills**: 
  - `minion_query`: Single-agent focused analysis
  - `minions_query`: Multi-agent parallel processing
- **Streaming Support**: Real-time responses via Server-Sent Events
- **Multi-modal Input**: Handles text, files (PDF), data (JSON), and images
- **Authentication**: API Keys, JWT tokens, and OAuth2 client credentials
- **Task Management**: Async task execution with status tracking
- **Error Handling**: Comprehensive validation and error responses
- **Monitoring**: Prometheus metrics for production observability

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   A2A Client    │────│  A2A-Minions    │────│ Minions Protocol│
│                 │    │     Server      │    │                 │
│ • Skill Request │    │ • Skill Router  │    │ • Local Model   │
│ • Document Send │    │ • Message Conv. │    │ • Remote Model  │
│ • Stream Listen │    │ • Task Manager  │    │ • Parallel Proc.│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Breakdown

1. **A2A Server**: Handles A2A protocol compliance and request routing
2. **Skill Router**: Automatically selects appropriate Minions protocol based on skill ID
3. **Message Converter**: Translates between A2A and Minions message formats
4. **Task Manager**: Manages long-running tasks with streaming capabilities
5. **Client Factory**: Creates and configures Minions protocol instances

## Installation

### Prerequisites

- Python 3.10+
- Access to at least one supported model provider (Ollama, OpenAI, etc.)
- The main Minions repository

### Installation Steps

1. **Clone the Minions repository**:
   ```bash
   git clone <repository-url>
   cd minions
   ```

2. **Install Minions in development mode**:
   ```bash
   pip install -e .
   ```

3. **Install A2A-specific dependencies**:
   ```bash
   cd apps/minions-a2a
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (required for some model providers):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
   export OLLAMA_HOST="http://localhost:11434"    # Optional, defaults to localhost
   ```

### Quick Verification

To verify installation:
```bash
# Check dependencies
python run_server.py --skip-checks

# Run with environment checks
python run_server.py
```

## Quick Start

### Start the Server

```bash
# From the apps/minions-a2a directory
python run_server.py --port 8001
```

The server will be available at `http://localhost:8001`.

### Test Basic Functionality

Run the test clients to verify everything is working:

```bash
# Test focused analysis (minion_query skill)
python tests/test_client_minion.py

# Test parallel processing (minions_query skill)
python tests/test_client_minions.py
```

### Check Server Status

- Health check: `http://localhost:8001/health`
- Agent card: `http://localhost:8001/.well-known/agent.json`

## Configuration
### Server Configuration

The server supports various configuration options:

```python
# Default configuration in config.py
class MinionsConfig:
    # Model settings
    local_provider: str = "ollama"
    local_model: str = "llama3.2"
    remote_provider: str = "openai" 
    remote_model: str = "gpt-4o-mini"
    
    # Processing settings
    max_rounds: int = 3
    max_jobs_per_round: int = 5
    num_tasks_per_round: int = 2
    num_samples_per_task: int = 1
    
    # Context settings
    num_ctx: int = 128000
    chunking_strategy: str = "chunk_by_section"
```

## Authentication

A2A-Minions implements A2A-compatible authentication following the protocol's security standards. The server supports multiple authentication methods as defined in the [A2A specification](https://auth0.com/blog/auth0-google-a2a/).

### Authentication Methods

The server supports three authentication schemes:

1. **API Key Authentication** (Default for local deployments)
   - Header: `X-API-Key: <your-api-key>`
   - Best for: Local testing, simple deployments

2. **Bearer Token Authentication** (JWT)
   - Header: `Authorization: Bearer <token>`
   - Best for: Production deployments, time-limited access

3. **OAuth2 Client Credentials Flow**
   - Endpoint: `POST /oauth/token`
   - Best for: Machine-to-machine (M2M) authentication, enterprise integrations

### Quick Start with Authentication

#### Running Without Authentication (Testing Only)
```bash
python run_server.py --no-auth
```

#### Running With Default Authentication
```bash
python run_server.py
# A default API key will be generated and displayed
# Save it securely - it won't be shown again!
```

#### Running With Custom API Key
```bash
python run_server.py --api-key "your-custom-api-key"
```

### Managing API Keys

Use the included CLI tool to manage API keys:

```bash
# List all API keys
python manage_api_keys.py list

# Generate a new API key
python manage_api_keys.py generate "my-client-name"

# Generate with specific scopes
python manage_api_keys.py generate "limited-client" --scopes minion:query tasks:read

# Revoke an API key (use last 8 characters)
python manage_api_keys.py revoke "abc12345"

# Export keys for backup
python manage_api_keys.py export backup_keys.json
```

### Available Scopes

The A2A-Minions server uses fine-grained scopes for authorization:

- `minion:query` - Execute focused minion queries
- `minions:query` - Execute parallel minions queries  
- `tasks:read` - Read task status and results
- `tasks:write` - Create and cancel tasks

### OAuth2 Client Credentials Flow

To obtain an access token using OAuth2:

```bash
curl -X POST http://localhost:8001/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=your-client-id" \
  -d "client_secret=your-client-secret" \
  -d "scope=minion:query tasks:read"
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "scope": "minion:query tasks:read"
}
```

### Using Authentication in Requests

#### With API Key
```python
headers = {
    "X-API-Key": "a2a_your_api_key_here",
    "Content-Type": "application/json"
}
```

#### With Bearer Token
```python
headers = {
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "Content-Type": "application/json"
}
```

### Security Best Practices

1. **Never commit API keys** - The `api_keys.json` file is gitignored by default
2. **Use environment variables** for JWT secrets: `export A2A_JWT_SECRET="your-secret"`
3. **Rotate keys regularly** - Use the CLI tool to generate new keys and revoke old ones
4. **Use minimal scopes** - Only grant the permissions needed for each client
5. **Enable HTTPS in production** - Authentication tokens should only be sent over encrypted connections

### Agent Card Security

The agent card advertises the supported security schemes:

```json
{
  "securitySchemes": {
    "api_key": {
      "type": "apiKey",
      "name": "X-API-Key",
      "in": "header"
    },
    "bearer_auth": {
      "type": "http",
      "scheme": "bearer",
      "bearerFormat": "JWT"
    },
    "oauth2_client_credentials": {
      "type": "oauth2",
      "flows": {
        "clientCredentials": {
          "tokenUrl": "http://localhost:8001/oauth/token",
          "scopes": {
            "minion:query": "Execute focused minion queries",
            "minions:query": "Execute parallel minions queries",
            "tasks:read": "Read task status and results",
            "tasks:write": "Create and cancel tasks"
          }
        }
      }
    }
  }
}
```

## Skills and Protocol Selection

### Automatic Skill Detection

The system automatically routes requests to the appropriate protocol based on the skill ID specified in the request metadata:

#### `minion_query` Skill
- **Purpose**: Focused analysis and single-document Q&A
- **Protocol**: Uses Minion (singular) for cost-efficient processing
- **Best For**: 
  - Specific questions about documents
  - Quick fact extraction
  - Simple analysis tasks
  - Single conversation flows

#### `minions_query` Skill  
- **Purpose**: Complex parallel processing and multi-document analysis
- **Protocol**: Uses Minions (parallel) for distributed processing
- **Best For**:
  - Large document analysis
  - Multi-document processing
  - Complex research tasks
  - Parallel task decomposition

### Skill Selection Examples

```json
// For focused analysis
{
  "metadata": {
    "skill_id": "minion_query",
    "max_rounds": 2
  }
}

// For parallel processing
{
  "metadata": {
    "skill_id": "minions_query", 
    "max_rounds": 3,
    "max_jobs_per_round": 5,
    "num_tasks_per_round": 3
  }
}
```

## API Reference

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns server status and health information.

#### Agent Card
```http
GET /.well-known/agent.json
```
Returns the public agent card with available skills.

#### Extended Agent Card
```http
GET /agent/authenticatedExtendedCard
```
Returns extended agent capabilities for authenticated users.

### Task Management

#### Send Task
```http
POST /
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "id": "task-uuid",
    "message": {
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Your question here"
        },
        {
          "kind": "text", 
          "text": "Document content here"
        }
      ]
    },
    "metadata": {
      "skill_id": "minion_query",
      "max_rounds": 2
    }
  },
  "id": "request-uuid"
}
```

#### Send Task with Streaming
```http
POST /
Content-Type: application/json
Accept: text/event-stream

{
  "jsonrpc": "2.0",
  "method": "tasks/sendSubscribe",
  "params": {
    "id": "task-uuid",
    "message": { /* same as above */ },
    "metadata": { /* same as above */ }
  },
  "id": "request-uuid"
}
```

#### Get Task Status
```http
POST /
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "tasks/get",
  "params": {
    "id": "task-uuid"
  },
  "id": "request-uuid"
}
```

#### Cancel Task
```http
POST /
Content-Type: application/json

{
  "jsonrpc": "2.0", 
  "method": "tasks/cancel",
  "params": {
    "id": "task-uuid"
  },
  "id": "request-uuid"
}
```

### Message Formats

#### Text Input
```json
{
  "kind": "text",
  "text": "Your question or document content"
}
```

#### File Input
```json
{
  "kind": "file",
  "file": {
    "name": "document.pdf",
    "mimeType": "application/pdf", 
    "bytes": "base64-encoded-content"
  }
}
```

#### Data Input
```json
{
  "kind": "data",
  "data": {
    "key": "value",
    "nested": {
      "data": "structure"
    }
  }
}
```

## Testing

### Running Tests

First, ensure the server is running with the test API key:

```bash
# Start server with test API key "abcd"
python run_server.py --api-key "abcd"

# Or add the key using the management tool
python manage_api_keys.py generate "test" --scopes minion:query minions:query tasks:read tasks:write
# Then use the generated key in tests
```

Then run the tests:

```bash
# Test focused analysis (minion_query)
python tests/test_client_minion.py

# Test parallel processing (minions_query) 
python tests/test_client_minions.py

# Custom server URL
python tests/test_client_minion.py --base-url http://localhost:8001
```

### Test Coverage

The test suites include:

- **Health and Discovery**: Server health, agent cards
- **Basic Functionality**: Simple queries, context processing
- **Document Processing**: PDF files, JSON data, multi-modal inputs
- **Streaming**: Real-time response handling
- **Error Handling**: Malformed requests, empty inputs
- **Task Management**: Status checking, cancellation
- **Long Context**: Large document processing capabilities

### Creating Custom Tests

```python
import asyncio
from test_client import A2AMinionsTestClient

async def custom_test():
    client = A2AMinionsTestClient("http://localhost:8000")
    
    message = {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Your question"},
            {"kind": "text", "text": "Your document"}
        ]
    }
    
    metadata = {
        "skill_id": "minion_query",  # or "minions_query"
        "max_rounds": 2
    }
    
    response = await client.send_task(message, metadata)
    task_id = response["result"]["id"]
    
    # Wait for completion
    result = await client.wait_for_completion(task_id)
    print(f"Result: {result}")
    
    await client.close()

asyncio.run(custom_test())
```

## Examples

### Example 1: Simple Document Q&A

```python
# Using minion_query for focused analysis
message = {
    "role": "user", 
    "parts": [
        {
            "kind": "text",
            "text": "What are the key findings mentioned in this research paper?"
        },
        {
            "kind": "text",
            "text": "Research paper content here..."
        }
    ]
}

metadata = {
    "skill_id": "minion_query",
    "local_provider": "ollama",
    "local_model": "llama3.2", 
    "remote_provider": "openai",
    "remote_model": "gpt-4o-mini",
    "max_rounds": 2
}
```

### Example 2: Complex Multi-Document Analysis

```python
# Using minions_query for parallel processing
message = {
    "role": "user",
    "parts": [
        {
            "kind": "text", 
            "text": "Analyze this large document and extract all performance metrics, categorize findings, and identify key trends."
        },
        {
            "kind": "file",
            "file": {
                "name": "large_report.pdf",
                "mimeType": "application/pdf",
                "bytes": "base64-encoded-pdf-content"
            }
        }
    ]
}

metadata = {
    "skill_id": "minions_query",
    "max_rounds": 3,
    "max_jobs_per_round": 5,
    "num_tasks_per_round": 3,
    "num_samples_per_task": 2
}
```

### Example 3: JSON Data Processing

```python
# Processing structured data
complex_data = {
    "quarterly_reports": {
        "Q1_2024": {"revenue": 2500000, "expenses": 1800000},
        "Q2_2024": {"revenue": 2800000, "expenses": 2000000}
    },
    "performance_metrics": {
        "customer_satisfaction": 4.7,
        "employee_retention": 0.92
    }
}

message = {
    "role": "user",
    "parts": [
        {
            "kind": "text",
            "text": "Calculate revenue growth and identify trends in this business data."
        },
        {
            "kind": "data", 
            "data": complex_data
        }
    ]
}

metadata = {
    "skill_id": "minions_query",  # Use parallel processing for complex analysis
    "max_rounds": 2,
    "max_jobs_per_round": 4
}
```

### Example 4: Streaming Response

```python
import json

async def stream_example():
    client = A2AMinionsTestClient()
    
    # Send streaming request
    task_id = await client.send_task_streaming(message, metadata)
    
    # Process real-time updates
    # (streaming events are automatically printed by the client)
    
    # Get final result
    final_task = await client.wait_for_completion(task_id)
    print(f"Final result: {final_task}")
```

## Troubleshooting

### Common Issues

#### 1. Skill Detection Problems
**Symptoms**: Wrong protocol being used, unexpected errors
**Solution**: Ensure `skill_id` is properly specified in metadata:
```json
{
  "metadata": {
    "skill_id": "minion_query"  // or "minions_query"
  }
}
```

#### 2. Context Not Being Used
**Symptoms**: Model responds "I don't have access to documents"
**Solution**: Check document formatting and ensure context is being extracted:
- Verify file uploads are base64 encoded correctly
- Check logs for context validation messages
- Ensure document content is non-empty

#### 3. Parameter Errors
**Symptoms**: "Unexpected keyword argument" errors
**Solution**: Verify skill and parameter compatibility:
- `minion_query`: Uses basic parameters (max_rounds, images)
- `minions_query`: Uses parallel parameters (max_jobs_per_round, num_tasks_per_round)

#### 4. Model Provider Issues
**Symptoms**: Connection errors, authentication failures
**Solution**: 
- Check environment variables are set correctly
- Verify model provider endpoints are accessible
- Ensure API keys have sufficient permissions

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Analysis

Key log messages to monitor:

```
INFO:a2a_minions.server:Executing skill: minions_query for task: <task-id>
INFO:a2a_minions.server:Context validation: X items, Y total characters
ERROR:a2a_minions.server:Task <task-id> failed: <error-message>
```

### Performance Tuning

#### For Large Documents
```json
{
  "skill_id": "minions_query",
  "max_rounds": 2,
  "max_jobs_per_round": 8,
  "num_tasks_per_round": 4,
  "chunking_strategy": "chunk_by_section"
}
```

#### For Quick Responses
```json
{
  "skill_id": "minion_query", 
  "max_rounds": 1,
  "local_model": "llama3.2:1b",
  "remote_model": "gpt-4o-mini"
}
```

## Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd minions/apps/minions-a2a
python -m venv dev-env
source dev-env/bin/activate
```

### Submission Guidelines

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues, questions, or contributions:

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check this README and inline code documentation

---

**A2A-Minions** - Bridging Agent-to-Agent communication with the power of the Minions protocol for efficient, scalable document processing and analysis.

### OAuth2 Client Management

Register and manage OAuth2 clients:

```bash
# List all OAuth2 clients
python manage_oauth2_clients.py list

# Register a new OAuth2 client
python manage_oauth2_clients.py register "my-app" --scopes minion:query tasks:read

# Register with all scopes
python manage_oauth2_clients.py register "admin-app"

# Revoke a client
python manage_oauth2_clients.py revoke oauth2_xxxxx

# Export client list (without secrets)
python manage_oauth2_clients.py export --output clients.json
```

### Using OAuth2 Authentication

After registering a client, use the client credentials flow:

```bash
# Get access token
curl -X POST http://localhost:8000/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "scope=minion:query tasks:read"

# Use the token
curl http://localhost:8000/agent/authenticatedExtendedCard \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Monitoring

The server exposes Prometheus metrics at `/metrics`:

```bash
# View raw metrics
curl http://localhost:8000/metrics

# Key metrics exposed:
# - a2a_minions_requests_total: Request count by method and status
# - a2a_minions_request_duration_seconds: Request latency
# - a2a_minions_tasks_total: Task count by skill and status
# - a2a_minions_task_duration_seconds: Task execution time
# - a2a_minions_active_tasks: Currently running tasks
# - a2a_minions_auth_attempts_total: Auth attempts by method
# - a2a_minions_pdf_processed_total: PDFs processed
# - a2a_minions_errors_total: Error count by type
```

Example Prometheus configuration:
```yaml
scrape_configs:
  - job_name: 'a2a-minions'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## Production Deployment

Before deploying to production, review the comprehensive checklist:

📋 **[PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)**

Key areas covered:
- Environment configuration
- Security hardening
- Infrastructure requirements
- Monitoring setup
- Performance optimization
- Operational procedures