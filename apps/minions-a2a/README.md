# A2A-Minions: Agent-to-Agent Integration with Minions Protocol

A comprehensive A2A (Agent-to-Agent) server implementation that provides seamless integration with the Minions protocol, enabling both focused single-document analysis and parallel processing capabilities for complex multi-document tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Skills and Protocol Selection](#skills-and-protocol-selection)
- [API Reference](#api-reference)
- [Authentication](#authentication)
- [Testing](#testing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Operations](#operations)
  - [Monitoring](#monitoring)
  - [Production Deployment](#production-deployment)
- [Contributing](#contributing)

## Overview

A2A-Minions bridges the gap between A2A protocol agents and the Minions framework, providing:

- **Dual Protocol Support**: Skills for both Minion and Minions protocols
- **Document Processing**: Full support for text, files (including PDFs), and structured data
- **Streaming Responses**: Real-time task execution updates
- **Cost-Efficient Architecture**: Leverages both local and cloud models strategically
- **A2A Protocol Compliance**: Full implementation of the Agent-to-Agent protocol specification
- **Multi-modal Input**: Handles text, files (PDF), data (JSON), and images
- **Authentication**: API Keys, JWT tokens, and OAuth2 client credentials
- **Task Management**: Async task execution with status tracking
- **Monitoring**: Prometheus metrics for production observability

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   A2A Client    │────│  A2A-Minions    │────│ Minions Protocol│
│                 │    │     Server      │    │                 │
│ • Skill Request │    │ • Skill Router  │    │ • Local Model   │
│ • Document Send │    │ • Message Conv. │    │ • Remote Model  │
│ • Stream Listen │    │ • Task Manager  │    │ • Parallel Proc.│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Component Breakdown:**
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

1. **Clone and install Minions**:
   ```bash
   git clone <repository-url>
   cd minions
   pip install -e .
   ```

2. **Install A2A-specific dependencies**:
   ```bash
   cd apps/minions-a2a
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (for model providers):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
   export OLLAMA_HOST="http://localhost:11434"    # Optional
   ```

### Quick Verification

```bash
# Check dependencies
python run_server.py --skip-checks

# Run with environment checks
python run_server.py
```

## Quick Start

### 1. Start the Server

```bash
# From the apps/minions-a2a directory
python run_server.py --port 8001
```

The server will be available at `http://localhost:8001`.

### 2. Test Basic Functionality

```bash
# Test focused analysis (minion_query skill)
python tests/test_client_minion.py

# Test parallel processing (minions_query skill)
python tests/test_client_minions.py
```

### 3. Check Server Status

- Health check: `http://localhost:8001/health`
- Agent card: `http://localhost:8001/.well-known/agent.json`

## Configuration

### Server Configuration

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

## Skills and Protocol Selection

The system automatically routes requests to the appropriate protocol based on the skill ID:

### `minion_query` Skill
- **Purpose**: Focused analysis and single-document Q&A
- **Protocol**: Uses Minion (singular) for cost-efficient processing
- **Best For**: Specific questions, quick fact extraction, simple analysis

### `minions_query` Skill  
- **Purpose**: Complex parallel processing and multi-document analysis
- **Protocol**: Uses Minions (parallel) for distributed processing
- **Best For**: Large document analysis, multi-document processing, complex research

### Skill Selection Examples

```json
// Focused analysis
{
  "metadata": {
    "skill_id": "minion_query",
    "max_rounds": 2
  }
}

// Parallel processing
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

```http
GET /health                                    # Health check
GET /.well-known/agent.json                  # Public agent card
GET /agent/authenticatedExtendedCard          # Extended capabilities (auth required)
POST /oauth/token                             # OAuth2 token endpoint
GET /metrics                                  # Prometheus metrics
```

### Task Management

All task operations use JSON-RPC 2.0 over HTTP POST to `/`:

#### Send Task
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "id": "task-uuid",
    "message": {
      "role": "user",
      "parts": [
        {"kind": "text", "text": "Your question here"},
        {"kind": "text", "text": "Document content here"}
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
Use `tasks/sendSubscribe` method with `Accept: text/event-stream` header for real-time updates.

#### Get Task Status
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/get",
  "params": {"id": "task-uuid"},
  "id": "request-uuid"
}
```

#### Cancel Task
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/cancel", 
  "params": {"id": "task-uuid"},
  "id": "request-uuid"
}
```

### Message Formats

```json
// Text input
{"kind": "text", "text": "Your content"}

// File input
{
  "kind": "file",
  "file": {
    "name": "document.pdf",
    "mimeType": "application/pdf", 
    "bytes": "base64-encoded-content"
  }
}

// Data input
{
  "kind": "data",
  "data": {"key": "value", "nested": {"data": "structure"}}
}
```

## Authentication

A2A-Minions supports multiple authentication methods following the A2A protocol specification:

### Authentication Methods

1. **API Key Authentication** (Default)
   - Header: `X-API-Key: <your-api-key>`
   - Best for: Local testing, simple deployments

2. **Bearer Token Authentication** (JWT)
   - Header: `Authorization: Bearer <token>`
   - Best for: Production deployments, time-limited access

3. **OAuth2 Client Credentials Flow**
   - Endpoint: `POST /oauth/token`
   - Best for: Machine-to-machine authentication

### Quick Setup

```bash
# No authentication (testing only)
python run_server.py --no-auth

# Default authentication (generates API key)
python run_server.py

# Custom API key
python run_server.py --api-key "your-custom-api-key"
```

### Managing Authentication

Use the included CLI tools:

```bash
# API Keys
python manage_api_keys.py list
python manage_api_keys.py generate "my-client" --scopes minion:query tasks:read
python manage_api_keys.py revoke "abc12345"

# OAuth2 Clients
python manage_oauth2_clients.py list
python manage_oauth2_clients.py register "my-app" --scopes minion:query tasks:read
python manage_oauth2_clients.py revoke oauth2_xxxxx
```

### Available Scopes

- `minion:query` - Execute focused minion queries
- `minions:query` - Execute parallel minions queries  
- `tasks:read` - Read task status and results
- `tasks:write` - Create and cancel tasks

### OAuth2 Flow Example

```bash
# Get access token
curl -X POST http://localhost:8001/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=your-client-id" \
  -d "client_secret=your-client-secret" \
  -d "scope=minion:query tasks:read"

# Use token in requests
curl -H "Authorization: Bearer <token>" http://localhost:8001/agent/authenticatedExtendedCard
```

## Testing

### Running Tests

1. **Start server with test authentication**:
   ```bash
   python run_server.py --api-key "abcd"
   ```

2. **Run test suites**:
   ```bash
   # Test focused analysis
   python tests/test_client_minion.py
   
   # Test parallel processing
   python tests/test_client_minions.py
   
   # Custom server URL
   python tests/test_client_minion.py --base-url http://localhost:8001
   ```

### Test Coverage

- Health and discovery endpoints
- Basic functionality and context processing
- Document processing (PDF, JSON, multi-modal)
- Streaming responses
- Error handling and edge cases
- Task management (status, cancellation)
- Long context processing

### Custom Test Example

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
    
    metadata = {"skill_id": "minion_query", "max_rounds": 2}
    
    response = await client.send_task(message, metadata)
    task_id = response["result"]["id"]
    
    result = await client.wait_for_completion(task_id)
    print(f"Result: {result}")
    
    await client.close()

asyncio.run(custom_test())
```

## Examples

### Simple Document Q&A (Focused Analysis)

```python
message = {
    "role": "user", 
    "parts": [
        {
            "kind": "text",
            "text": "What are the key findings in this research paper?"
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

### Complex Multi-Document Analysis (Parallel Processing)

```python
message = {
    "role": "user",
    "parts": [
        {
            "kind": "text", 
            "text": "Analyze this document and extract performance metrics, categorize findings, and identify trends."
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
    "num_tasks_per_round": 3
}
```

### JSON Data Processing

```python
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
    "skill_id": "minions_query",
    "max_rounds": 2,
    "max_jobs_per_round": 4
}
```

### Streaming Response

```python
async def stream_example():
    client = A2AMinionsTestClient()
    
    # Send streaming request
    task_id = await client.send_task_streaming(message, metadata)
    
    # Real-time updates are automatically handled
    final_task = await client.wait_for_completion(task_id)
    print(f"Final result: {final_task}")
```

## Troubleshooting

### Common Issues

#### Skill Detection Problems
**Symptoms**: Wrong protocol being used, unexpected errors  
**Solution**: Ensure `skill_id` is properly specified in metadata:
```json
{"metadata": {"skill_id": "minion_query"}}  // or "minions_query"
```

#### Context Not Being Used
**Symptoms**: Model responds "I don't have access to documents"  
**Solution**: 
- Verify file uploads are base64 encoded correctly
- Check logs for context validation messages
- Ensure document content is non-empty

#### Parameter Errors
**Symptoms**: "Unexpected keyword argument" errors  
**Solution**: Verify skill and parameter compatibility:
- `minion_query`: Uses basic parameters (max_rounds, images)
- `minions_query`: Uses parallel parameters (max_jobs_per_round, num_tasks_per_round)

#### Model Provider Issues
**Symptoms**: Connection errors, authentication failures  
**Solution**: 
- Check environment variables are set correctly
- Verify model provider endpoints are accessible
- Ensure API keys have sufficient permissions

### Performance Tuning

```json
// For large documents
{
  "skill_id": "minions_query",
  "max_rounds": 2,
  "max_jobs_per_round": 8,
  "num_tasks_per_round": 4,
  "chunking_strategy": "chunk_by_section"
}

// For quick responses
{
  "skill_id": "minion_query", 
  "max_rounds": 1,
  "local_model": "llama3.2:1b",
  "remote_model": "gpt-4o-mini"
}
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Operations

### Monitoring

The server exposes Prometheus metrics at `/metrics`:

```bash
# View metrics
curl http://localhost:8000/metrics
```

**Key metrics:**
- `a2a_minions_requests_total`: Request count by method and status
- `a2a_minions_request_duration_seconds`: Request latency
- `a2a_minions_tasks_total`: Task count by skill and status
- `a2a_minions_task_duration_seconds`: Task execution time
- `a2a_minions_active_tasks`: Currently running tasks
- `a2a_minions_auth_attempts_total`: Auth attempts by method

**Prometheus configuration:**
```yaml
scrape_configs:
  - job_name: 'a2a-minions'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Production Deployment

Before deploying to production, review the comprehensive checklist:

📋 **[PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)**

**Key areas covered:**
- Environment configuration
- Security hardening
- Infrastructure requirements
- Monitoring setup
- Performance optimization
- Operational procedures

## Contributing

### Development Setup

```bash
git clone <repository-url>
cd minions/apps/minions-a2a
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate
pip install -r requirements.txt
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

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions  
- **Documentation**: Check this README and inline code documentation

---

**A2A-Minions** - Bridging Agent-to-Agent communication with the power of the Minions protocol for efficient, scalable document processing and analysis.