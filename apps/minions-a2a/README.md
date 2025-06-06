# A2A-Minions Integration

This project provides an A2A (Agent-to-Agent) protocol server that wraps the Minions protocol, enabling any A2A-compatible client to leverage Minions' cost-efficient local-cloud LLM collaboration.

## Overview

The A2A-Minions server exposes Minions functionality through two core A2A agent skills:

- **Minion Query**: Focused document Q&A using single conversation workflow (first part: question, second part: document)
- **Minions Query**: Parallel processing for complex document analysis (first part: question, second part: document)

## Architecture

```
A2A Client → A2A Protocol → A2A-Minions Server → Minions Protocol → Local/Remote LLMs
```

## Features

- **Protocol Preservation**: Core Minions functionality through A2A interface
- **Document Processing**: PDF extraction, multi-modal support (text/file/data parts)
- **Streaming Support**: Real-time updates via Server-Sent Events
- **Flexible Configuration**: Dynamic model and protocol configuration
- **Parts-Based Input**: A2A standard format with question as first part, document as second part
- **Cost Optimization**: Local-cloud LLM collaboration for efficient processing

## Quick Start

1. Install dependencies:
```bash
pip install -e .
```

2. Run the A2A server:
```bash
python -m a2a_minions.server
```

3. The server will be available at `http://localhost:8000` with the agent card at `/.well-known/agent.json`

## Usage Examples

### Minion Query (Focused Q&A)
```json
{
  "method": "tasks/send",
  "params": {
    "message": {
      "parts": [
        {"kind": "text", "text": "What are the main conclusions?"},
        {"kind": "file", "file": {"name": "document.pdf", "bytes": "..."}}
      ]
    },
    "metadata": {"skill_id": "minion_query"}
  }
}
```

### Minions Query (Parallel Processing)
```json
{
  "method": "tasks/send", 
  "params": {
    "message": {
      "parts": [
        {"kind": "text", "text": "Extract all insights and patterns"},
        {"kind": "text", "text": "Large document content..."}
      ]
    },
    "metadata": {
      "skill_id": "minions_query",
      "max_jobs_per_round": 5
    }
  }
}
```

## Testing

Run the comprehensive test suite:
```bash
cd tests
python run_tests.py
```

Or run individual skill tests:
```bash
python run_tests.py minion    # Test minion_query
python run_tests.py minions   # Test minions_query
```

## Configuration

The server can be configured via environment variables or task metadata for LLM providers, model selection, and processing parameters. 