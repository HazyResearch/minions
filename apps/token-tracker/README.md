# Claude Code Token Tracker

A real-time web dashboard for monitoring token usage in Claude Code sessions.

## Features

- **Real-time Monitoring**: Automatically refreshes every 5 seconds to show latest token usage
- **Comprehensive Metrics**: Track input, output, and cache tokens for all Claude models
- **Message History**: View recent Claude Code responses with content previews
- **Cost Estimation**: Automatic cost calculation based on current Claude API pricing
- **Model Breakdown**: See token usage by model (Opus, Sonnet, Haiku)
- **Cache Tracking**: Monitor cache creation and read operations
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

1. **Start the Token Tracker Server**:
   ```bash
   cd apps/token-tracker
   python server.py
   ```

2. **Open the Dashboard**:
   Navigate to http://localhost:5555 in your web browser

## How It Works

The Token Tracker monitors the `~/.claude/projects/` directory for JSONL files created by Claude Code sessions. It parses these files to extract:

- Token usage (input, output, cache creation, and cache read)
- Message timestamps and content previews
- Model information (Opus, Sonnet, Haiku)
- Project context
- Request IDs for tracking

## Dashboard Components

### Metrics Cards
- **Total Tokens**: Combined input and output tokens across all models
- **Cache Tokens**: Tokens saved through caching (creation and reads)
- **Messages**: Total number of Claude assistant responses
- **Estimated Cost**: Calculated based on current Claude API pricing

### Model Usage Breakdown
Shows token usage statistics for each Claude model used:
- Input tokens per model
- Output tokens per model
- Total tokens per model

### Recent Messages Table
Displays the 20 most recent Claude Code messages with:
- Timestamp
- Project directory
- Model used (with visual badge)
- Response content preview
- Input/output token counts
- Cache status (created/read)

## API Endpoints

The server provides REST API endpoints for programmatic access:

- `GET /api/usage` - Get all token usage data
- `GET /api/usage/latest` - Get the most recent session's usage

## Cost Calculation

The dashboard estimates costs using current Claude API pricing:
- **Claude 3.5 Sonnet**: $3/1M input tokens, $15/1M output tokens
- **Claude 3.5 Haiku**: $1/1M input tokens, $5/1M output tokens
- **Claude 3 Opus**: $15/1M input tokens, $75/1M output tokens
- **Opus 4**: $15/1M input tokens, $75/1M output tokens

Note: Cached token reads significantly reduce costs!

## Requirements

- Python 3.8+
- Flask (optional - for full-featured server)
- flask-cors (optional - for full-featured server)

## Installation

The token tracker can run in two modes:

1. **Simple Mode** (no installation required):
   - Uses Python's built-in HTTP server
   - No external dependencies
   - Full functionality

2. **Flask Mode** (optional):
   ```bash
   pip install flask flask-cors
   ```
   - Same functionality with Flask framework
   - Automatically detected and used if available

## Configuration

The server automatically detects Claude Code's data directory at `~/.claude/projects/`. No configuration needed - it works out of the box with your Claude Code installation.

## Notes

- The tracker monitors Claude Code session files in `~/.claude/projects/`
- Data is updated in real-time as you use Claude Code
- The dashboard updates automatically every 5 seconds
- Shows the last 100 messages to keep performance optimal
- Each project's sessions are stored in separate JSONL files