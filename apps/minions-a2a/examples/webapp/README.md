# A2A-Minions Web Interface

A modern, user-friendly web interface for the A2A-Minions server that provides an intuitive way to interact with the Agent-to-Agent minions protocol.

## Features

- **🎯 Dual Processing Modes**: Support for both focused analysis (`minion_query`) and parallel processing (`minions_query`)
- **🤖 Configurable Model Providers**: Choose between Ollama (local), OpenAI, and Anthropic for both local and remote processing
- **🦙 Automatic Ollama Management**: Automatically starts and manages Ollama server when needed
- **📏 Dynamic Context Adjustment**: Automatically calculates and sets optimal context window size for Ollama based on input length
- **📄 Multi-format Input**: Text, file uploads (PDF, TXT, JSON, CSV, Markdown), and structured JSON data
- **🔄 Real-time Streaming**: Live updates and streaming responses via WebSockets
- **📊 Task Management**: Monitor task status, view results, track progress, and repeat previous queries
- **🔧 Developer Debug Panel**: Real-time logging of all API requests and responses for debugging
- **🎨 Modern UI**: Clean, responsive interface with Tailwind CSS
- **⚡ One-click Launch**: Single script to start both A2A server and web interface

## Quick Start

### 1. Installation

First, make sure you have the main minions repository installed:

```bash
cd minions
pip install -e .
```

Then install the webapp dependencies:

```bash
cd apps/minions-a2a/examples/webapp
pip install -r requirements.txt
```

### 2. Launch the Application

Use the simple launcher script:

```bash
python run_webapp.py
```

This will start both:
- A2A server on `http://localhost:8001`
- Web interface on `http://localhost:5000`

### 3. Open in Browser

Navigate to `http://localhost:5000` and start querying!

## Usage Guide

### Processing Types

**Focused Analysis (`minion_query`)**
- Best for: Quick questions, simple analysis, fact extraction
- Uses: Single model instance for cost-efficient processing
- Example: "What is the main argument in this document?"

**Parallel Processing (`minions_query`)**
- Best for: Complex analysis, multi-document processing, research tasks
- Uses: Multiple parallel model instances for distributed processing
- Example: "Analyze this document, extract key metrics, categorize findings, and identify trends"

### Input Methods

#### 1. Text Context
- Paste any text content directly
- Good for: Articles, documents, research papers
- Example: Copy-paste a research paper and ask questions about it

#### 2. File Upload
- **Supported formats**: PDF, TXT, JSON, CSV, Markdown
- **Max size**: 50MB
- **Use case**: Upload documents for analysis

#### 3. JSON Data
- Provide structured data for analysis
- Example:
```json
{
  "sales_data": {
    "Q1": {"revenue": 100000, "units": 450},
    "Q2": {"revenue": 120000, "units": 520}
  }
}
```

### Advanced Options

- **Max Rounds**: Control how many processing rounds to use (1-10)
- **Streaming**: Enable real-time response streaming
- **Parallel Settings** (for `minions_query`):
  - Jobs per Round: Number of parallel jobs (1-100)
  - Tasks per Round: Number of tasks per round (1-10)
  - Samples per Task: Samples per task (1-5)

## Example Queries

### Simple Q&A
```
Question: What are the key benefits of renewable energy?
Processing: Focused Analysis
```

### Document Analysis
```
Question: Summarize the main findings and recommendations
Context: [Upload a research paper PDF]
Processing: Focused Analysis
Max Rounds: 2
```

### Complex Data Analysis
```
Question: Analyze quarterly performance and identify trends
JSON Data: {"quarterly_sales": {...}}
Processing: Parallel Processing
Max Rounds: 3
Jobs per Round: 5
```

### Multi-document Research
```
Question: Compare these documents and extract common themes
Context: [Multiple documents pasted or uploaded]
Processing: Parallel Processing
Streaming: Enabled
```

### Large Document Analysis with Dynamic Context
```
Question: Analyze this 50-page research paper and extract key insights
File: [Upload large PDF document]
Local Provider: Ollama
Local Model: llama3.2
Processing: Focused Analysis
Note: Context window automatically adjusted to 65K tokens
```

## Interface Overview

### Main Dashboard
- **Server Status**: Real-time health indicator
- **Query Form**: Input fields and options
- **Results Panel**: Live task updates and results

### Task Management
- **Real-time Updates**: See processing progress as it happens
- **Task History**: View all submitted tasks
- **Detailed Results**: Click any task to see full details
- **Status Indicators**: 
  - 🟡 Running
  - 🟢 Completed  
  - 🔴 Failed

### Features in Detail

#### Streaming Responses
When enabled, you'll see live updates as the AI processes your query:
- Model interactions
- Intermediate results
- Progress indicators
- Final answers

#### File Processing
The webapp automatically handles file encoding and MIME type detection:
- PDFs are extracted and processed
- JSON files are parsed and validated
- Text files are loaded with proper encoding

#### Task Management Features
- **Repeat Query Button**: Green redo button next to each task to instantly populate the form with previous settings
- **Task Details Modal**: Click the eye icon to view complete task information, history, and results
- **Real-time Status**: Live updates showing elapsed time, poll count, and completion status

#### Developer Debug Panel
- **API Request Logging**: All requests and responses to the A2A server are automatically logged
- **Color-coded Entries**: Green (success), yellow (warning), red (error), blue (info)
- **Expandable Data**: Click "View Data" to see complete request/response JSON
- **Auto-cleanup**: Keeps only the last 50 entries to prevent memory issues
- **Collapsible Interface**: Toggle visibility with the chevron button

## Configuration

### Command Line Options

```bash
python run_webapp.py [options]

Options:
  --a2a-port PORT       A2A server port (default: 8001)
  --web-host HOST       Web server host (default: 127.0.0.1)
  --web-port PORT       Web server port (default: 5000)
  --api-key KEY         API key for authentication (default: abcd)
  --skip-deps           Skip dependency checks
  --no-ollama           Don't auto-start Ollama server
```

### Environment Variables

Set these for the A2A server:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
export OLLAMA_HOST="http://localhost:11434"    # Optional
```

### Ollama Setup (Optional)

The webapp can automatically manage Ollama for local model processing:

1. **Install Ollama** (if you want local models):
   ```bash
   # Visit https://ollama.ai/download for installation instructions
   # Or use package managers:
   
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull models** (recommended):
   ```bash
   ollama pull llama3.2
   ollama pull llama3.2:1b  # Smaller, faster model
   ```

3. **Auto-start behavior**:
   - By default, the webapp will automatically start Ollama server
   - If Ollama is not installed, it will show a warning but continue
   - Use `--no-ollama` flag to disable auto-start
   - You can manually start Ollama with `ollama serve`

4. **Dynamic Context Adjustment**:
   - When using Ollama, the webapp automatically calculates optimal context window size
   - Based on total input length (query + context + files + JSON data)
   - Available context sizes: 2K, 4K, 8K, 16K, 32K, 64K, 131K tokens
   - Includes 8K token padding for model responses
   - Real-time preview shows estimated context size as you type

## Troubleshooting

### Common Issues

**"Server Offline" Error**
- Check if the A2A server is running on port 8001
- Verify your API keys are set correctly
- Try restarting both servers

**Unicode/Emoji Errors on Windows**
- The webapp automatically handles UTF-8 encoding
- If issues persist, set environment variable: `set PYTHONIOENCODING=utf-8`

**Minion (Focused Analysis) Fails**
- This is often due to Unicode encoding issues on Windows
- The webapp now handles this automatically with UTF-8 encoding
- Check the A2A server logs for specific error details

**Task Stuck in "Running"**
- The webapp now has intelligent completion detection
- Tasks timeout after 5 minutes of polling
- Check browser console (F12) for detailed status logs
- Verify model providers are accessible

**File Upload Failed**
- Ensure file is under 50MB
- Check that file format is supported
- Verify file is not corrupted

**Streaming Not Working**
- Ensure WebSocket connections are allowed
- Check firewall settings
- Try disabling streaming and using regular mode

**Ollama Issues**
- If Ollama fails to start automatically, install it manually from https://ollama.ai
- Use `--no-ollama` flag to disable auto-start
- Manually start with `ollama serve` if needed
- Pull models first: `ollama pull llama3.2`

**Context Size Issues**
- Large documents automatically trigger higher context windows (up to 131K tokens)
- If you get "context length exceeded" errors, the input may be too large
- Try splitting very large documents into smaller chunks
- Monitor the context size preview when using Ollama providers
- Check browser console for context calculation logs

### Debug Mode

Run with debug mode for detailed logging:

```bash
python app.py --debug
```

### Health Checks

The webapp provides health check endpoints:
- Web server: `http://localhost:5000/health`
- A2A server: `http://localhost:8001/health`

## API Reference

The webapp exposes these endpoints:

### Web Interface Endpoints
- `GET /` - Main interface
- `GET /health` - Health check
- `GET /agent-card` - A2A agent information
- `POST /send-task` - Submit a task
- `GET /task-status/<id>` - Get task status

### WebSocket Events
- `task_update` - Real-time task updates
- `task_error` - Task error notifications

## Development

### Project Structure
```
webapp/
├── app.py              # Main Flask application
├── run_webapp.py       # Launcher script
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main interface template
└── README.md          # This file
```

### Extending the Interface

The webapp is built with:
- **Backend**: Flask + Flask-SocketIO
- **Frontend**: HTML5 + Tailwind CSS + Vanilla JavaScript
- **Communication**: HTTP + WebSockets

To add new features:
1. Add routes in `app.py`
2. Update the frontend in `templates/index.html`
3. Add any new dependencies to `requirements.txt`

## License

This webapp is part of the A2A-Minions project and follows the same license terms.

## Support

For issues and questions:
- Check the main A2A-Minions README
- Review the troubleshooting section above
- Submit GitHub issues for bugs or feature requests

---

**A2A-Minions Web Interface** - Making Agent-to-Agent communication accessible through a beautiful web interface. 🚀 