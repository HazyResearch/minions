# 🤖 Minions Tools Comparison App

A Streamlit application for comparing different approaches to using MCP (Model Context Protocol) tools with language models. This app allows you to test and compare:

1. **Direct Ollama + MCP**: Using OllamaClient with MCP tools directly
2. **Minions MCP**: Using the full SyncMinionsMCP framework with task decomposition


## 📋 Prerequisites

Before running the app, make sure you have:

1. **Ollama installed** with `llama3.2:1b` model available
   ```bash
   ollama pull llama3.2:1b
   ```

2. **MCP configuration** (`mcp.json`) in the root directory with filesystem server setup
   
3. **OpenAI API key** configured for the remote client (gpt-4o-mini)

4. **Python dependencies** installed (see requirements.txt)

## 🛠️ Installation

1. **Navigate to the app directory:**
   ```bash
   cd apps/minions-tools
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure MCP configuration exists:**
   Make sure you have a `mcp.json` file in the project root with filesystem server configuration.

## 🎯 Usage

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Configure your comparison:**
   - Enter a task in the sidebar (filesystem-related tasks work best)
   - Select which methods you want to compare
   - Click "🚀 Run Comparison"

3. **View results:**
   - Each method's results appear in separate tabs
   - Performance metrics are displayed for each method
   - A summary table compares execution times

## 📊 Method Comparison

### Direct Ollama + MCP
- **Best for**: Simple, direct tool calling scenarios
- **Pros**: Lightweight, fast, direct integration
- **Cons**: Limited coordination capabilities

### Minion Protocol
- **Best for**: Tasks requiring coordination between models
- **Pros**: Balance of performance and capability
- **Cons**: More complex than direct approach

### Minions MCP
- **Best for**: Complex tasks requiring planning and decomposition
- **Pros**: Most sophisticated approach with multi-round processing
- **Cons**: Most resource-intensive, slower execution

## 🔧 Configuration

The app uses several configuration options:

- **Local Model**: `llama3.2:1b` (configurable in the code)
- **Remote Model**: `gpt-4o-mini` (requires OpenAI API key)
- **MCP Server**: `filesystem` (configurable via mcp.json)
- **Max Rounds**: 3 (configurable per method)

## 📝 Example Tasks

Here are some example tasks that work well with the filesystem MCP server:

1. **Directory Analysis**: 
   ```
   Can you show me the directory structure of the examples folder and summarize what you find?
   ```

2. **File Search**:
   ```
   Find all Python files in the current directory and list their names.
   ```

3. **Content Analysis**:
   ```
   Read the contents of README.md and provide a summary of the main points.
   ```

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to initialize clients"**
   - Check that Ollama is running and the model is available
   - Verify MCP configuration exists and is valid
   - Ensure OpenAI API key is properly configured

2. **"Tool execution failed"**
   - Verify MCP server is running
   - Check file paths and permissions
   - Review MCP server logs

3. **Import errors**
   - Ensure you're running from the correct directory
   - Check that the minions package is properly installed
   - Verify Python path includes the project root

### Performance Tips

- Start with simpler tasks to verify setup
- Use the Direct Ollama + MCP method for quick tests
- Enable only necessary methods for faster comparisons
- Monitor system resources when running multiple methods

## 🔗 Related Documentation

- [Streamlit Documentation](https://docs.streamlit.io)
- [MCP Documentation](https://modelcontextprotocol.io)
- [Ollama Documentation](https://ollama.ai/docs)

