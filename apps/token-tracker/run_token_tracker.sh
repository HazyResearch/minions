#!/bin/bash

# Token Tracker Launcher Script
echo "🚀 Starting Claude Code Token Tracker..."
echo "📊 Dashboard will be available at http://localhost:5555"
echo ""

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
    PYTHON_CMD="python"
else
    # Use python3 by default
    PYTHON_CMD="python3"
fi

# Check which server to use
if $PYTHON_CMD -c "import flask; import flask_cors" &> /dev/null; then
    echo "✅ Flask and flask-cors detected - using Claude Code server"
    SERVER_FILE="claude_code_server.py"
else
    echo "ℹ️  Flask not found. Please install: pip install flask flask-cors"
    echo "   Attempting to run anyway..."
    SERVER_FILE="claude_code_server.py"
fi

# Start the server
echo "Starting server with $PYTHON_CMD..."
echo "Monitoring Claude Code logs in ~/.claude/projects/"
$PYTHON_CMD $SERVER_FILE