#!/bin/bash

# Minions Setup and Run Script
# ----------------------------
# This script sets up and runs the Minions app with the following steps:
# 1. Finds or installs Python 3.11 using pyenv or conda if available
# 2. Creates a virtual environment with the correct Python version
# 3. Installs the required dependencies
# 4. Sets the necessary environment variables from .env file
# 5. Ensures Ollama is running with Flash Attention enabled
# 6. Runs the Streamlit app

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print with color
print_green() { echo -e "${GREEN}$1${NC}"; }
print_yellow() { echo -e "${YELLOW}$1${NC}"; }
print_red() { echo -e "${RED}$1${NC}"; }

# Check for Python 3.10 or 3.11
print_yellow "Checking for compatible Python version..."

# Try to find Python 3.11 or 3.10
PYTHON_CMD=""
for cmd in "python3.11" "python3.10" "python3"; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version | cut -d' ' -f2)
        major=$(echo $version | cut -d'.' -f1)
        minor=$(echo $version | cut -d'.' -f2)
        
        if [[ $major -eq 3 && ($minor -eq 10 || $minor -eq 11) ]]; then
            PYTHON_CMD=$cmd
            print_green "Found compatible Python: $cmd (version $version) ✓"
            break
        fi
    fi
done

# If no compatible Python found, check for pyenv or conda
if [[ -z "$PYTHON_CMD" ]]; then
    print_yellow "No compatible Python version found. Checking for pyenv or conda..."
    
    # Check for pyenv
    if command -v pyenv &> /dev/null; then
        print_yellow "pyenv found. Attempting to install Python 3.11..."
        pyenv install -s 3.11.0
        pyenv local 3.11.0
        PYTHON_CMD="python"
        print_green "Python 3.11.0 installed via pyenv ✓"
    # Check for conda/mamba
    elif command -v conda &> /dev/null || command -v mamba &> /dev/null; then
        CONDA_CMD="conda"
        if command -v mamba &> /dev/null; then
            CONDA_CMD="mamba"
        fi
        
        print_yellow "$CONDA_CMD found. Creating a conda environment with Python 3.11..."
        $CONDA_CMD create -y -n minions_env python=3.11
        eval "$($CONDA_CMD shell.bash hook)"
        $CONDA_CMD activate minions_env
        PYTHON_CMD="python"
        print_green "Conda environment with Python 3.11 created and activated ✓"
    else
        print_red "Error: No compatible Python version found and neither pyenv nor conda is available."
        print_yellow "Please install Python 3.10 or 3.11 and try again."
        print_yellow "You can install it from https://www.python.org/downloads/ or use a version manager like pyenv or conda."
        exit 1
    fi
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
print_green "Using Python $PYTHON_VERSION ✓"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_yellow "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    print_green "Virtual environment created ✓"
else
    print_yellow "Virtual environment already exists. Recreating with correct Python version..."
    rm -rf .venv
    $PYTHON_CMD -m venv .venv
    print_green "Virtual environment recreated ✓"
fi

# Activate virtual environment
print_yellow "Activating virtual environment..."
source .venv/bin/activate
print_green "Virtual environment activated ✓"

# Install dependencies
print_yellow "Installing dependencies..."
pip install -e .
pip install python-dotenv
print_green "Dependencies installed ✓"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    print_red "Error: Ollama is not installed."
    print_yellow "Please install Ollama from https://ollama.com/download and try again."
    exit 1
fi

print_green "Ollama is installed ✓"

# Check if llama3.2 model is available
print_yellow "Checking if llama3.2 model is available..."
if ! ollama list | grep -q "llama3.2"; then
    print_yellow "llama3.2 model not found. Pulling it now (this may take a while)..."
    ollama pull llama3.2
else
    print_green "llama3.2 model is available ✓"
fi

# Check for .env file and load environment variables
print_yellow "Checking for .env file..."
if [ -f ".env" ]; then
    print_green ".env file found. Loading environment variables ✓"
    # We don't need to export them here as the app will load them using python-dotenv
else
    print_yellow "No .env file found. You may need to set up API keys in the app."
    print_yellow "For Azure OpenAI, run ./setup_azure_openai.sh to create a .env file."
    
    # Create a minimal .env file for Ollama settings
    echo "# Minions environment settings" > .env
    echo "OLLAMA_FLASH_ATTENTION=1" >> .env
    print_green "Created minimal .env file with Ollama settings ✓"
fi

# Set Ollama environment variables
print_yellow "Setting Ollama environment variables..."
export OLLAMA_FLASH_ATTENTION=1
print_green "Ollama environment variables set ✓"

# Check if Ollama is running
print_yellow "Checking if Ollama server is running..."
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    print_yellow "Starting Ollama server with Flash Attention enabled..."
    # Start Ollama in the background
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    print_yellow "Waiting for Ollama server to start..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/version &> /dev/null; then
            break
        fi
        sleep 1
        echo -n "."
    done
    echo ""
    
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        print_red "Error: Failed to start Ollama server."
        exit 1
    fi
    
    print_green "Ollama server started ✓"
else
    print_green "Ollama server is already running ✓"
fi

# Add memory optimization flags
print_yellow "Setting memory optimization flags..."
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10
print_green "Memory optimization flags set ✓"

# Run the Streamlit app
print_yellow "Starting the Minions app..."
print_yellow "Note: If the app crashes with 'Killed: 9', try using a smaller model in the UI"
streamlit run app.py

# Cleanup
print_yellow "Cleaning up..."
deactivate
print_green "Done! ✓"