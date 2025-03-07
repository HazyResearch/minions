#!/bin/bash

# Azure OpenAI Setup Script
# -------------------------
# This script helps set up the environment variables needed for Azure OpenAI integration with Minions.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print with color
print_green() { echo -e "${GREEN}$1${NC}"; }
print_yellow() { echo -e "${YELLOW}$1${NC}"; }
print_red() { echo -e "${RED}$1${NC}"; }

print_yellow "Azure OpenAI Setup for Minions"
print_yellow "==============================="
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    print_yellow "Found existing .env file. We'll update it with Azure OpenAI settings."
    source .env
else
    print_yellow "Creating new .env file for Azure OpenAI settings."
    touch .env
fi

# Get Azure OpenAI settings from user
echo ""
print_yellow "Please enter your Azure OpenAI settings:"
echo ""

# API Key
read -p "Azure OpenAI API Key: " AZURE_OPENAI_API_KEY
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    print_red "Error: API Key is required."
    exit 1
fi

# Endpoint
read -p "Azure OpenAI Endpoint (e.g., https://your-resource-name.openai.azure.com/): " AZURE_OPENAI_ENDPOINT
if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    print_red "Error: Endpoint is required."
    exit 1
fi

# API Version (with default)
read -p "Azure OpenAI API Version [2024-02-15-preview]: " AZURE_OPENAI_API_VERSION
AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-02-15-preview}

# Update .env file
echo "# Azure OpenAI Settings" >> .env
echo "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" >> .env
echo "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" >> .env
echo "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION" >> .env

# Set environment variables for current session
export AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY
export AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION

print_green "Azure OpenAI settings saved to .env file and set in current environment."
echo ""
print_yellow "To use these settings in a new terminal session, run:"
echo "source .env"
echo ""
print_yellow "To run Minions with Azure OpenAI, use:"
echo "streamlit run app.py"
echo ""
print_green "Done! ✓" 