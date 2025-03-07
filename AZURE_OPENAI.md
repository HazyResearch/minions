# Using Azure OpenAI with Minions

This guide explains how to set up and use Azure OpenAI as a provider for the Minions project.

## Prerequisites

1. An Azure account with access to Azure OpenAI Service
2. An Azure OpenAI deployment with models like `gpt-4o`, `gpt-4`, etc.
3. The Minions project installed and set up according to the main README

## Setup

### Option 1: Using the Setup Script

We've provided a setup script that will guide you through the process of configuring Azure OpenAI:

```bash
./setup_azure_openai.sh
```

This script will:
1. Ask for your Azure OpenAI API Key
2. Ask for your Azure OpenAI Endpoint URL
3. Ask for your Azure OpenAI API Version (defaults to 2024-02-15-preview)
4. Save these settings to a `.env` file
5. Set the environment variables for your current session

### Option 2: Manual Setup

If you prefer to set up the environment variables manually, you need to set the following:

```bash
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Using Azure OpenAI with Minions

1. Start the Minions app:
   ```bash
   streamlit run app.py
   ```

2. In the sidebar, select "AzureOpenAI" as the Remote Provider

3. Enter your API key if prompted (or it will use the one from environment variables)

4. Select the model deployment name that matches your Azure OpenAI deployment

5. Configure other settings as needed and run your queries

## Troubleshooting

### Common Issues

1. **"Azure OpenAI endpoint not set"**: Make sure you've set the `AZURE_OPENAI_ENDPOINT` environment variable or entered it in the setup script.

2. **Authentication errors**: Verify that your API key is correct and has access to the Azure OpenAI resource.

3. **Model not found**: Ensure that the model name you select in the UI matches exactly with a deployment name in your Azure OpenAI resource.

4. **API Version issues**: If you encounter API compatibility issues, try updating the `AZURE_OPENAI_API_VERSION` to a more recent version.

## Azure OpenAI Model Deployments

When using Azure OpenAI, the "model" parameter refers to your deployment name in the Azure OpenAI service. Make sure your deployment names match the model options in the UI:

- `gpt-4o`
- `gpt-4`
- `gpt-4-turbo`
- `gpt-35-turbo`

If your deployment names are different, you'll need to modify the code in `app.py` to match your specific deployment names.

## Example Code

Here's an example of how to use Azure OpenAI with the Minions protocol in your own code:

```python
from minions.clients.ollama import OllamaClient
from minions.clients.azure_openai import AzureOpenAIClient
from minions.minion import Minion

local_client = OllamaClient(
    model_name="llama3.2",
)

remote_client = AzureOpenAIClient(
    model_name="gpt-4o",  # This should match your deployment name
    api_key="your-api-key",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
    api_version="2024-02-15-preview",
)

# Instantiate the Minion object with both clients
minion = Minion(local_client, remote_client)

context = """
Your long context here...
"""

task = "Your task description here..."

# Execute the minion protocol
output = minion(
    task=task,
    context=[context],
    max_rounds=2
) 