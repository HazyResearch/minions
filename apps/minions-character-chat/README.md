# Minions Character Chat

A lightweight Streamlit app for role-playing with richly defined personas. Uses GPT-4o to generate detailed character exposés and local Gemma-3n (via Ollama) for fast, inexpensive chat conversations.

## Features

- **AI-Powered Character Generation**: Uses GPT-4o to expand simple persona descriptions into detailed character exposés
- **Local Chat**: Fast, cost-effective conversations using Gemma via Ollama
- **Preset Characters**: Curated collection of interesting personas to choose from
- **Custom Characters**: Create your own character descriptions
- **Character Upload/Export**: Import and export character JSON files for sharing
- **Character Editor**: Manually edit and customize any character's attributes
- **Blank Character Creation**: Start with a template and build characters from scratch
- **Chat History Export**: Download conversations as CSV files for analysis or archiving
- **Streaming Chat**: Real-time conversation with token-by-token streaming
- **Session Management**: Maintains conversation history and character state

## Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)
2. **Pull Gemma**: `ollama pull gemma3:4b` (or any Gemma variant)
3. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)

### Installation

```bash
# From the root minions directory
cd apps/minions-character-chat
pip install -r requirements.txt
```

### Running the App

```bash
# Start Ollama (if not already running)
ollama serve

# In another terminal, run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Basic Usage

1. **Enter API Key**: Paste your OpenAI API key in the sidebar (stored in session only)
2. **Choose Creation Method**: 
   - **Presets**: Select from curated character collection
   - **Custom**: Enter your own character description
   - **Upload**: Import a character JSON file
   - **Blank**: Create and manually edit a character template
3. **Generate/Load Character**: Create or load your character
4. **Start Chatting**: Use the chat interface to role-play with your character

### Character Management

- **Edit Character**: Click "Edit Character" in the sidebar to modify any aspect
- **Export Character**: Download your character as a JSON file for sharing
- **Import Character**: Upload character files created by others or exported previously
- **Character Templates**: Download sample character files to understand the format
- **Export Chat History**: Download your conversation as a CSV file for analysis or record-keeping

### Character File Format

Characters are stored as JSON files with the following structure:

```json
{
  "system_prompt": "How the character behaves and responds",
  "bio": "Brief character description",
  "quirks": ["Unique trait 1", "Unique trait 2"],
  "goals": ["Goal 1", "Goal 2"],
  "backstory": "Character's background and history"
}
```

### Chat History Export

The CSV export includes:
- Message numbering for conversation flow
- Speaker identification (User/Assistant)
- Full message content
- Timestamps for each exchange
- Character information header

CSV format:
```csv
Chat History with [Character Name]
Exported on: 2024-01-15 14:30:22

Message #,Speaker,Message,Timestamp
1a,User,"Hello there!",2024-01-15 14:25:10
1b,Assistant,"Greetings! How may I assist you today?",2024-01-15 14:25:12
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, can be entered in UI)
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)

### Character Presets

Edit `presets.yaml` to add your own character presets:

```yaml
characters:
  - name: "Stoic Roman Senator"
    description: "A wise Roman senator from the 1st century CE who speaks with gravitas and philosophical insight"
  - name: "Cheerful Space Mechanic"  
    description: "An optimistic engineer working on a space station, always ready with technical solutions and dad jokes"
```

## Architecture

- **Character Generation**: GPT-4o creates detailed personas from simple descriptions
- **Chat Engine**: Gemma models provide fast, contextual responses
- **UI**: Streamlit provides clean, responsive interface
- **State Management**: Session state preserves conversation history

## Troubleshooting

### Common Issues

1. **Ollama not running**: Start with `ollama serve`
2. **Model not found**: Pull a Gemma model with `ollama pull gemma3:4b`
3. **API key errors**: Check your OpenAI API key and quota
4. **Connection errors**: Verify Ollama is accessible at the configured host

### Performance Tips

- Gemma models provide a good balance of speed and quality for chat
- Character exposés are cached to avoid repeated API calls
- Use preset characters for faster startup

## Development

### Local Development

```bash
# Start Ollama
ollama serve

# Run in development mode
streamlit run app.py --server.runOnSave true
```

### Docker (Optional)

```bash
# Build and run with Docker
docker build -t minions-character-chat .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key minions-character-chat
```

## Contributing

This app is part of the larger Minions project. See the main project README for contribution guidelines. 