"""
Utility functions for character generation and chat functionality.
"""

import json
import os
from typing import Dict, Generator, List, Optional, Tuple
import yaml
import streamlit as st
from openai import OpenAI
import ollama
import csv
import io
from datetime import datetime

from prompts import CHARACTER_BUILDER_SYSTEM_PROMPT, get_character_generation_prompt


def load_character_presets() -> List[Dict[str, str]]:
    """Load character presets from YAML file."""
    try:
        with open("presets.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data.get("characters", [])
    except FileNotFoundError:
        st.error("presets.yaml not found. Please ensure it exists in the app directory.")
        return []
    except yaml.YAMLError as e:
        st.error(f"Error parsing presets.yaml: {e}")
        return []


def check_ollama_status(host: str = "http://localhost:11434") -> Tuple[bool, str]:
    """Check if Ollama is running and has the required model."""
    try:
        client = ollama.Client(host=host)
        models = client.list()
        model_names = [model['model'] for model in models['models']]
        
        # Check for gemma:3b or similar models
        gemma_models = [name for name in model_names if 'gemma3n' in name.lower() and '3' in name]
        
        if gemma_models:
            return True, f"✅ Ollama running with model: {gemma_models[0]}"
        else:
            return False, "❌ Gemma-3b model not found. Run: ollama pull gemma:3b"
            
    except Exception as e:
        return False, f"❌ Ollama not accessible: {str(e)}"


def build_character_expose(description: str, openai_api_key: str) -> Dict:
    """
    Use GPT-4o to expand a character description into a full exposé.
    
    Args:
        description: Brief character description
        openai_api_key: OpenAI API key
        
    Returns:
        Dictionary containing character exposé
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
            messages=[
                {"role": "system", "content": CHARACTER_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": get_character_generation_prompt(description)}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if it's wrapped in markdown code blocks
        if content.startswith("```json"):
            content = content[7:-3]  # Remove ```json and ```
        elif content.startswith("```"):
            content = content[3:-3]  # Remove ``` and ```
            
        character_expose = json.loads(content)
        
        # Validate required fields
        required_fields = ["system_prompt", "bio", "quirks", "goals", "backstory"]
        for field in required_fields:
            if field not in character_expose:
                raise ValueError(f"Missing required field: {field}")
                
        return character_expose
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from GPT-4o: {e}")
    except Exception as e:
        raise Exception(f"Error generating character: {e}")


def get_available_gemma_model(host: str = "http://localhost:11434") -> Optional[str]:
    """Get the first available Gemma model."""
    try:
        client = ollama.Client(host=host)
        models = client.list()
        model_names = [model['model'] for model in models['models']]
        
        # Look for gemma models in order of preference
        preferred_models = ["gemma3n:e2b"]
        
        for preferred in preferred_models:
            if preferred in model_names:
                return preferred
                
        # If no preferred model, return first gemma model found
        gemma_models = [name for name in model_names if 'gemma' in name.lower()]
        if gemma_models:
            return gemma_models[0]
            
        return None
        
    except Exception:
        return None


def chat_with_character(
    message: str, 
    character_expose: Dict, 
    chat_history: List[Dict[str, str]], 
    host: str = "http://localhost:11434"
) -> Generator[str, None, None]:
    """
    Chat with the character using Ollama streaming.
    
    Args:
        message: User message
        character_expose: Character exposé dictionary
        chat_history: Previous chat messages
        host: Ollama host URL
        
    Yields:
        Streaming response tokens
    """
    try:
        client = ollama.Client(host=host)
        model = get_available_gemma_model(host)
        
        if not model:
            yield "Error: gemma3n:e2b model not available. Please install with: ollama pull gemma3n:e2b"
            return
        
        # Build message history
        messages = [
            {"role": "system", "content": f"Character Description: {character_expose['system_prompt']}.\n\nYour job is to role-play as this character and respond to the user's messages in a way that is consistent with the character's personality and backstory."}
        ]
        
        # Add chat history
        for chat in chat_history:
            messages.append({"role": "user", "content": chat["user"]})
            messages.append({"role": "assistant", "content": chat["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Stream the response
        stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
                
    except Exception as e:
        yield f"Error: {str(e)}"


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or session state."""
    # First check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Then check session state
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    
    return None


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "character_expose" not in st.session_state:
        st.session_state.character_expose = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    if "character_cache" not in st.session_state:
        st.session_state.character_cache = {}
    
    if "editing_character" not in st.session_state:
        st.session_state.editing_character = False


def cache_character_expose(description: str, expose: Dict):
    """Cache a character exposé to avoid regeneration."""
    if "character_cache" not in st.session_state:
        st.session_state.character_cache = {}
    
    st.session_state.character_cache[description] = expose


def get_cached_character_expose(description: str) -> Optional[Dict]:
    """Get a cached character exposé."""
    if "character_cache" not in st.session_state:
        return None
    
    return st.session_state.character_cache.get(description)


def validate_character_data(character_data: Dict) -> Tuple[bool, str]:
    """
    Validate character data structure.
    
    Args:
        character_data: Dictionary containing character information
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["system_prompt", "bio", "quirks", "goals", "backstory"]
    
    # Check if it's a dictionary
    if not isinstance(character_data, dict):
        return False, "Character data must be a JSON object"
    
    # Check required fields
    missing_fields = []
    for field in required_fields:
        if field not in character_data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate field types
    if not isinstance(character_data["system_prompt"], str):
        return False, "system_prompt must be a string"
    
    if not isinstance(character_data["bio"], str):
        return False, "bio must be a string"
    
    if not isinstance(character_data["backstory"], str):
        return False, "backstory must be a string"
    
    if not isinstance(character_data["quirks"], list):
        return False, "quirks must be a list of strings"
    
    if not isinstance(character_data["goals"], list):
        return False, "goals must be a list of strings"
    
    # Check if lists contain strings
    if not all(isinstance(q, str) for q in character_data["quirks"]):
        return False, "All quirks must be strings"
    
    if not all(isinstance(g, str) for g in character_data["goals"]):
        return False, "All goals must be strings"
    
    return True, "Valid character data"


def import_character_from_json(json_content: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Import character data from JSON string.
    
    Args:
        json_content: JSON string containing character data
        
    Returns:
        Tuple of (success, character_data, message)
    """
    try:
        character_data = json.loads(json_content)
        is_valid, message = validate_character_data(character_data)
        
        if is_valid:
            return True, character_data, "Character imported successfully"
        else:
            return False, None, f"Invalid character data: {message}"
            
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, None, f"Error importing character: {str(e)}"


def export_character_to_json(character_expose: Dict) -> str:
    """
    Export character data to JSON string.
    
    Args:
        character_expose: Character exposé dictionary
        
    Returns:
        JSON string representation of character
    """
    return json.dumps(character_expose, indent=2, ensure_ascii=False)


def create_character_template() -> Dict:
    """
    Create a blank character template for editing.
    
    Returns:
        Dictionary with empty character template
    """
    return {
        "system_prompt": "",
        "bio": "",
        "quirks": [],
        "goals": [],
        "backstory": ""
    }


def update_character_field(character_expose: Dict, field: str, value) -> Dict:
    """
    Update a specific field in character exposé.
    
    Args:
        character_expose: Current character exposé
        field: Field name to update
        value: New value for the field
        
    Returns:
        Updated character exposé
    """
    updated_character = character_expose.copy()
    updated_character[field] = value
    return updated_character


def reset_character_and_chat():
    """Reset character and chat history."""
    st.session_state.character_expose = None
    st.session_state.chat_history = []
    st.rerun()


def convert_chat_history_to_csv(chat_history: List[Dict[str, str]], character_name: Optional[str] = None) -> str:
    """Convert chat history to CSV format for download."""
    if not chat_history:
        return "No chat history to export."

    # Create a CSV writer
    output = io.StringIO()
    csv_writer = csv.writer(output)

    # Write CSV header with character info if available
    if character_name:
        csv_writer.writerow([f"Chat History with {character_name}"])
        csv_writer.writerow([f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        csv_writer.writerow([])  # Empty row for spacing

    csv_writer.writerow(["Message #", "Speaker", "Message", "Timestamp"])

    # Write chat history to CSV
    for i, chat in enumerate(chat_history, 1):
        # Handle backward compatibility - old chat history might not have timestamps
        if "timestamp" in chat:
            timestamp = datetime.fromtimestamp(chat["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp = "Unknown"
            
        # Write user message
        csv_writer.writerow([f"{i}a", "User", chat["user"], timestamp])
        # Write assistant message
        csv_writer.writerow([f"{i}b", "Assistant", chat["assistant"], timestamp])

    return output.getvalue() 