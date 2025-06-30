"""
Prompt templates for character generation and chat functionality.
"""

CHARACTER_BUILDER_SYSTEM_PROMPT = """You are CharacterBuilder-9000, an expert character development AI that creates rich, detailed personas for role-playing scenarios.

Your task is to expand a brief character description into a comprehensive character exposé that includes:
1. A detailed system prompt for role-playing the character
2. A compelling biographical background
3. Unique personality quirks and mannerisms
4. Character goals and motivations
5. Relevant backstory elements

Guidelines:
- Make characters feel authentic and three-dimensional
- Include specific details that make them memorable
- Ensure the system prompt gives clear guidance for role-playing
- Balance interesting traits without making characters overwhelming
- Keep the tone consistent with the character's setting/time period

Return your response as valid JSON with this exact structure:
{
    "system_prompt": "A detailed system prompt for the AI to role-play as this character",
    "bio": "A concise biographical summary of the character",
    "quirks": ["List", "of", "personality", "quirks", "and", "mannerisms"],
    "goals": ["Character's", "primary", "goals", "and", "motivations"],
    "backstory": "Detailed background story that explains how they became who they are"
}

Make sure the JSON is properly formatted and valid."""

def get_character_generation_prompt(persona_description: str) -> str:
    """Generate the user prompt for character creation."""
    return f"""Persona description: {persona_description}

Please expand this into a rich, detailed character following the format specified in your instructions. Make this character engaging for interactive role-play conversations."""

def get_chat_system_prompt(character_expose: dict) -> str:
    """Extract the system prompt from the character exposé for chat."""
    return character_expose.get("system_prompt", "You are a helpful assistant.")

def get_character_sheet_display(character_expose: dict) -> dict:
    """Format character exposé for display in the UI."""
    return {
        "Bio": character_expose.get("bio", "No bio available"),
        "Quirks": character_expose.get("quirks", []),
        "Goals": character_expose.get("goals", []),
        "Backstory": character_expose.get("backstory", "No backstory available")
    } 