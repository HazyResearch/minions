"""
Utility functions for the Minions Story Teller app
"""

import re
import json
from typing import Dict, Any, List, Optional


def enhance_image_prompt(chapter_content: str, chapter_title: str, story_theme: str) -> str:
    """
    Create an enhanced image prompt based on chapter content and story theme.
    
    Args:
        chapter_content: The text content of the chapter
        chapter_title: The title of the chapter
        story_theme: The overall theme/idea of the story
        
    Returns:
        Enhanced image prompt suitable for FLUX.1-schnell
    """
    # Analyze chapter content for specific visual elements
    content_lower = chapter_content.lower()
    
    # Start with base prompt
    prompt_parts = ["Children's book illustration"]
    
    # Identify main character (Whiskers the cat)
    main_character = "orange tabby cat"
    if 'whiskers' in content_lower:
        prompt_parts.append(f"featuring {main_character}")
    
    # Determine scene based on chapter content
    scene_description = ""
    
    # Chapter 1 scenes - Morning, window, neighborhood
    if any(word in content_lower for word in ['woke up', 'morning', 'window', 'stretched', 'yawned']):
        scene_description = "waking up in a cozy bedroom, stretching and looking out a sunny window"
    elif any(word in content_lower for word in ['neighborhood', 'decorations', 'flags', 'outside']):
        scene_description = "looking out window at a neighborhood decorated with colorful American flags and patriotic bunting"
    elif any(word in content_lower for word in ['bella', 'rabbit', 'hopping', 'street']):
        scene_description = "meeting a friendly white rabbit on a sunny street with red, white and blue decorations"
    
    # Chapter 2 scenes - Planning, backyard, friends
    elif any(word in content_lower for word in ['planning', 'backyard', 'basket', 'decorations']):
        scene_description = "in a sunny backyard with colorful balloons and picnic tables, planning a party"
    elif any(word in content_lower for word in ['friends', 'gathering', 'max', 'dog']):
        scene_description = "gathered with animal friends including a wise old dog and playful rabbit in a backyard"
    elif any(word in content_lower for word in ['craft', 'making', 'hats', 'flags']):
        scene_description = "doing arts and crafts, making patriotic hats and decorations with friends"
    
    # Chapter 3 scenes - Celebration, sparklers, fireworks
    elif any(word in content_lower for word in ['sparkler', 'parade', 'lighting', 'twinkling']):
        scene_description = "holding sparklers that create magical twinkling lights in the evening"
    elif any(word in content_lower for word in ['fireworks', 'sky', 'night', 'burst']):
        scene_description = "watching colorful fireworks bursting in the dark night sky"
    elif any(word in content_lower for word in ['celebration', 'party', 'dancing', 'music']):
        scene_description = "celebrating at a joyful outdoor party with music and dancing"
    
    # Story time scenes
    elif any(word in content_lower for word in ['story', 'listening', 'tree', 'gathered']):
        scene_description = "sitting in a circle with friends under a shady tree listening to stories"
    
    # Playing/games scenes
    elif any(word in content_lower for word in ['playing', 'tag', 'games', 'running']):
        scene_description = "playing tag and games with friends in a colorful backyard"
    
    # Default scene
    else:
        scene_description = "in a cheerful outdoor setting with patriotic decorations"
    
    # Add the scene description
    if scene_description:
        prompt_parts.append(scene_description)
    
    # Add 4th of July theme elements if relevant
    if any(word in story_theme.lower() for word in ['july', '4th', 'patriotic', 'independence']):
        prompt_parts.append("with American flags and red, white, and blue decorations")
    
    # Add other characters mentioned
    if 'bella' in content_lower and 'rabbit' in content_lower:
        prompt_parts.append("with a friendly white rabbit companion")
    if 'max' in content_lower and 'dog' in content_lower:
        prompt_parts.append("with a wise golden retriever dog")
    
    # Style specifications - emphasize NO TEXT
    style_specs = [
        "bright vibrant colors",
        "warm sunny lighting", 
        "cartoon illustration style",
        "whimsical and friendly",
        "suitable for children ages 4-8",
        "storybook art style",
        "clean composition",
        "cheerful atmosphere",
        "NO TEXT OR WORDS visible in the image",
        "no speech bubbles or letters"
    ]
    
    # Combine all parts
    prompt = ", ".join(prompt_parts) + ", " + ", ".join(style_specs)
    
    # Ensure it's not too long (FLUX has token limits)
    if len(prompt) > 450:
        prompt = prompt[:447] + "..."
    
    return prompt


def extract_story_title(story_text: str) -> str:
    """
    Extract the story title from the generated text.
    
    Args:
        story_text: The full story text
        
    Returns:
        Extracted title or a default title
    """
    # Look for explicit title patterns
    title_patterns = [
        r'Title:\s*(.+?)(?:\n|$)',
        r'# (.+?)(?:\n|$)',
        r'## (.+?)(?:\n|$)',
        r'^(.+?)(?:\n|$)',  # First line as title
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, story_text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip()
            if title and not title.lower().startswith('chapter'):
                return title
    
    return "A Wonderful Story"


def split_into_chapters(story_text: str) -> List[Dict[str, str]]:
    """
    Split story text into chapters with titles and content.
    
    Args:
        story_text: The full story text
        
    Returns:
        List of chapter dictionaries with 'title' and 'content' keys
    """
    chapters = []
    
    # Split by chapter markers
    chapter_pattern = r'(Chapter\s+\d+[^\n]*)'
    parts = re.split(chapter_pattern, story_text, flags=re.IGNORECASE)
    
    current_chapter = None
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        # Check if this is a chapter title
        if re.match(r'Chapter\s+\d+', part, re.IGNORECASE):
            # Save previous chapter if exists
            if current_chapter and current_chapter['content']:
                chapters.append(current_chapter)
            
            # Start new chapter
            current_chapter = {
                'title': part,
                'content': ''
            }
        elif current_chapter:
            # Add content to current chapter
            current_chapter['content'] += part + '\n'
    
    # Add the last chapter
    if current_chapter and current_chapter['content']:
        chapters.append(current_chapter)
    
    # If no chapters found, create chapters from paragraphs
    if not chapters:
        paragraphs = [p.strip() for p in story_text.split('\n\n') if p.strip()]
        if paragraphs:
            # Group paragraphs into chapters (2-3 paragraphs per chapter)
            chapter_size = max(1, len(paragraphs) // 3)
            for i in range(0, len(paragraphs), chapter_size):
                chapter_paragraphs = paragraphs[i:i+chapter_size]
                chapters.append({
                    'title': f'Chapter {len(chapters) + 1}',
                    'content': '\n\n'.join(chapter_paragraphs)
                })
    
    return chapters


def clean_story_text(story_text: str) -> str:
    """
    Clean and format story text for better presentation.
    
    Args:
        story_text: Raw story text
        
    Returns:
        Cleaned and formatted story text
    """
    # Remove excessive whitespace
    story_text = re.sub(r'\n\s*\n', '\n\n', story_text)
    story_text = re.sub(r'[ \t]+', ' ', story_text)
    
    # Fix common formatting issues
    story_text = story_text.replace('\n\n\n', '\n\n')
    story_text = story_text.strip()
    
    return story_text


def validate_story_structure(story_data: Dict[str, Any]) -> bool:
    """
    Validate that the story structure is complete and properly formatted.
    
    Args:
        story_data: Story data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(story_data, dict):
        return False
    
    if 'title' not in story_data or not story_data['title']:
        return False
    
    if 'chapters' not in story_data or not story_data['chapters']:
        return False
    
    chapters = story_data['chapters']
    if not isinstance(chapters, list) or len(chapters) < 1:
        return False
    
    # Check each chapter
    for chapter in chapters:
        if not isinstance(chapter, dict):
            return False
        if 'title' not in chapter or not chapter['title']:
            return False
        if 'content' not in chapter or not chapter['content']:
            return False
    
    return True


def generate_story_metadata(story_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metadata for the story.
    
    Args:
        story_data: Story data dictionary
        
    Returns:
        Metadata dictionary
    """
    if not validate_story_structure(story_data):
        return {}
    
    chapters = story_data['chapters']
    total_words = sum(len(chapter['content'].split()) for chapter in chapters)
    
    metadata = {
        'title': story_data['title'],
        'chapter_count': len(chapters),
        'total_words': total_words,
        'estimated_reading_time': max(1, total_words // 100),  # Assume 100 words per minute for children
        'suitable_age': '4-8 years',
        'genre': 'Children\'s Picture Book'
    }
    
    return metadata


def format_story_for_display(story_data: Dict[str, Any]) -> str:
    """
    Format story data for display in the UI.
    
    Args:
        story_data: Story data dictionary
        
    Returns:
        Formatted story string
    """
    if not validate_story_structure(story_data):
        return "Invalid story structure"
    
    formatted_parts = []
    
    # Add title
    formatted_parts.append(f"# {story_data['title']}\n")
    
    # Add chapters
    for i, chapter in enumerate(story_data['chapters']):
        formatted_parts.append(f"## {chapter['title']}\n")
        formatted_parts.append(f"{chapter['content']}\n")
        
        # Add separator between chapters (except for the last one)
        if i < len(story_data['chapters']) - 1:
            formatted_parts.append("---\n")
    
    return '\n'.join(formatted_parts) 