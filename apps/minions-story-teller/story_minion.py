"""
Simplified Minion protocol for story generation.
This version focuses on a simple turn-taking protocol between remote and local models.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
import os
import time
from datetime import datetime

# Import the story-specific prompts
from minion_story_prompts import (
    STORY_SUPERVISOR_INITIAL_PROMPT,
    STORY_SUPERVISOR_CONTINUE_PROMPT,
    STORY_SUPERVISOR_FINAL_PROMPT,
    STORY_WORKER_SYSTEM_PROMPT
)

class Usage:
    """Simple usage tracking for story generation"""
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
    
    def __add__(self, other):
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens
        )
    
    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    # Try to find JSON in code blocks first
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    else:
        # Look for JSON-like structure
        bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
        if bracket_matches:
            json_str = bracket_matches[-1].group(0)
        else:
            json_str = text
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        # Return a default structure if parsing fails
        return {
            "decision": "continue",
            "message": text,
            "content": text
        }


class StoryMinion:
    """
    Simplified Minion for story generation with turn-taking protocol.
    """
    
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="story_logs"
    ):
        """Initialize the Story Minion.

        Args:
            local_client: Client for the local model (worker/writer)
            remote_client: Client for the remote model (supervisor/director)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
            log_dir: Directory for logging conversation history
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def generate_story(
        self,
        story_idea: str,
        target_chapters: int = 4,
        max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a story using the turn-taking protocol.
        
        Args:
            story_idea: The basic idea for the story
            target_chapters: Number of chapters to generate
            max_rounds: Override default max_rounds if provided
            
        Returns:
            Dict containing the generated story and metadata
        """
        
        print(f"\n=== STORY GENERATION STARTED ===")
        print(f"Story idea: {story_idea}")
        print(f"Target chapters: {target_chapters}")
        
        if max_rounds is None:
            max_rounds = self.max_rounds
            
        # Ensure max_rounds is at least as large as target_chapters
        # We need at least one round per chapter
        if max_rounds < target_chapters:
            original_max_rounds = max_rounds
            max_rounds = target_chapters
            print(f"🔧 Adjusted max_rounds from {original_max_rounds} to {max_rounds} to accommodate {target_chapters} chapters")
            if self.callback:
                self.callback("supervisor", f"🔧 Adjusted max rounds to {max_rounds} for {target_chapters} chapters", is_final=False)
        
        # Initialize tracking
        start_time = time.time()
        remote_usage = Usage()
        local_usage = Usage()
        conversation_log = []
        
        # Initialize the story structure
        story_structure = {
            "title": "",
            "chapters": [],
            "current_chapter": 1,
            "target_chapters": target_chapters,
            "complete": False
        }
        
        # Initial supervisor prompt to plan the story
        supervisor_messages = [{
            "role": "user",
            "content": STORY_SUPERVISOR_INITIAL_PROMPT.format(
                story_idea=story_idea,
                target_chapters=target_chapters
            )
        }]
        
        if self.callback:
            self.callback("supervisor", f"🎯 Planning story structure for '{story_idea}'...", is_final=False)
        
        print(f"🎯 Supervisor ({datetime.now().strftime('%H:%M:%S')}) planning story structure for '{story_idea}'...")
        
        # Get initial story plan from supervisor
        supervisor_response, usage = self.remote_client.chat(
            messages=supervisor_messages,
            response_format={"type": "json_object"}
        )
        remote_usage += usage
        
        supervisor_messages.append({
            "role": "assistant",
            "content": supervisor_response[0]
        })
        
        # Parse the initial plan
        try:
            initial_plan = json.loads(supervisor_response[0])
            story_structure["title"] = initial_plan.get("title", "Untitled Story")
            story_structure["outline"] = initial_plan.get("outline", [])
        except:
            initial_plan = _extract_json(supervisor_response[0])
            story_structure["title"] = initial_plan.get("title", "Untitled Story")
            story_structure["outline"] = initial_plan.get("outline", [])
        
        conversation_log.append({
            "role": "supervisor",
            "content": supervisor_response[0],
            "type": "planning"
        })
        
        if self.callback:
            self.callback("supervisor", f"📖 Story planned: '{story_structure['title']}' with {len(story_structure.get('outline', []))} chapters outlined", is_final=False)
        
        # Generate chapters through turn-taking
        for round_num in range(max_rounds):
            if story_structure["complete"]:
                break
                
            # Supervisor provides direction for next chapter
            if story_structure["current_chapter"] <= target_chapters:
                supervisor_messages.append({
                    "role": "user",
                    "content": STORY_SUPERVISOR_CONTINUE_PROMPT.format(
                        current_chapter=story_structure["current_chapter"],
                        target_chapters=target_chapters,
                        story_so_far=self._format_story_so_far(story_structure),
                        outline=story_structure.get("outline", [])
                    )
                })
            else:
                # Final round - ask supervisor to conclude
                supervisor_messages.append({
                    "role": "user",
                    "content": STORY_SUPERVISOR_FINAL_PROMPT.format(
                        story_so_far=self._format_story_so_far(story_structure)
                    )
                })
            
            if self.callback:
                self.callback("supervisor", f"🎯 Directing chapter {story_structure['current_chapter']} ({datetime.now().strftime('%H:%M:%S')})", is_final=False)
            
            print(f"🎯 Supervisor ({datetime.now().strftime('%H:%M:%S')}) directing chapter {story_structure['current_chapter']}...")
            
            # Get supervisor direction
            supervisor_response, usage = self.remote_client.chat(
                messages=supervisor_messages,
                response_format={"type": "json_object"}
            )
            remote_usage += usage
            
            supervisor_messages.append({
                "role": "assistant",
                "content": supervisor_response[0]
            })
            
            # Parse supervisor direction
            supervisor_direction = _extract_json(supervisor_response[0])
            
            conversation_log.append({
                "role": "supervisor",
                "content": supervisor_response[0],
                "type": "direction",
                "round": round_num + 1
            })
            
            # Check if supervisor wants to conclude
            if supervisor_direction.get("decision") == "conclude":
                if self.callback:
                    self.callback("supervisor", f"🏁 Story concluded after {len(story_structure['chapters'])} chapters", is_final=False)
                print(f"🏁 Supervisor decided to conclude the story after {len(story_structure['chapters'])} chapters")
                story_structure["complete"] = True
                break
            
            # Worker writes the chapter based on supervisor direction
            worker_messages = [
                {
                    "role": "system",
                    "content": STORY_WORKER_SYSTEM_PROMPT.format(
                        story_title=story_structure["title"],
                        story_idea=story_idea,
                        target_chapters=target_chapters
                    )
                },
                {
                    "role": "user",
                    "content": supervisor_direction.get("message", "Write the next chapter of the story.")
                }
            ]
            
            if self.callback:
                self.callback("worker", f"🖊️ Writing chapter {story_structure['current_chapter']} ({datetime.now().strftime('%H:%M:%S')})", is_final=False)
            
            # Print detailed log message
            print(f"🖊️  Worker ({datetime.now().strftime('%H:%M:%S')}) writing chapter {story_structure['current_chapter']}!!!")
            print(f"📝 Chapter direction: {supervisor_direction.get('message', 'Write the next chapter')[:100]}...")
            
            # Get worker's chapter content
            worker_response, usage, _ = self.local_client.chat(messages=worker_messages)
            local_usage += usage
            
            chapter_content = worker_response[0]
            
            if self.callback:
                self.callback("worker", f"✅ Chapter {story_structure['current_chapter']} written ({len(chapter_content)} chars)", is_final=False)
            
            print(f"✅ Worker completed chapter {story_structure['current_chapter']} ({len(chapter_content)} characters)")
            
            # Parse chapter title if the worker provided one
            chapter_title = f"Chapter {story_structure['current_chapter']}"
            if "# " in chapter_content[:200]:  # Check for markdown title in first 200 chars
                try:
                    first_line = chapter_content.split('\n')[0]
                    if first_line.startswith('#'):
                        chapter_title = first_line.replace('#', '').strip()
                except:
                    pass
            
            # Now ask worker to generate image prompt for this chapter
            image_prompt_messages = [
                {
                    "role": "system", 
                    "content": "You are an expert at creating image prompts for children's book illustrations. Create a detailed, specific image prompt based on the chapter content provided."
                },
                {
                    "role": "user",
                    "content": f"""Based on this chapter content, create a detailed image prompt for a children's book illustration:

CHAPTER CONTENT:
{chapter_content}

Create an image prompt that:
1. Describes the main visual scene from this specific chapter
2. Includes the characters mentioned (dragon, salamander, bird, etc.)
3. Describes the setting/location specifically mentioned
4. Uses children's book illustration style
5. Is bright, colorful, and age-appropriate
6. Does NOT include any text or words in the image

Format: Just return the image prompt as a single paragraph, no extra formatting."""
                }
            ]
            
            if self.callback:
                self.callback("worker", f"🎨 Creating image prompt for chapter {story_structure['current_chapter']} ({datetime.now().strftime('%H:%M:%S')})", is_final=False)
            
            print(f"🎨 Worker ({datetime.now().strftime('%H:%M:%S')}) creating image prompt for chapter {story_structure['current_chapter']}...")
            
            # Get image prompt from worker
            image_prompt_response, usage, _ = self.local_client.chat(messages=image_prompt_messages)
            local_usage += usage
            
            image_prompt = image_prompt_response[0].strip()
            
            if self.callback:
                self.callback("worker", f"🖼️ Image prompt created: {image_prompt[:50]}{'...' if len(image_prompt) > 50 else ''}", is_final=False)
            
            print(f"🖼️  Generated image prompt: {image_prompt[:100]}{'...' if len(image_prompt) > 100 else ''}")
            
            # Add the chapter to the story with its custom image prompt
            new_chapter = {
                "number": story_structure["current_chapter"],
                "title": chapter_title,
                "content": chapter_content,
                "image_prompt": image_prompt,
                "round": round_num + 1
            }
            
            story_structure["chapters"].append(new_chapter)
            story_structure["current_chapter"] += 1
            
            conversation_log.append({
                "role": "worker",
                "content": chapter_content,
                "type": "chapter",
                "chapter_number": new_chapter["number"],
                "round": round_num + 1
            })
            
            if self.callback:
                self.callback("worker", f"📖 Chapter {new_chapter['number']} '{chapter_title}' completed!", is_final=False)
            
            print(f"📖 Chapter {new_chapter['number']} '{chapter_title}' completed in round {round_num + 1}!")
            
            # Check if we've reached the target number of chapters
            if story_structure["current_chapter"] > target_chapters:
                story_structure["complete"] = True
        
        # Final processing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Create the final story structure
        final_story = {
            "title": story_structure["title"],
            "chapters": []
        }
        
        for chapter in story_structure["chapters"]:
            final_story["chapters"].append({
                "title": chapter["title"],
                "content": chapter["content"],
                "image_prompt": chapter["image_prompt"]
            })
        
        # Create result
        result = {
            "story": final_story,
            "metadata": {
                "total_chapters": len(story_structure["chapters"]),
                "total_time": total_time,
                "rounds_used": min(round_num + 1, max_rounds),
                "completed": story_structure["complete"]
            },
            "usage": {
                "remote": remote_usage.to_dict(),
                "local": local_usage.to_dict()
            },
            "conversation_log": conversation_log
        }
        
        # Log the session
        self._log_session(story_idea, result)
        
        # Send final completion message through callback
        if self.callback:
            self.callback("supervisor", f"🎉 Story '{final_story['title']}' completed! {len(final_story['chapters'])} chapters in {total_time:.2f}s", is_final=True)
        
        print(f"=== STORY GENERATION COMPLETED ===")
        print(f"Generated {len(final_story['chapters'])} chapters in {total_time:.2f} seconds")
        print(f"Story title: {final_story['title']}")
        print("Chapters created:")
        for i, chapter in enumerate(final_story['chapters'], 1):
            print(f"  {i}. {chapter['title']} ({len(chapter['content'])} chars)")
            print(f"     Image prompt: {chapter['image_prompt'][:60]}{'...' if len(chapter['image_prompt']) > 60 else ''}")
        
        return result
    
    def _format_story_so_far(self, story_structure: Dict[str, Any]) -> str:
        """Format the story written so far for context."""
        if not story_structure["chapters"]:
            return "No chapters written yet."
        
        formatted_parts = [f"Title: {story_structure['title']}\n"]
        
        for chapter in story_structure["chapters"]:
            formatted_parts.append(f"{chapter['title']}:\n{chapter['content']}\n")
        
        return "\n".join(formatted_parts)
    
    def _log_session(self, story_idea: str, result: Dict[str, Any]):
        """Log the story generation session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_idea = re.sub(r"[^a-zA-Z0-9]", "_", story_idea[:20])
        log_filename = f"{timestamp}_{safe_idea}_story.json"
        log_path = os.path.join(self.log_dir, log_filename)
        
        log_data = {
            "story_idea": story_idea,
            "timestamp": timestamp,
            "result": result
        }
        
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"Session logged to: {log_path}")
        except Exception as e:
            print(f"Error logging session: {e}") 