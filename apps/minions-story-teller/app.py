import streamlit as st
import json
import time
import base64
import requests
import sys
import os
from typing import Dict, Any, List, Optional
from io import BytesIO
import re

# Add the parent directory to the path to import minions modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.together import TogetherClient

# Import the simplified story minion
from story_minion import StoryMinion

# Import utility functions
from utils import (
    enhance_image_prompt,
    extract_story_title,
    split_into_chapters,
    clean_story_text,
    validate_story_structure,
    generate_story_metadata
)

# Configure Streamlit page
st.set_page_config(
    page_title="Minions Story Teller",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .story-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chapter-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4ecdc4;
    }
    
    .chapter-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .chapter-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #34495e;
        margin-bottom: 1rem;
    }
    
    .progress-container {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .story-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StoryTellerApp:
    def __init__(self):
        self.story_data = {}
        self.images = {}
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'story_generated' not in st.session_state:
            st.session_state.story_generated = False
        if 'story_data' not in st.session_state:
            st.session_state.story_data = {}
        if 'images' not in st.session_state:
            st.session_state.images = {}
        if 'generation_progress' not in st.session_state:
            st.session_state.generation_progress = {}
    
    def setup_clients(self):
        """Set up the AI clients based on user selection"""
        try:
            # Remote client setup
            if st.session_state.remote_client_type == "OpenAI":
                api_key = st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    st.error("OpenAI API key is required")
                    return None, None
                remote_client = OpenAIClient(
                    model_name=st.session_state.remote_model,
                    api_key=api_key,
                    temperature=0.7
                )
            elif st.session_state.remote_client_type == "Anthropic":
                api_key = st.session_state.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    st.error("Anthropic API key is required")
                    return None, None
                remote_client = AnthropicClient(
                    model_name=st.session_state.remote_model,
                    api_key=api_key,
                    temperature=0.7
                )
            else:
                st.error("Invalid remote client type")
                return None, None
            
            # Local client setup
            if st.session_state.local_client_type == "Ollama":
                local_client = OllamaClient(
                    model_name=st.session_state.local_model,
                    temperature=0.8
                )
            else:
                st.error("Invalid local client type")
                return None, None
            
            return remote_client, local_client
        
        except Exception as e:
            st.error(f"Error setting up clients: {str(e)}")
            return None, None
    
    def generate_image_with_together(self, prompt: str, together_api_key: str) -> Optional[str]:
        """Generate image using Together AI FLUX.1-schnell API"""
        try:
            print(f"🎨 Generating image with prompt: {prompt[:100]}...")
            
            headers = {
                "Authorization": f"Bearer {together_api_key}",
                "Content-Type": "application/json"
            }
            
            # Use the free FLUX.1-schnell model
            data = {
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "prompt": f"A children's book illustration of {prompt}. No text or words in image.",
                "width": 1024,
                "height": 768,
                "steps": 4,
                "n": 1
            }
            
            print(f"📡 Making request to Together AI...")
            response = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers=headers,
                json=data,
                timeout=120  # Increased timeout for image generation
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Received response from Together AI")
                
                if result.get('data') and len(result['data']) > 0:
                    image_data = result['data'][0]
                    if 'b64_json' in image_data:
                        print(f"🖼️ Image generated successfully")
                        return image_data['b64_json']
                    elif 'url' in image_data:
                        # If URL is provided instead of base64, download it
                        print(f"📥 Downloading image from URL...")
                        img_response = requests.get(image_data['url'], timeout=30)
                        if img_response.status_code == 200:
                            import base64
                            return base64.b64encode(img_response.content).decode('utf-8')
                
                try:
                    st.error("No image data found in Together AI response")
                except:
                    pass
                return None
            else:
                error_msg = f"Image generation failed: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
                
                try:
                    st.error(error_msg)
                except:
                    pass
                print(f"❌ {error_msg}")
                return None
                
        except requests.exceptions.Timeout:
            try:
                st.error("Image generation timed out. Please try again.")
            except:
                pass
            return None
        except requests.exceptions.RequestException as e:
            try:
                st.error(f"Network error during image generation: {str(e)}")
            except:
                pass
            return None
        except Exception as e:
            try:
                st.error(f"Error generating image: {str(e)}")
            except:
                pass
            print(f"❌ Image generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def story_callback(self, role: str, message: Any, is_final: bool = False):
        """Callback function to display story generation progress"""
        try:
            if 'progress_placeholder' in st.session_state and st.session_state.progress_placeholder is not None:
                with st.session_state.progress_placeholder:
                    # Handle different message types
                    if isinstance(message, str):
                        display_message = message
                    elif isinstance(message, dict) and 'content' in message:
                        display_message = message['content']
                    else:
                        display_message = str(message)
                    
                    # Truncate very long messages for UI display
                    if len(display_message) > 200:
                        display_message = display_message[:200] + "..."
                    
                    # Display the message based on role
                    if role.lower() == 'supervisor':
                        st.info(f"🎯 **Supervisor**: {display_message}")
                    elif role.lower() == 'worker':
                        st.success(f"🔨 **Worker**: {display_message}")
                    else:
                        st.write(f"📝 **{role}**: {display_message}")
                        
                    # Reduce delay to prevent UI conflicts
                    time.sleep(0.05)
        except Exception as e:
            # Silently handle callback errors to prevent app crashes
            pass
    
    def parse_story_structure(self, story_text: str, story_idea: str = "") -> Dict[str, Any]:
        """Parse the generated story text into structured chapters"""
        try:
            # Try to parse as JSON first
            if story_text.strip().startswith('{'):
                return json.loads(story_text)
            
            # Clean the story text
            story_text = clean_story_text(story_text)
            
            # Extract title
            title = extract_story_title(story_text)
            
            # Split into chapters
            chapters = split_into_chapters(story_text)
            
            # Generate enhanced image prompts for each chapter
            for chapter in chapters:
                if chapter['content']:
                    chapter['image_prompt'] = enhance_image_prompt(
                        chapter['content'],
                        chapter['title'],
                        story_idea
                    )
            
            story_data = {
                'title': title,
                'chapters': chapters
            }

            
            # Validate the structure
            if not validate_story_structure(story_data):
                st.warning("Story structure validation failed. Using fallback parsing.")
                # Fallback: create a single chapter with all content
                story_data = {
                    'title': title,
                    'chapters': [{
                        'title': 'Chapter 1',
                        'content': story_text,
                        'image_prompt': enhance_image_prompt(story_text, 'Chapter 1', story_idea)
                    }]
                }
            
            return story_data
            
        except Exception as e:
            st.error(f"Error parsing story structure: {str(e)}")
            return {
                'title': 'A Wonderful Story',
                'chapters': [{
                    'title': 'Chapter 1',
                    'content': story_text,
                    'image_prompt': f"Children's book illustration about {story_idea}. Colorful, friendly, suitable for young readers."
                }]
            }
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.markdown("### 🎛️ Configuration")
            
            # Remote client configuration
            st.markdown("#### Remote Client (Supervisor)")
            st.session_state.remote_client_type = st.selectbox(
                "Remote Client Type",
                ["OpenAI", "Anthropic"],
                key="remote_client_select"
            )
            
            if st.session_state.remote_client_type == "OpenAI":
                st.session_state.remote_model = st.selectbox(
                    "Remote Model",
                    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                    key="remote_model_select"
                )
                st.session_state.openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Enter your OpenAI API key",
                    key="openai_api_key_input"
                )
            elif st.session_state.remote_client_type == "Anthropic":
                st.session_state.remote_model = st.selectbox(
                    "Remote Model",
                    ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                    key="remote_model_select"
                )
                st.session_state.anthropic_api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    help="Enter your Anthropic API key",
                    key="anthropic_api_key_input"
                )
            
            # Together AI API key for image generation
            st.markdown("#### Image Generation")
            st.session_state.together_api_key = st.text_input(
                "Together AI API Key",
                type="password",
                help="Enter your Together AI API key for image generation",
                key="together_api_key_input"
            )
            
            # Local client configuration
            st.markdown("#### Local Client (Worker)")
            st.session_state.local_client_type = st.selectbox(
                "Local Client Type",
                ["Ollama"],
                key="local_client_select"
            )
            
            if st.session_state.local_client_type == "Ollama":
                st.session_state.local_model = st.text_input(
                    "Ollama Model",
                    value="llama3.2:3b",
                    help="Enter the Ollama model name (e.g., llama3.2:3b)",
                    key="local_model_input"
                )
            
            # Story configuration
            st.markdown("#### Story Settings")
            st.session_state.target_chapters = st.slider(
                "Number of Chapters",
                min_value=3,
                max_value=8,
                value=4,
                help="How many chapters to include in the story"
            )
            
            st.session_state.max_rounds = st.slider(
                "Max Rounds",
                min_value=1,
                max_value=8,
                value=4,
                help="Maximum rounds of conversation between supervisor and worker. Will be automatically adjusted to match number of chapters if needed."
            )
            
            st.session_state.generate_images = st.checkbox(
                "Generate Images",
                value=True,
                help="Generate AI images for each chapter"
            )
            
            # Status indicators
            st.markdown("#### Status")
            
            # Check API keys
            remote_key_ok = False
            if st.session_state.remote_client_type == "OpenAI":
                remote_key_ok = bool(st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY'))
            elif st.session_state.remote_client_type == "Anthropic":
                remote_key_ok = bool(st.session_state.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY'))
            
            together_key_ok = bool(st.session_state.get('together_api_key') or os.getenv('TOGETHER_API_KEY'))
            
            st.success("✅ Remote API Key") if remote_key_ok else st.error("❌ Remote API Key")
            st.success("✅ Together API Key") if together_key_ok else st.error("❌ Together API Key")
            
            # Try to check Ollama connection
            try:
                ollama_client = OllamaClient(model_name=st.session_state.local_model)
                st.success("✅ Ollama Connected")
            except:
                st.error("❌ Ollama Connection")
    
    def render_main_interface(self):
        """Render the main story generation interface"""
        st.markdown('<h1 class="main-header">📚 Minions Story Teller</h1>', unsafe_allow_html=True)
        
        # st.markdown("""
        # <div class="story-container">
        #     <p style="font-size: 1.2rem; text-align: center; color: #2c3e50;">
        #         Create magical children's books using AI! Enter your story idea and watch as our AI creates 
        #         a complete illustrated book with the power of the Minion protocol.
        #     </p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # Story input
        st.markdown("### 💭 Tell me your story idea (a few samples):")
        
        # Sample story ideas
        sample_stories = [
            "A book about a cat celebrating the 4th of July",
            "A story about a little robot who wants to learn how to paint",
            "A tale of a brave mouse who saves the forest",
            "A story about a dragon who's afraid of fire",
            # "A book about a young girl who discovers a magical garden",
            # "A story about friendship between a penguin and a polar bear"
        ]
        
        # Quick story idea buttons
        cols = st.columns(2)
        for i, sample in enumerate(sample_stories[:2]):
            with cols[i]:
                if st.button(f"📝 {sample}", key=f"sample_{i}"):
                    st.session_state.story_idea_input = sample
        
        cols2 = st.columns(2)
        for i, sample in enumerate(sample_stories[2:]):
            with cols2[i]:
                if st.button(f"📝 {sample}", key=f"sample_{i+3}"):
                    st.session_state.story_idea_input = sample
        
        story_idea = st.text_area(
            "",
            placeholder="Example: A book about a cat celebrating the 4th of July",
            height=100,
            key="story_idea_input"
        )
        
        # Generate button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✨ Generate Story", disabled=not story_idea.strip(), key="generate_button", use_container_width=True):
                self.generate_story(story_idea.strip())
        
        # Demo button
        st.markdown("---")
        # st.markdown("### 🎭 Try a Demo")
        # st.markdown("Want to see how it works? Try our demo story (no API keys needed):")
        
        # col1, col2, col3 = st.columns([1, 2, 1])
        # with col2:
        #     if st.button("🎪 Generate Demo Story", key="demo_button"):
        #         self.generate_demo_story()
        
        # Display progress with error handling
        try:
            if 'progress_placeholder' not in st.session_state or st.session_state.progress_placeholder is None:
                st.session_state.progress_placeholder = st.empty()
        except:
            # If we can't create progress placeholder, disable it
            st.session_state.progress_placeholder = None
        
        # Display generated story
        if st.session_state.story_generated and st.session_state.story_data:
            self.display_story()
    
    def generate_story(self, story_idea: str):
        """Generate the story using the simplified Story Minion protocol"""
        try:
            # Set up clients
            remote_client, local_client = self.setup_clients()
            if not remote_client or not local_client:
                return
            
            # Initialize progress safely
            try:
                if st.session_state.progress_placeholder:
                    with st.session_state.progress_placeholder:
                        st.info("🚀 Starting story generation...")
            except Exception as e:
                try:
                    st.info("🚀 Starting story generation...")
                except:
                    pass  # Skip if UI update fails
            
            # Create story minion
            story_minion = StoryMinion(
                remote_client=remote_client,
                local_client=local_client,
                max_rounds=st.session_state.max_rounds,
                callback=self.story_callback,
                log_dir="story_logs"
            )
            
            # Determine target chapters
            target_chapters = getattr(st.session_state, 'target_chapters', 4)
            
            try:
                if st.session_state.progress_placeholder:
                    with st.session_state.progress_placeholder:
                        st.info("🎯 Generating story with turn-taking protocol...")
            except Exception as e:
                try:
                    st.info("🎯 Generating story with turn-taking protocol...")
                except:
                    pass  # Skip if UI update fails
            
            # Generate the story using the new protocol
            # Ensure max_rounds is at least as large as target_chapters
            effective_max_rounds = max(st.session_state.max_rounds, target_chapters)
            
            # Show adjustment message if needed
            if effective_max_rounds > st.session_state.max_rounds:
                try:
                    if st.session_state.progress_placeholder:
                        with st.session_state.progress_placeholder:
                            st.info(f"🔧 Adjusted max rounds from {st.session_state.max_rounds} to {effective_max_rounds} to generate {target_chapters} chapters")
                        time.sleep(1)
                except:
                    pass
            
            result = story_minion.generate_story(
                story_idea=story_idea,
                target_chapters=target_chapters,
                max_rounds=effective_max_rounds
            )
            
            # Extract the story from the result
            story_data = result['story']
            
            # Image prompts are now generated by the worker model during story creation
            # No need to add generic prompts here
            
            st.session_state.story_data = story_data
            
            # Generate images if enabled
            if st.session_state.generate_images:
                self.generate_story_images(story_data)
            
            st.session_state.story_generated = True
            
            try:
                if st.session_state.progress_placeholder:
                    with st.session_state.progress_placeholder:
                        st.success("✅ Story generation complete!")
                        
                        # Show some metadata
                        metadata = result['metadata']
                        st.info(f"📊 Generated {metadata['total_chapters']} chapters in {metadata['total_time']:.1f} seconds using {metadata['rounds_used']} rounds")
                        
                        time.sleep(2)
                        st.session_state.progress_placeholder.empty()
                else:
                    st.success("✅ Story generation complete!")
                    metadata = result['metadata']
                    st.info(f"📊 Generated {metadata['total_chapters']} chapters in {metadata['total_time']:.1f} seconds using {metadata['rounds_used']} rounds")
            except Exception as e:
                try:
                    st.success("✅ Story generation complete!")
                except:
                    pass  # Skip if UI update fails
        
        except Exception as e:
            try:
                st.error(f"Error generating story: {str(e)}")
            except:
                pass  # Skip if error display fails
            st.session_state.story_generated = False
    
    def generate_demo_story(self):
        """Generate a demo story without requiring API keys"""
        try:
            # Create a demo story
            demo_story = {
                'title': 'Luna the Cat and the 4th of July Adventure',
                'chapters': [
                    {
                        'title': 'Chapter 1: The Mysterious Fireworks',
                        'content': 'Luna the cat lived in a cozy house with her family. On the 4th of July, she noticed colorful lights flashing outside her window. "What could those sparkly lights be?" she wondered, pressing her nose against the glass. Her curiosity got the better of her, and she decided to investigate.',
                        'image_prompt': "Children's book illustration featuring a curious orange tabby cat looking out a window at fireworks in the distance. Bright colors, friendly and whimsical, cartoon style, suitable for young children, warm and inviting, high quality digital art, storybook illustration style"
                    },
                    {
                        'title': 'Chapter 2: Meeting New Friends',
                        'content': 'As Luna ventured into the backyard, she met Benny the bunny and Sarah the squirrel. "Those are fireworks!" explained Benny excitedly. "They celebrate America\'s birthday!" Sarah added, "Would you like to watch them with us?" Luna\'s whiskers twitched with delight. She had never seen fireworks before!',
                        'image_prompt': "Children's book illustration showing a cat, bunny, and squirrel in a backyard looking up at the sky. Bright colors, friendly characters, whimsical style, suitable for young children, warm and inviting, high quality digital art, storybook illustration style"
                    },
                    {
                        'title': 'Chapter 3: The Perfect Viewing Spot',
                        'content': 'The three friends climbed up to the roof where they had the best view of the fireworks. Red, white, and blue lights danced across the sky like magical stars. "This is amazing!" purred Luna, feeling grateful for her new friends. As they watched together, Luna learned that the best part of any celebration is sharing it with friends.',
                        'image_prompt': "Children's book illustration of three animal friends on a rooftop watching spectacular fireworks in red, white, and blue. Bright colors, celebratory atmosphere, cartoon style, suitable for young children, warm and inviting, high quality digital art, storybook illustration style"
                    }
                ]
            }
            
            # Set the story data
            st.session_state.story_data = demo_story
            st.session_state.story_generated = True
            
            # Show success message
            try:
                st.success("🎉 Demo story generated! Scroll down to see Luna's adventure!")
            except:
                pass  # Skip if success message fails
            
        except Exception as e:
            try:
                st.error(f"Error generating demo story: {str(e)}")
            except:
                pass  # Skip if error display fails
    
    def generate_story_images(self, story_data: Dict[str, Any]):
        """Generate images for each chapter"""
        try:
            together_api_key = st.session_state.get('together_api_key') or os.getenv('TOGETHER_API_KEY')
            if not together_api_key:
                try:
                    st.warning("Together AI API key not found. Please add your Together AI API key in the sidebar to generate images.")
                except:
                    pass  # Skip if warning display fails
                return
            
            chapters = story_data.get('chapters', [])
            if not chapters:
                try:
                    st.error("No chapters found to generate images for.")
                except:
                    pass  # Skip if error display fails
                return
            
            # Create progress indicators with error handling
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
            except:
                # Fallback if progress indicators can't be created
                progress_bar = None
                status_text = None
            
            images = {}
            successful_generations = 0
            
            for i, chapter in enumerate(chapters):
                chapter_num = i + 1
                progress = (i + 1) / len(chapters)
                
                try:
                    if status_text:
                        status_text.info(f"🎨 Generating image {chapter_num} of {len(chapters)}...")
                    if progress_bar:
                        progress_bar.progress(progress)
                except:
                    pass  # Continue if progress update fails
                
                # Use the worker-generated image prompt directly
                if chapter.get('image_prompt'):
                    # Worker has already created a detailed, chapter-specific prompt
                    enhanced_prompt = chapter['image_prompt']
                else:
                    # Fallback for chapters without worker-generated prompts
                    content_preview = chapter.get('content', '')[:150]
                    enhanced_prompt = f"Children's book illustration showing {content_preview}. Bright vibrant colors, friendly cartoon style, whimsical illustration, digital art, suitable for children ages 4-8, storybook art style, warm and inviting, no text or words in image."
                
                # Limit prompt length
                if len(enhanced_prompt) > 500:
                    enhanced_prompt = enhanced_prompt[:500] + "..."
                
                print(f"🎨 Chapter {chapter_num} prompt: {enhanced_prompt}")
                
                # Generate the image
                image_b64 = self.generate_image_with_together(enhanced_prompt, together_api_key)
                
                if image_b64:
                    images[f"chapter_{chapter_num}"] = image_b64
                    successful_generations += 1
                    try:
                        if status_text:
                            status_text.success(f"✅ Generated image {chapter_num} of {len(chapters)}")
                    except:
                        pass
                else:
                    try:
                        if status_text:
                            status_text.error(f"❌ Failed to generate image {chapter_num}")
                    except:
                        pass
                
                # Small delay to prevent rate limiting
                time.sleep(1)
            
            # Update session state
            st.session_state.images = images
            
            # Final status with error handling
            try:
                if status_text:
                    if successful_generations > 0:
                        status_text.success(f"🎉 Successfully generated {successful_generations} out of {len(chapters)} images!")
                    else:
                        status_text.error("❌ Failed to generate any images. Please check your Together AI API key and try again.")
                
                # Clean up progress indicators safely
                time.sleep(2)
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
            except:
                pass  # Ignore cleanup errors
            
        except Exception as e:
            try:
                st.error(f"Error generating images: {str(e)}")
            except:
                pass  # Skip if error display fails
            print(f"❌ Image generation error: {e}")
            import traceback
            traceback.print_exc()
    
    def display_story(self):
        """Display the generated story with pinwheel navigation"""
        story_data = st.session_state.story_data
        images = st.session_state.images
        
        if not story_data:
            return
        
        # Generate and display metadata
        metadata = generate_story_metadata(story_data)
        
        # Display story title and metadata
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FFE4E1 0%, #FFF8DC 100%);
            padding: 2rem;
            border-radius: 25px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(255, 228, 225, 0.4);
            border: 3px solid #F0E68C;
        ">
            <h1 style="
                text-align: center; 
                color: #8B4513; 
                margin-bottom: 1rem;
                font-size: 2.5rem;
                text-shadow: 2px 2px 4px rgba(139, 69, 19, 0.3);
            ">
                📖 {story_data.get('title', 'Your Story')}
            </h1>
            <div style="
                text-align: center; 
                color: #A0522D; 
                margin-bottom: 1rem;
                font-size: 1.1rem;
                font-weight: 500;
            ">
                {metadata.get('chapter_count', 0)} chapters • {metadata.get('total_words', 0)} words • 
                {metadata.get('estimated_reading_time', 1)} min read • Ages {metadata.get('suitable_age', '4-8')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize current chapter if not set
        if 'current_chapter_index' not in st.session_state:
            st.session_state.current_chapter_index = 0
        
        chapters = story_data.get('chapters', [])
        if not chapters:
            st.error("No chapters found in the story.")
            return
        
        current_index = st.session_state.current_chapter_index
        current_chapter = chapters[current_index]
        chapter_num = current_index + 1
        
        # # Beautiful navigation controls
        # st.markdown(f"""
        # <div style="
        #     background: linear-gradient(135deg, #E6E6FA 0%, #F0E68C 100%);
        #     padding: 1.5rem;
        #     border-radius: 20px;
        #     margin-bottom: 2rem;
        #     box-shadow: 0 4px 16px rgba(230, 230, 250, 0.3);
        #     border: 2px solid #DDA0DD;
        # ">
        #     <div style="
        #         display: flex;
        #         justify-content: space-between;
        #         align-items: center;
        #         flex-wrap: wrap;
        #         gap: 1rem;
        #     ">
        #         <div style="
        #             font-size: 1.2rem;
        #             font-weight: bold;
        #             color: #8B4513;
        #             text-align: center;
        #             flex-grow: 1;
        #         ">
        #             📖 Chapter {chapter_num} of {len(chapters)}
        #         </div>
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 2, 1, 1.5])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(current_index == 0), key="prev_btn"):
                if current_index > 0:
                    st.session_state.current_chapter_index = max(0, current_index - 1)
                    st.rerun()
        
        with col2:
            # Chapter indicator dots with beautiful styling
            dots = ""
            for i in range(len(chapters)):
                if i == current_index:
                    dots += "🟡 "
                else:
                    dots += "⚪ "
            st.markdown(f"<div style='text-align: center; font-size: 1.3rem; padding: 0.5rem;'>{dots}</div>", unsafe_allow_html=True)
        
        with col3:
            # Auto-generate images button if no images
            if not images and st.session_state.get('together_api_key'):
                if st.button("🎨 Generate Images", key="gen_images_btn"):
                    try:
                        self.generate_story_images(story_data)
                        st.rerun()
                    except Exception as e:
                        try:
                            st.error(f"Error generating images: {str(e)}")
                        except:
                            pass  # Skip if error display fails
        
        with col5:
            if st.button("Next ➡️", disabled=(current_index == len(chapters) - 1), key="next_btn"):
                if current_index < len(chapters) - 1:
                    st.session_state.current_chapter_index = min(len(chapters) - 1, current_index + 1)
                    st.rerun()
        
        # No chapter banner - cleaner look
        
        # Two-column layout for image and text
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Display image if available
            if f"chapter_{chapter_num}" in images:
                try:
                    image_b64 = images[f"chapter_{chapter_num}"]
                    image_data = base64.b64decode(image_b64)
                    
                    st.image(
                        image_data,
                        caption=f"Chapter {chapter_num} Illustration",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            else:
                # Beautiful placeholder for missing image - matching text height
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #FFB6C1 0%, #FFC0CB 50%, #FFE4E1 100%);
                    height: 500px;
                    border-radius: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #8B4513;
                    font-size: 1.1rem;
                    text-align: center;
                    margin-bottom: 1rem;
                    box-shadow: 0 8px 32px rgba(255, 182, 193, 0.3);
                    border: 3px solid #FFE4E1;
                ">
                    <div>
                        🎨<br/>
                        <div style="font-size: 1.4rem; font-weight: bold; margin: 0.5rem 0;">
                            Chapter {chapter_num}
                        </div>
                        <div style="font-size: 0.9rem; color: #A0522D;">
                            Illustration will appear here
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            # Display chapter content with beautiful styling - matching image height
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #FFF8DC 0%, #FFFACD 100%);
                padding: 2.5rem;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(255, 215, 0, 0.2);
                height: 500px;
                overflow-y: auto;
                font-size: 1.15rem;
                line-height: 1.8;
                color: #2F4F4F;
                border: 3px solid #F0E68C;
                font-family: 'Georgia', serif;
                display: flex;
                flex-direction: column;
            ">
                <div style="
                    font-size: 1.3rem; 
                    font-weight: bold; 
                    color: #8B4513; 
                    margin-bottom: 1.5rem;
                    text-align: center;
                    padding-bottom: 0.5rem;
                    border-bottom: 2px solid #DEB887;
                    flex-shrink: 0;
                ">
                    {current_chapter.get('title', f'Chapter {chapter_num}')}
                </div>
                <div style="
                    text-align: justify;
                    flex-grow: 1;
                    overflow-y: auto;
                ">
                    {current_chapter.get('content', '').strip()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clean spacing
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Export options (only show on last chapter)
        if current_index == len(chapters) - 1:
            st.markdown("---")
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download story as JSON
                story_json = json.dumps(story_data, indent=2)
                st.download_button(
                    label="📄 Download Story (JSON)",
                    data=story_json,
                    file_name=f"{story_data.get('title', 'story').replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Download all images
                if images:
                    if st.button("🖼️ Download Images"):
                        st.info("Image download feature coming soon!")
            
            with col3:
                # Start over button
                if st.button("🔄 Create New Story"):
                    # Reset state
                    st.session_state.story_generated = False
                    st.session_state.story_data = {}
                    st.session_state.images = {}
                    st.session_state.current_chapter_index = 0
                    st.rerun()

def main():
    """Main application entry point"""
    app = StoryTellerApp()
    app.render_sidebar()
    app.render_main_interface()

if __name__ == "__main__":
    main() 