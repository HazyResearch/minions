"""
Minions Character Chat - A Streamlit app for role-playing with AI-generated personas.

Uses GPT-4o for character generation and local Gemma models for chat conversations.
"""

import os
import streamlit as st
from typing import Dict, List
import time

from utils import (
    initialize_session_state,
    load_character_presets,
    check_ollama_status,
    build_character_expose,
    chat_with_character,
    get_openai_api_key,
    cache_character_expose,
    get_cached_character_expose,
    reset_character_and_chat,
    validate_character_data,
    import_character_from_json,
    export_character_to_json,
    create_character_template,
    update_character_field,
    convert_chat_history_to_csv
)
from prompts import get_character_sheet_display


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Minions Character Chat",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🎭 Minions Character Chat")
    st.markdown("Role-play with richly defined AI personas")
    
    # Sidebar for configuration
    render_sidebar()
    
    # Main content area
    if st.session_state.character_expose is None:
        render_character_creation()
    elif st.session_state.get("editing_character", False):
        render_character_editor()
    else:
        render_chat_interface()


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        st.subheader("OpenAI API Key")
        api_key_input = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            placeholder="sk-...",
            help="Required for character generation"
        )
        
        if api_key_input != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = api_key_input
        
        # Show API key status
        if get_openai_api_key():
            st.success("✅ API Key configured")
        else:
            st.warning("❌ API Key required")
                
        # Ollama Status
        st.subheader("Local Model Status")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        with st.spinner("Checking Ollama..."):
            is_running, status_message = check_ollama_status(ollama_host)
        
        if is_running:
            st.success(status_message)
        else:
            st.error(status_message)
        
        
        # Character Actions
        if st.session_state.character_expose is not None:
            st.divider()
            st.subheader("Character Actions")
            
            if st.button("🔄 New Character", type="secondary"):
                reset_character_and_chat()
            
            if st.button("🧹 Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("✏️ Edit Character", type="secondary"):
                st.session_state.editing_character = True
                st.rerun()
            
            # Export character
            character_json = export_character_to_json(st.session_state.character_expose)
            st.download_button(
                label="📥 Export Character",
                data=character_json,
                file_name="character.json",
                mime="application/json",
                type="secondary"
            )
            
            # Export chat history if available
            if st.session_state.chat_history:
                character_name = st.session_state.character_expose.get("bio", "Character")[:50]  # Use first 50 chars of bio as name
                chat_csv = convert_chat_history_to_csv(st.session_state.chat_history, character_name)
                st.download_button(
                    label="💬 Export Chat as CSV",
                    data=chat_csv,
                    file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="secondary",
                    help="Download the conversation history as a CSV file"
                )
        
        # Footer
        st.markdown("---")
        st.markdown("🤖 **Powered by:**")
        st.markdown("• GPT-4o (Character Generation)")
        st.markdown("• Gemma3n (Local Chat)")
        st.markdown("**Note**: After character generation, no chat history or messages are passed to the cloud. All conversations are stored locally on your machine.")

        
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("💡 **Tip:** Export your chat as CSV to save or analyze your conversations!")


def render_character_creation():
    """Render the character creation interface."""
    st.header("🎨 Create Your Character")
    
    # Check prerequisites
    api_key = get_openai_api_key()
    ollama_running, _ = check_ollama_status()
    
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
        return
    
    if not ollama_running:
        st.error("⚠️ Ollama is not running. Please start Ollama and ensure a Gemma model is installed.")
        return
    
    # Character selection
    col1, col2 = st.columns([2, 1])
    
    #with col1:
    st.subheader("Choose a Persona")
    
    # Load presets
    presets = load_character_presets()
    
    # Character selection method
    selection_method = st.radio(
        "Selection Method",
        ["Choose from Presets", "Create Custom Character", "Upload Character File", "Create Blank Character"],
        horizontal=True
    )
    
    if selection_method == "Choose from Presets":
        if presets:
            preset_names = [preset["name"] for preset in presets]
            selected_preset = st.selectbox("Select a preset character:", preset_names)
            
            if selected_preset:
                selected_character = next(p for p in presets if p["name"] == selected_preset)
                st.info(f"**{selected_character['name']}**\n\n{selected_character['description']}")
                character_description = selected_character["description"]
            else:
                character_description = ""
        else:
            st.error("No character presets available.")
            character_description = ""
    elif selection_method == "Create Custom Character":
        character_description = st.text_area(
            "Describe your character:",
            placeholder="e.g., A wise old wizard who speaks in riddles and has a pet dragon...",
            height=100
        )
    elif selection_method == "Upload Character File":
        render_character_upload()
        character_description = ""
    else:  # Create Blank Character
        render_blank_character_creator()
        character_description = ""
  
    # Generate character button
    if character_description:
        if st.button("🎭 Generate Character", type="primary", use_container_width=True):
            generate_character(character_description)
    else:
        st.button("🎭 Generate Character", type="primary", disabled=True, use_container_width=True)
        st.info("👆 Please select or describe a character first.")


def generate_character(description: str):
    """Generate a character exposé from the description."""
    # Check if we have a cached version
    cached_expose = get_cached_character_expose(description)
    
    if cached_expose:
        st.session_state.character_expose = cached_expose
        st.success("✨ Character loaded from cache!")
        st.rerun()
        return
    
    # Generate new character
    api_key = get_openai_api_key()
    
    with st.spinner("🎨 Generating your character... This may take a moment."):
        try:
            character_expose = build_character_expose(description, api_key)
            
            # Cache and store the character
            cache_character_expose(description, character_expose)
            st.session_state.character_expose = character_expose
            
            st.success("✨ Character generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating character: {str(e)}")
            st.info("Please check your API key and try again.")


def render_chat_interface():
    """Render the chat interface with the generated character."""
    character_expose = st.session_state.character_expose
    
    # Character sheet (collapsible)
    with st.expander("📋 Character Sheet", expanded=False):
        render_character_sheet(character_expose)
    
    st.divider()
    
    # Chat history
    st.subheader("💬 Chat")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["user"])
        
        with st.chat_message("assistant"):
            st.write(chat["assistant"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        handle_chat_message(prompt)


def render_character_sheet(character_expose: Dict):
    """Render the character sheet display."""
    sheet_data = get_character_sheet_display(character_expose)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📖 Bio:**")
        st.write(sheet_data["Bio"])
        
        st.markdown("**🎯 Goals:**")
        for goal in sheet_data["Goals"]:
            st.write(f"• {goal}")
    
    with col2:
        st.markdown("**✨ Quirks:**")
        for quirk in sheet_data["Quirks"]:
            st.write(f"• {quirk}")
        
        st.markdown("**📚 Backstory:**")
        st.write(sheet_data["Backstory"])


def handle_chat_message(message: str):
    """Handle a new chat message."""
    # Display user message
    with st.chat_message("user"):
        st.write(message)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            
            for token in chat_with_character(
                message, 
                st.session_state.character_expose, 
                st.session_state.chat_history,
                ollama_host
            ):
                full_response += token
                message_placeholder.write(full_response + "▋")
            
            # Final display without cursor
            message_placeholder.write(full_response)
            
            # Save to chat history with timestamp
            st.session_state.chat_history.append({
                "user": message,
                "assistant": full_response,
                "timestamp": time.time()
            })
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")


def render_character_upload():
    """Render the character upload interface."""
    st.subheader("📁 Upload Character File")
    
    uploaded_file = st.file_uploader(
        "Choose a character JSON file",
        type=["json"],
        help="Upload a character file exported from this app or manually created"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file content
            json_content = uploaded_file.read().decode('utf-8')
            
            # Import the character
            success, character_data, message = import_character_from_json(json_content)
            
            if success:
                st.success(f"✅ {message}")
                
                # Preview the character
                st.subheader("Character Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📖 Bio:**")
                    st.write(character_data["bio"])
                    
                    st.markdown("**🎯 Goals:**")
                    for goal in character_data["goals"]:
                        st.write(f"• {goal}")
                
                with col2:
                    st.markdown("**✨ Quirks:**")
                    for quirk in character_data["quirks"]:
                        st.write(f"• {quirk}")
                    
                    st.markdown("**📚 Backstory:**")
                    st.write(character_data["backstory"])
                
                # Load character button
                if st.button("🎭 Load Character", type="primary", use_container_width=True):
                    st.session_state.character_expose = character_data
                    st.success("Character loaded successfully!")
                    st.rerun()
            else:
                st.error(f"❌ {message}")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")


def render_blank_character_creator():
    """Render interface to create a blank character for manual editing."""
    st.subheader("📝 Create Blank Character")
    st.info("This will create a blank character template that you can edit manually.")
    
    if st.button("📝 Create Blank Template", type="primary", use_container_width=True):
        blank_character = create_character_template()
        st.session_state.character_expose = blank_character
        st.session_state.editing_character = True
        st.success("Blank character created! You can now edit it.")
        st.rerun()


def render_character_editor():
    """Render the character editor interface."""
    st.header("✏️ Edit Character")
    
    if st.session_state.character_expose is None:
        st.error("No character to edit!")
        return
    
    character = st.session_state.character_expose.copy()
    
    st.subheader("Character Editor")
    st.divider()
    
    # Character editing form
    with st.form("character_editor"):
        st.subheader("🤖 System Prompt")
        character["system_prompt"] = st.text_area(
            "System Prompt",
            value=character.get("system_prompt", ""),
            height=150,
            help="This defines how the character behaves and responds"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📖 Bio")
            character["bio"] = st.text_area(
                "Bio",
                value=character.get("bio", ""),
                height=100,
                help="Brief description of the character"
            )
            
            st.subheader("🎯 Goals")
            goals_text = st.text_area(
                "Goals (one per line)",
                value="\n".join(character.get("goals", [])),
                height=100,
                help="Character's goals and motivations"
            )
            character["goals"] = [goal.strip() for goal in goals_text.split("\n") if goal.strip()]
        
        with col2:
            st.subheader("✨ Quirks")
            quirks_text = st.text_area(
                "Quirks (one per line)",
                value="\n".join(character.get("quirks", [])),
                height=100,
                help="Character's unique traits and quirks"
            )
            character["quirks"] = [quirk.strip() for quirk in quirks_text.split("\n") if quirk.strip()]
            
            st.subheader("📚 Backstory")
            character["backstory"] = st.text_area(
                "Backstory",
                value=character.get("backstory", ""),
                height=100,
                help="Character's background and history"
            )
        
        # Form submission buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.form_submit_button("💾 Update Character", type="primary", use_container_width=True):
                # Validate the character data
                is_valid, message = validate_character_data(character)
                
                if is_valid:
                    st.session_state.character_expose = character
                    st.success("✅ Character updated successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Invalid character data: {message}")
        
        with col2:
            if st.form_submit_button("💾 Save & Exit", type="secondary", use_container_width=True):
                # Validate the character data before saving
                is_valid, message = validate_character_data(character)
                
                if is_valid:
                    st.session_state.character_expose = character
                    st.session_state.editing_character = False
                    st.success("Character saved!")
                    st.rerun()
                else:
                    st.error(f"❌ Invalid character data: {message}")
        
        with col3:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.editing_character = False
                st.rerun()
    
    # Live preview
    st.divider()
    st.subheader("📋 Live Preview")
    
    with st.expander("Character Preview", expanded=True):
        if character.get("bio"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📖 Bio:**")
                st.write(character["bio"])
                
                if character.get("goals"):
                    st.markdown("**🎯 Goals:**")
                    for goal in character["goals"]:
                        st.write(f"• {goal}")
            
            with col2:
                if character.get("quirks"):
                    st.markdown("**✨ Quirks:**")
                    for quirk in character["quirks"]:
                        st.write(f"• {quirk}")
                
                if character.get("backstory"):
                    st.markdown("**📚 Backstory:**")
                    st.write(character["backstory"])
        else:
            st.info("Fill in the character details to see the preview.")


if __name__ == "__main__":
    main() 