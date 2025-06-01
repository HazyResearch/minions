"""
Minions Secure Protocol Streamlit App

A Streamlit web interface for running the Minions secure protocol with local Ollama models.
This app allows users to:
- Choose a local Ollama model
- Set up secure communication with a supervisor
- Run secure minion workloads with streaming intermediate steps
- Upload files (images, PDFs, folders) for processing

Run with:
    streamlit run minions_secure_app.py
"""

import streamlit as st
from streamlit_theme import st_theme
import os
import tempfile
import uuid
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
import time
import threading
from typing import Optional, Dict, Any

from secure.minions_secure import SecureMinionProtocol
from minions.clients import OllamaClient

# System prompt for the secure protocol
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant working in a secure environment. 
You have access to context documents and can analyze them to answer questions accurately."""

# ── Helper Functions ────────────────────────────────────────────────────────

def do_rerun():
    """Streamlit renamed `experimental_rerun` → `rerun` in v1.27."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def is_mobile():
    """Check if we're in mobile mode (based on session state toggle)."""
    return st.session_state.get("mobile_mode", False)

def is_dark_mode():
    """Check if we're in dark mode based on theme."""
    theme = st_theme()
    if theme and "base" in theme:
        if theme["base"] == "dark":
            return True
    return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory and return the path."""
    temp_dir = Path(tempfile.gettempdir()) / "minions_secure_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = temp_dir / unique_filename
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    try:
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found: {pdf_path}")
            return None

        pdf_content = ""
        with fitz.open(pdf_path) as doc:
            num_pages = len(doc)
            st.info(f"Processing PDF with {num_pages} pages...")
            
            for page_num in range(num_pages):
                page = doc[page_num]
                pdf_content += page.get_text()
                if page_num < num_pages - 1:
                    pdf_content += "\n\n"
        
        return pdf_content
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_available_ollama_models():
    """Get list of available Ollama models."""
    try:
        return OllamaClient.get_available_models()
    except Exception as e:
        st.error(f"Failed to get Ollama models: {e}")
        return []

# ── Page Configuration ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Minions Secure Protocol", 
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
/* Center elements */
.stCheckbox {
    display: flex;
    justify-content: center;
}
.stCheckbox > label {
    display: flex;
    justify-content: center;
    width: 100%;
}

div[data-testid="column"] {
    display: flex;
    justify-content: center;
}

.stButton button {
    display: flex;
    align-items: center;
    justify-content: center;
}

h1, .subtitle {
    text-align: center;
}

hr {
    margin-left: auto;
    margin-right: auto;
}

/* Status indicators */
.status-box {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.status-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}

.status-info {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}

.status-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
}

.status-error {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────

# Choose image based on theme
dark_mode = is_dark_mode()
if dark_mode:
    image_path = "assets/minions_logo_no_background.png"
else:
    image_path = "assets/minions_logo_light.png"

# Display logo if it exists
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)

st.markdown("<hr style='width: 100%;'>", unsafe_allow_html=True)

st.title("🔒 Minions Secure Protocol")
st.markdown(
    "<p class='subtitle' style='font-size: 20px; color: #888;'>"
    "End-to-end encrypted local-cloud inference: connect local LLMs to cloud LLMs running in secure NVIDIA GPU enclaves."
    "</p>",
    unsafe_allow_html=True,
)

# ── Configuration Section ────────────────────────────────────────────────────

with st.expander("🔧 Configuration", expanded=True):
    
    # Mobile mode toggle
    mobile_mode = st.checkbox(
        "📱 Mobile mode",
        value=st.session_state.get("mobile_mode", False),
        help="Enable for better layout on small screens"
    )
    st.session_state.mobile_mode = mobile_mode
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ Local Model Configuration")
        
        # Get available Ollama models
        available_models = get_available_ollama_models()
        
        if not available_models:
            st.error("No Ollama models found. Please install Ollama and pull some models.")
            st.stop()
        
        # Model selection
        default_models = ["llama3.2", "llama3.1:8b", "qwen2.5:3b", "qwen2.5:7b"]
        recommended_models = [m for m in default_models if m in available_models]
        
        if recommended_models:
            default_model = recommended_models[0]
        else:
            default_model = available_models[0]
        
        selected_model = st.selectbox(
            "Select Local Model",
            options=available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            help="Choose the local Ollama model for processing"
        )
        
        # Model parameters
        local_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Controls randomness in model responses"
        )
        
        local_max_tokens = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=8192,
            value=2048,
            step=256,
            help="Maximum tokens for local model responses"
        )
        
        num_ctx = st.number_input(
            "Context Window",
            min_value=2048,
            max_value=131072,
            value=4096,
            step=1024,
            help="Context window size for the local model"
        )
    
    with col2:
        st.subheader("🌐 Supervisor Configuration")
        
        # Advanced settings toggle
        show_advanced = st.checkbox("Show Advanced Settings", value=False)
        
        if show_advanced:
            supervisor_url = st.text_input(
                "Supervisor URL",
                value=st.session_state.get("supervisor_url", "http://20.57.33.122:5056"),
                help="URL of the secure supervisor server"
            )
        else:
            supervisor_url = st.session_state.get("supervisor_url", "http://20.57.33.122:5056")
        
        # System prompt
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            height=100,
            help="System prompt for the worker model"
        )

# ── Protocol Management ────────────────────────────────────────────────────

st.subheader("🔗 Protocol Management")

# Protocol parameters
max_rounds = st.number_input(
    "Max Rounds",
    min_value=1,
    max_value=10,
    value=3,
    help="Maximum number of conversation rounds between supervisor and worker"
)

# Initialize session state
if "protocol" not in st.session_state:
    st.session_state.protocol = None
if "protocol_status" not in st.session_state:
    st.session_state.protocol_status = "disconnected"
if "session_info" not in st.session_state:
    st.session_state.session_info = {}
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Status display
status_colors = {
    "disconnected": "status-error",
    "connecting": "status-warning", 
    "connected": "status-success",
    "running": "status-info"
}

status_messages = {
    "disconnected": "❌ Not connected to supervisor",
    "connecting": "🔄 Connecting to supervisor...",
    "connected": "✅ Connected and ready",
    "running": "🔄 Running secure protocol..."
}

st.markdown(
    f'<div class="status-box {status_colors[st.session_state.protocol_status]}">'
    f'{status_messages[st.session_state.protocol_status]}'
    f'</div>',
    unsafe_allow_html=True
)

# Connection buttons
if is_mobile():
    button_cols = st.columns([1, 1, 1])
    connect_btn = button_cols[0].button("🔌 Connect", use_container_width=True)
    disconnect_btn = button_cols[1].button("❌ Disconnect", use_container_width=True)
    clear_btn = button_cols[2].button("🗑️ Clear", use_container_width=True)
else:
    button_cols = st.columns([1, 1, 1])
    connect_btn = button_cols[0].button("🔌 Connect to Supervisor", use_container_width=True)
    disconnect_btn = button_cols[1].button("❌ Disconnect", use_container_width=True)
    clear_btn = button_cols[2].button("🗑️ Clear Session", use_container_width=True)

# Handle button clicks
if connect_btn:
    try:
        st.session_state.protocol_status = "connecting"
        
        # Show connecting status
        with st.spinner("Connecting to supervisor..."):
            # Initialize local client
            local_client = OllamaClient(
                model_name=selected_model,
                temperature=local_temperature,
                max_tokens=local_max_tokens,
                num_ctx=num_ctx
            )
            
            # Initialize protocol
            protocol = SecureMinionProtocol(
                supervisor_url=supervisor_url,
                local_client=local_client,
                max_rounds=max_rounds,
                system_prompt=system_prompt
            )
            
            # Initialize secure session
            session_info = protocol.initialize_secure_session()
            
            st.session_state.protocol = protocol
            st.session_state.protocol_status = "connected"
            st.session_state.session_info = session_info
            st.session_state.supervisor_url = supervisor_url
            st.session_state.system_prompt = system_prompt
        
        st.success(f"✅ Connected! Session ID: {protocol.session_id}")
        do_rerun()
        
    except Exception as e:
        st.session_state.protocol_status = "disconnected"
        st.error(f"❌ Connection failed: {str(e)}")
        st.error(f"Debug info: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

if disconnect_btn and st.session_state.protocol:
    try:
        st.session_state.protocol.end_session()
        st.session_state.protocol = None
        st.session_state.protocol_status = "disconnected"
        st.session_state.session_info = {}
        st.success("✅ Disconnected successfully")
        do_rerun()
    except Exception as e:
        st.error(f"❌ Error during disconnect: {str(e)}")

if clear_btn:
    st.session_state.protocol = None
    st.session_state.protocol_status = "disconnected"
    st.session_state.session_info = {}
    if "task_results" in st.session_state:
        del st.session_state.task_results
    if "conversation_log" in st.session_state:
        del st.session_state.conversation_log
    st.success("✅ Session cleared")
    do_rerun()

# ── Task Execution ────────────────────────────────────────────────────────

if st.session_state.protocol_status == "connected":
    st.subheader("🎯 Task Execution")
    
    # Task input
    task = st.text_area(
        "Enter your task",
        placeholder="What would you like the secure protocol to help you with?",
        height=100
    )
    
    # Context input
    context_input = st.text_area(
        "Additional Context (optional)",
        placeholder="Provide any additional context for the task...",
        height=80
    )
    
    # File upload section
    st.subheader("📎 File Attachments")
    
    # Initialize attachment state
    if "show_attachment" not in st.session_state:
        st.session_state.show_attachment = False
    if "attachment_type" not in st.session_state:
        st.session_state.attachment_type = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    
    # Attachment type selection
    attachment_type = st.radio(
        "Select attachment type",
        options=["None", "Image", "PDF Document", "Folder"],
        horizontal=True
    )
    
    image_path = None
    pdf_path = None
    folder_path = None
    
    if attachment_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "gif"],
            help="Upload an image to include with your task"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                image_path = save_uploaded_file(uploaded_file)
                st.success(f"✅ Image uploaded: {uploaded_file.name}")
                try:
                    st.image(image_path, caption="Uploaded image", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display image preview: {str(e)}")
    
    elif attachment_type == "PDF Document":
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF document to include as context"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                pdf_path = save_uploaded_file(uploaded_file)
                pdf_content = extract_text_from_pdf(pdf_path)
                if pdf_content:
                    st.success(f"✅ PDF processed: {uploaded_file.name} ({len(pdf_content)} characters)")
                else:
                    st.error("❌ Failed to extract text from PDF")
                    pdf_path = None
    
    elif attachment_type == "Folder":
        uploaded_files = st.file_uploader(
            "Upload multiple files",
            type=["txt", "pdf", "png", "jpg", "jpeg", "gif"],
            accept_multiple_files=True,
            help="Upload multiple files to create a folder context"
        )
        
        if uploaded_files:
            with st.spinner("Processing folder..."):
                temp_dir = Path(tempfile.gettempdir()) / f"minions_secure_folder_{uuid.uuid4().hex}"
                temp_dir.mkdir(exist_ok=True)
                
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    progress = int((i / len(uploaded_files)) * 100)
                    progress_bar.progress(progress)
                    
                    file_path = temp_dir / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                progress_bar.progress(100)
                folder_path = str(temp_dir)
                
                # Count file types
                txt_count = len(list(temp_dir.glob("*.txt")))
                pdf_count = len(list(temp_dir.glob("*.pdf")))
                img_count = sum(len(list(temp_dir.glob(f"*.{ext}"))) 
                               for ext in ["jpg", "jpeg", "png", "gif"])
                
                st.success(f"✅ Folder created: {txt_count} text files, {pdf_count} PDFs, {img_count} images")
    
    # Execute task button
    if st.button("🚀 Execute Secure Task", type="primary", use_container_width=True):
        if not task.strip():
            st.error("❌ Please enter a task")
        elif st.session_state.protocol is None:
            st.error("❌ Please connect to supervisor first")
        else:
            # Prepare context
            context_list = []
            if context_input.strip():
                context_list.append(context_input.strip())
            
            # Clear previous conversation history for new task
            st.session_state.conversation_history = []
            
            # Add user message to conversation history
            user_message = {
                "role": "user",
                "content": task,
                "attachments": []
            }
            
            # Add attachment info to user message
            if image_path:
                user_message["attachments"].append({"type": "image", "path": image_path})
            if pdf_path:
                user_message["attachments"].append({"type": "pdf", "path": pdf_path})
            if folder_path:
                user_message["attachments"].append({"type": "folder", "path": folder_path})
            
            st.session_state.conversation_history.append(user_message)
            
            # Create chat container for conversation display
            st.subheader("Secure Local-Cloud Inference Workload")
            chat_container = st.container()
            
            # Display user message
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**Task:** {task}")
                    
                    # Show attachments
                    if user_message["attachments"]:
                        st.markdown("**Attachments:**")
                        for attachment in user_message["attachments"]:
                            if attachment["type"] == "image":
                                st.markdown("📷 Image file")
                                try:
                                    st.image(attachment["path"], caption="Uploaded image", use_container_width=True)
                                except:
                                    st.markdown("*Image preview unavailable*")
                            elif attachment["type"] == "pdf":
                                st.markdown("📄 PDF document")
                            elif attachment["type"] == "folder":
                                st.markdown("📁 Folder with multiple files")
                    
                    if context_input.strip():
                        st.markdown(f"**Additional Context:** {context_input}")
            
            # Initialize placeholders for streaming messages
            placeholders = {"supervisor": None, "worker": None}
            
            def message_callback(role: str, message: Optional[Dict[str, Any]], is_final: bool = True):
                """Callback function to handle streaming messages in chat format"""
                
                with chat_container:
                    if role == "supervisor":
                        if not is_final:
                            # Create placeholder for supervisor thinking
                            placeholders["supervisor"] = st.empty()
                            with placeholders["supervisor"]:
                                with st.chat_message("assistant", avatar="🌐"):
                                    st.markdown("**Remote Supervisor:** *Analyzing and coordinating...*")
                        else:
                            # Clear placeholder and show actual message
                            if placeholders["supervisor"]:
                                placeholders["supervisor"].empty()
                                placeholders["supervisor"] = None
                            
                            if message and "content" in message:
                                with st.chat_message("assistant", avatar="🌐"):
                                    st.markdown("**Remote Supervisor:**")
                                    
                                    # Try to parse and display JSON content nicely
                                    try:
                                        import json
                                        content = json.loads(message["content"])
                                        if "message" in content:
                                            st.markdown(content["message"])
                                        if "decision" in content:
                                            if content["decision"] == "request_additional_info":
                                                st.info("🔄 Requesting additional information from local worker")
                                            elif content["decision"] == "provide_final_answer":
                                                st.success("✅ Providing final answer")
                                    except:
                                        st.markdown(message["content"])
                    
                    elif role == "worker":
                        if not is_final:
                            # Create placeholder for worker thinking
                            placeholders["worker"] = st.empty()
                            with placeholders["worker"]:
                                with st.chat_message("assistant", avatar="🖥️"):
                                    st.markdown("**Local Worker:** *Processing locally...*")
                        else:
                            # Clear placeholder and show actual message
                            if placeholders["worker"]:
                                placeholders["worker"].empty()
                                placeholders["worker"] = None
                            
                            if message and "content" in message:
                                with st.chat_message("assistant", avatar="🖥️"):
                                    st.markdown("**Local Worker:**")
                                    st.markdown(message["content"])
            
            try:
                st.session_state.protocol_status = "running"
                
                # Set callback for the protocol
                st.session_state.protocol.callback = message_callback
                
                # Execute the protocol
                with st.spinner("🔄 Executing secure protocol..."):
                    result = st.session_state.protocol(
                        task=task,
                        context=context_list,
                        image_path=image_path,
                        pdf_path=pdf_path,
                        folder_path=folder_path
                    )
                
                st.session_state.protocol_status = "connected"
                st.session_state.task_results = result
                
                # Display final answer in chat format
                with chat_container:
                    with st.chat_message("assistant", avatar="🎯"):
                        st.markdown("**Final Answer:**")
                        st.markdown(result["final_answer"])
                        
                        # Show performance metrics
                        if "timing" in result:
                            timing = result["timing"]
                            st.markdown("---")
                            st.markdown("**Performance Metrics:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Time", f"{timing.get('total_time', 0):.2f}s")
                            with col2:
                                st.metric("Rounds", len(timing.get('rounds', [])))
                            with col3:
                                if 'setup' in timing and 'attestation_exchange' in timing['setup']:
                                    st.metric("Setup Time", f"{timing['setup']['attestation_exchange']:.2f}s")
                
                # Show success message and log info
                st.success("✅ Task completed successfully!")
                if "log_file" in result:
                    st.info(f"📁 Detailed log saved to: {result['log_file']}")
                
            except Exception as e:
                st.session_state.protocol_status = "connected"
                st.error(f"❌ Task execution failed: {str(e)}")
                
                # Clear any remaining placeholders
                for placeholder in placeholders.values():
                    if placeholder:
                        placeholder.empty()

# ── Session Information ────────────────────────────────────────────────────

if st.session_state.session_info:
    with st.expander("ℹ️ Session Information"):
        info = st.session_state.session_info
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Session ID:** `{info.get('session_id', 'Unknown')}`")
            if 'attestation_time' in info:
                st.markdown(f"**Attestation Time:** {info['attestation_time']:.3f}s")
        
        with col2:
            if 'key_exchange_time' in info:
                st.markdown(f"**Key Exchange Time:** {info['key_exchange_time']:.3f}s")
            st.markdown(f"**Status:** {st.session_state.protocol_status}")

# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "🔒 Minions Secure Protocol - Encrypted AI Processing with Local Models"
    "</p>",
    unsafe_allow_html=True
) 