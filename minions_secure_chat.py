# secure_streamlit_chat.py
"""
A minimal Streamlit web interface that uses the SecureMinionChat
client to talk to a remote LLM over the Minions secure‑chat protocol.

Run locally with:
    streamlit run secure_streamlit_chat.py

If you see an AttributeError for `st.experimental_rerun`, you are
running a new Streamlit (>v1.27) where the function was renamed to
`st.rerun`.  This version handles both APIs transparently.
"""

import streamlit as st
from streamlit_theme import st_theme
import os
import tempfile
import uuid
from pathlib import Path
import fitz  # PyMuPDF for PDF processing

from secure.minions_chat import SecureMinionChat

# SYSTEM_PROMPT = """
# You are a helpful AI assistant, chatting with a user through a secure protocol. The messages between you are encrypted and decrypted on the way, your computations are happening inside a trusted execution environment.
# """

SYSTEM_PROMPT = """You are a helpful AI assistant. 
"""
# ── helpers ────────────────────────────────────────────────────────────────


def do_rerun():
    """Streamlit renamed `experimental_rerun` → `rerun` in v1.27.
    Call the one that exists.
    """
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def is_mobile():
    """Check if we're in mobile mode (based on session state toggle)."""
    return st.session_state.get("mobile_mode", False)


def render_history(history):
    for msg in history:
        message_container = st.chat_message(msg["role"])

        # If the message has an image, display it first
        if "image_url" in msg:
            # For now, we don't display images in history as we're not storing them
            message_container.markdown("*[Image was shared with this message]*")

        # Display the text content
        message_container.markdown(msg["content"])


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory and return the path."""
    # Create a temporary directory if it doesn't exist
    temp_dir = Path(tempfile.gettempdir()) / "secure_chat_uploads"
    temp_dir.mkdir(exist_ok=True)

    # Generate a unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = temp_dir / unique_filename

    # Save the file
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
        try:
            # Open the PDF file
            with fitz.open(pdf_path) as doc:
                # Get the number of pages
                num_pages = len(doc)
                st.info(f"Processing PDF with {num_pages} pages...")

                # Iterate through each page
                for page_num in range(num_pages):
                    # Get the page
                    page = doc[page_num]
                    # Extract text from the page
                    pdf_content += page.get_text()
                    # Add a separator between pages
                    if page_num < num_pages - 1:
                        pdf_content += "\n\n"
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None

        return pdf_content

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None


def stream_response(
    chat: SecureMinionChat,
    user_msg: str,
    image_path=None,
    pdf_path=None,
    folder_path=None,
):
    # Create a container for the assistant's message
    container = st.chat_message("assistant")
    placeholder = container.empty()
    buf = [""]

    def _cb(chunk: str):
        buf[0] += chunk
        placeholder.markdown(buf[0])

    chat.send_message_stream(
        user_msg,
        image_path=image_path,
        pdf_path=pdf_path,
        folder_path=folder_path,
        callback=_cb,
    )


# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Secure Chat", page_icon="🔒")

# Add custom CSS for centering elements
st.markdown(
    """
<style>
/* Center checkboxes and their labels */
.stCheckbox {
    display: flex;
    justify-content: center;
}
.stCheckbox > label {
    display: flex;
    justify-content: center;
    width: 100%;
}

/* Center button rows */
div[data-testid="column"] {
    display: flex;
    justify-content: center;
}

/* Center text in buttons */
.stButton button {
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Center title and subtitle */
h1, .subtitle {
    text-align: center;
}

/* Center the horizontal rule */
hr {
    margin-left: auto;
    margin-right: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


def is_dark_mode():
    theme = st_theme()
    if theme and "base" in theme:
        if theme["base"] == "dark":
            return True
    return False


# Check theme setting
dark_mode = is_dark_mode()

# Choose image based on theme
if dark_mode:
    image_path = (
        "assets/minions_logo_no_background.png"  # Replace with your dark mode image
    )
else:
    image_path = "assets/minions_logo_light.png"  # Replace with your light mode image


# Display Minions logo at the top
st.image(image_path, use_container_width=True)

# add a horizontal line that is width of image
st.markdown("<hr style='width: 100%;'>", unsafe_allow_html=True)

st.title("🔒 Secure Chat")
# add a one line that says "secure encrypted chat running inside a trusted execution environment"
st.markdown(
    "<p class='subtitle' style='font-size: 20px; color: #888;'>Secure encrypted chat running inside a trusted execution environment!</p>",
    unsafe_allow_html=True,
)


# ── connection settings area (collapsed by default) ─────────────────────────────
with st.expander("Connection Settings", expanded=True):

    # Add mobile mode toggle
    mobile_mode = st.checkbox(
        "📱 Mobile mode",
        value=st.session_state.get("mobile_mode", False),
        help="Enable for better layout on small screens or mobile devices",
    )
    st.session_state.mobile_mode = mobile_mode

    # Use a checkbox to toggle visibility of advanced settings
    show_advanced = st.checkbox("Show Advanced Server Settings", value=False)

    # Show supervisor URL only if advanced settings checkbox is checked
    if show_advanced:
        supervisor_url = st.text_input(
            "Supervisor URL (if using hosted app, don't change this!)",
            value=st.session_state.get("supervisor_url", "http://20.57.33.122:5056"),
        )
    else:
        # Keep the variable in memory but don't show the input
        supervisor_url = st.session_state.get(
            "supervisor_url", "http://20.57.33.122:5056"
        )

    if is_mobile():
        system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.get("system_prompt", SYSTEM_PROMPT),
            height=68,
        )

        # Center the buttons in mobile view
        st.markdown(
            "<div style='display: flex; justify-content: center;'>",
            unsafe_allow_html=True,
        )
        button_cols = st.columns([1, 1, 1])
        connect_btn = button_cols[0].button("🔌 Connect", use_container_width=True)
        clear_btn = button_cols[1].button("🗑️ Clear", use_container_width=True)
        end_btn = button_cols[2].button("❌ End", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if connect_btn:
            try:
                chat = SecureMinionChat(supervisor_url, system_prompt)
                info = chat.initialize_secure_session()
                st.session_state.chat = chat
                st.session_state.supervisor_url = supervisor_url
                st.session_state.system_prompt = system_prompt
                st.session_state.stream = True
                st.success(f"Connected – session ID: {info['session_id']}")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

        if clear_btn and "chat" in st.session_state:
            st.session_state.chat.clear_conversation()
            do_rerun()

        if end_btn and "chat" in st.session_state:
            st.session_state.chat.end_session()
            del st.session_state.chat
            do_rerun()

    else:
        system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.get("system_prompt", SYSTEM_PROMPT),
            height=68,
        )

        # Center the buttons in desktop view
        st.markdown(
            "<div style='display: flex; justify-content: center;'>",
            unsafe_allow_html=True,
        )
        button_cols = st.columns([1, 1, 1])
        connect_btn = button_cols[0].button(
            "🔌 Connect / Re‑connect", use_container_width=True
        )
        clear_btn = button_cols[1].button("🗑️ Clear chat", use_container_width=True)
        end_btn = button_cols[2].button("✂️ End session", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if connect_btn:
            try:
                chat = SecureMinionChat(supervisor_url, system_prompt)
                info = chat.initialize_secure_session()
                st.session_state.chat = chat
                st.session_state.supervisor_url = supervisor_url
                st.session_state.system_prompt = system_prompt
                st.session_state.stream = True
                st.success(f"Connected – session ID: {info['session_id']}")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

        if clear_btn and "chat" in st.session_state:
            st.session_state.chat.clear_conversation()
            do_rerun()

        if end_btn and "chat" in st.session_state:
            st.session_state.chat.end_session()
            del st.session_state.chat
            do_rerun()


# ── main chat area ─────────────────────────────────────────────────────────
chat = st.session_state.get("chat")
if chat is None:
    st.info("Click **Connect** to start your secure chat session.")
    st.stop()

# Initialize session state for attachment UI
if "show_attachment" not in st.session_state:
    st.session_state.show_attachment = False
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None
if "folder_path" not in st.session_state:
    st.session_state.folder_path = None
if "folder_content_summary" not in st.session_state:
    st.session_state.folder_content_summary = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "attachment_type" not in st.session_state:
    st.session_state.attachment_type = None

# Create a container for the chat history
chat_container = st.container()

# Render the conversation history in the chat container
with chat_container:
    render_history(chat.get_conversation_history())

    # Process user message and display response (if a message was submitted)
    if "last_prompt" in st.session_state and st.session_state.last_prompt:
        prompt = st.session_state.last_prompt
        user_message_container = st.chat_message("user")

        # If there's PDF content, modify the prompt to include it
        if st.session_state.pdf_content:
            # Format the prompt to include PDF content and user's question
            modified_prompt = f"Context from PDF:\n\n{st.session_state.pdf_content}\n\nQuery: {prompt}"
            # Display a note that PDF was attached
            user_message_container.markdown("*[PDF document was attached]*")
            # Update the prompt with the modified version that includes PDF content
            prompt = modified_prompt

        # If there's a folder path, display a note (actual processing happens in the send_message_stream function)
        if st.session_state.folder_path:
            user_message_container.markdown("*[Folder with files was attached]*")
            if st.session_state.folder_content_summary:
                user_message_container.markdown(
                    f"*{st.session_state.folder_content_summary}*"
                )

        # If there's an uploaded image, display it
        if st.session_state.image_path:
            # Also display the image preview in the user message
            try:
                user_message_container.image(
                    st.session_state.image_path, caption="Uploaded image"
                )
            except Exception as e:
                st.warning(f"Could not display image preview: {str(e)}")

        # Display the text message
        user_message_container.markdown(
            st.session_state.last_prompt
        )  # Show original prompt to user

        if st.session_state.get("stream"):
            try:
                stream_response(
                    chat,
                    prompt,
                    st.session_state.image_path,
                    st.session_state.pdf_path,
                    st.session_state.folder_path,
                )
                # Clear after sending
                st.session_state.last_prompt = None
                st.session_state.image_path = None
                st.session_state.pdf_path = None
                st.session_state.pdf_content = None
                st.session_state.folder_path = None
                st.session_state.folder_content_summary = None
                st.session_state.uploaded_file = None
                st.session_state.show_attachment = False
                st.session_state.attachment_type = None
            except Exception as exc:
                st.error(f"Streaming failed: {exc}")
        else:
            with st.spinner("Thinking …"):
                try:
                    res = chat.send_message(
                        prompt,
                        st.session_state.image_path,
                        st.session_state.pdf_path,
                        st.session_state.folder_path,
                    )
                    assistant_container = st.chat_message("assistant")
                    assistant_container.markdown(res["response"])

                    # Clear after sending
                    st.session_state.last_prompt = None
                    st.session_state.image_path = None
                    st.session_state.pdf_path = None
                    st.session_state.pdf_content = None
                    st.session_state.folder_path = None
                    st.session_state.folder_content_summary = None
                    st.session_state.uploaded_file = None
                    st.session_state.show_attachment = False
                    st.session_state.attachment_type = None
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

# Create a container at the bottom for the input area
input_container = st.container()

# Chat input and attachment UI
with input_container:
    # Create columns for the attachment button and file uploader
    if st.session_state.show_attachment:
        # Add attachment type selection
        attachment_type = st.radio(
            "Select attachment type",
            options=["Image", "PDF Document", "Folder"],
            horizontal=True,
            key="attachment_type_radio",
        )
        st.session_state.attachment_type = attachment_type.lower()

        if st.session_state.attachment_type == "image":
            # Add file uploader for images
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg", "gif"],
                help="Upload an image to include with your next message",
                key="image_uploader",
            )

            # If a file is uploaded, save it and update image_path
            if uploaded_file is not None:
                with st.spinner("Processing image..."):
                    image_path = save_uploaded_file(uploaded_file)
                    st.session_state.image_path = image_path
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.pdf_path = None
                    st.session_state.pdf_content = None
                    st.session_state.folder_path = None
                    st.session_state.folder_content_summary = None

        elif st.session_state.attachment_type == "pdf document":
            # Add file uploader for PDFs
            uploaded_file = st.file_uploader(
                "Upload a PDF document",
                type=["pdf"],
                help="Upload a PDF document to include as context with your next message",
                key="pdf_uploader",
            )

            # If a PDF is uploaded, save it and process the text
            if uploaded_file is not None:
                with st.spinner("Processing PDF document..."):
                    pdf_path = save_uploaded_file(uploaded_file)
                    st.session_state.pdf_path = pdf_path
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.image_path = None
                    st.session_state.folder_path = None
                    st.session_state.folder_content_summary = None

                    # Process the PDF to extract text
                    pdf_content = extract_text_from_pdf(pdf_path)
                    if pdf_content:
                        # Store the PDF content in the session state
                        st.session_state.pdf_content = pdf_content
                        st.success(
                            f"PDF processed successfully! ({len(pdf_content)} characters extracted)"
                        )
                    else:
                        st.error(
                            "Failed to extract text from the PDF. Please try another document."
                        )

        elif st.session_state.attachment_type == "folder":
            # Create folder upload interface with improved guidance
            st.info(
                "📁 Upload multiple files to create a virtual folder for the LLM to analyze jointly. "
            )

            # Multiple file uploader
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["txt", "pdf", "png", "jpg", "jpeg", "gif"],
                accept_multiple_files=True,
                help="Upload multiple files to include as context with your next message",
                key="folder_uploader",
            )

            # Customize the button text based on whether it's mobile mode or not
            if uploaded_files:
                with st.spinner("Setting up folder..."):
                    # Create a temporary directory for this batch of files
                    temp_dir = (
                        Path(tempfile.gettempdir())
                        / f"secure_chat_folder_{uuid.uuid4().hex}"
                    )
                    temp_dir.mkdir(exist_ok=True)

                    # Display progress information
                    progress_bar = st.progress(0)
                    file_status = st.empty()
                    total_files = len(uploaded_files)

                    # Save all files to the temporary directory
                    for i, file in enumerate(uploaded_files):
                        progress = int((i / total_files) * 100)
                        progress_bar.progress(progress)
                        file_status.text(
                            f"Processing file {i+1}/{total_files}: {file.name}"
                        )

                        file_path = temp_dir / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())

                    # Complete the progress bar
                    progress_bar.progress(100)

                    # Count file types
                    file_status.text("Analyzing files...")
                    txt_count = len(list(temp_dir.glob("*.txt")))
                    pdf_count = len(list(temp_dir.glob("*.pdf")))
                    img_count = sum(
                        len(list(temp_dir.glob(f"*.{ext}")))
                        for ext in ["jpg", "jpeg", "png", "gif"]
                    )

                    # Store the folder path and summary in session state
                    folder_summary = f"Folder contains {txt_count} text files, {pdf_count} PDF files, and {img_count} images."
                    st.session_state.folder_path = str(temp_dir)
                    st.session_state.folder_content_summary = folder_summary
                    st.session_state.image_path = None
                    st.session_state.pdf_path = None
                    st.session_state.pdf_content = None

                    # Clear temporary UI elements
                    file_status.empty()

                    # Show success message
                    st.success(
                        f"✅ {folder_summary} Ready to use in your next message!"
                    )

    # Create a row with the chat input and attachment button
    if is_mobile():
        cols = st.columns([0.6, 0.4])
        if cols[1].button(
            "📎 Attach a file",
            help="Attach an image, PDF, or folder of files",
            use_container_width=True,
        ):
            st.session_state.show_attachment = not st.session_state.show_attachment
            do_rerun()
    else:
        cols = st.columns([0.9, 0.1])
        # Attachment button in the second (smaller) column
        if cols[1].button("📎", help="Attach an image, PDF, or folder of files"):
            st.session_state.show_attachment = not st.session_state.show_attachment
            do_rerun()

    # Display the current attachment if there is one
    if st.session_state.image_path:
        st.caption("✅ Image attached")
    elif st.session_state.pdf_path:
        st.caption(f"✅ PDF attached: {os.path.basename(st.session_state.pdf_path)}")
    elif st.session_state.folder_path:
        st.caption(f"✅ Folder attached: {st.session_state.folder_content_summary}")

    # Text input in the first (larger) column
    prompt = cols[0].chat_input("Type your message …")

    # Process the prompt
    if prompt:
        st.session_state.last_prompt = prompt
        do_rerun()  # This will trigger a rerun, and the message will be processed above
