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


def stream_response(chat: SecureMinionChat, user_msg: str, image_path=None):
    # Create a container for the assistant's message
    container = st.chat_message("assistant")
    placeholder = container.empty()
    buf = [""]

    def _cb(chunk: str):
        buf[0] += chunk
        placeholder.markdown(buf[0])

    chat.send_message_stream(user_msg, image_path=image_path, callback=_cb)


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
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Create a container for the chat history
chat_container = st.container()

# Render the conversation history in the chat container
with chat_container:
    render_history(chat.get_conversation_history())

    # Process user message and display response (if a message was submitted)
    if "last_prompt" in st.session_state and st.session_state.last_prompt:
        prompt = st.session_state.last_prompt
        user_message_container = st.chat_message("user")

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
        user_message_container.markdown(prompt)

        if st.session_state.get("stream"):
            try:
                stream_response(chat, prompt, st.session_state.image_path)
                # Clear after sending
                st.session_state.last_prompt = None
                st.session_state.image_path = None
                st.session_state.uploaded_file = None
                st.session_state.show_attachment = False
            except Exception as exc:
                st.error(f"Streaming failed: {exc}")
        else:
            with st.spinner("Thinking …"):
                try:
                    res = chat.send_message(prompt, st.session_state.image_path)
                    assistant_container = st.chat_message("assistant")
                    assistant_container.markdown(res["response"])

                    # Clear after sending
                    st.session_state.last_prompt = None
                    st.session_state.image_path = None
                    st.session_state.uploaded_file = None
                    st.session_state.show_attachment = False
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

# Create a container at the bottom for the input area
input_container = st.container()

# Chat input and attachment UI
with input_container:
    # Create columns for the attachment button and file uploader
    if st.session_state.show_attachment:
        # Add file uploader for images
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "gif"],
            help="Upload an image to include with your next message",
            key="attachment_uploader",
        )

        # If a file is uploaded, save it and update image_path
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                image_path = save_uploaded_file(uploaded_file)
                st.session_state.image_path = image_path
                st.session_state.uploaded_file = uploaded_file

            # # Add a button to remove the attachment
            # if st.button("❌ Remove image"):
            #     st.session_state.image_path = None
            #     st.session_state.uploaded_file = None
            #     st.session_state.show_attachment = False
            #     do_rerun()

    # Create a row with the chat input and attachment button
    if is_mobile():
        cols = st.columns([0.6, 0.4])
        if cols[1].button(
            "📎 Attach a file/image", help="Attach an image", use_container_width=True
        ):
            st.session_state.show_attachment = not st.session_state.show_attachment
            do_rerun()
    else:
        cols = st.columns([0.9, 0.1])
        # Attachment button in the second (smaller) column
        if cols[1].button("📎", help="Attach an image"):
            st.session_state.show_attachment = not st.session_state.show_attachment
            do_rerun()

    # Text input in the first (larger) column
    prompt = cols[0].chat_input("Type your message …")

    # Process the prompt
    if prompt:
        st.session_state.last_prompt = prompt
        do_rerun()  # This will trigger a rerun, and the message will be processed above
