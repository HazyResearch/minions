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


def render_history(history):
    for msg in history:
        st.chat_message(msg["role"]).markdown(msg["content"])


def stream_response(chat: SecureMinionChat, user_msg: str):
    container = st.chat_message("assistant")
    placeholder = container.empty()
    buf = [""]

    def _cb(chunk: str):
        buf[0] += chunk
        placeholder.markdown(buf[0])

    chat.send_message_stream(user_msg, callback=_cb)


# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Secure Minion Chat", page_icon="🔒")


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

st.title("🔒 Secure Minion Chat")


# ── connection settings area (open by default) ─────────────────────────────
with st.expander("Connection Settings", expanded=True):
    supervisor_url = st.text_input(
        "Supervisor URL (if using hosted app, don't change this!)",
        value=st.session_state.get("supervisor_url", "http://20.57.33.122:5056"),
    )
    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.get("system_prompt", SYSTEM_PROMPT),
        height=68,
    )

    cols = st.columns(3)
    if cols[0].button("🔌 Connect / Re‑connect"):
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

    if cols[1].button("🗑️ Clear chat") and "chat" in st.session_state:
        st.session_state.chat.clear_conversation()
        do_rerun()

    if cols[2].button("✂️ End session") and "chat" in st.session_state:
        st.session_state.chat.end_session()
        del st.session_state.chat
        do_rerun()


# ── main chat area ─────────────────────────────────────────────────────────
chat = st.session_state.get("chat")
if chat is None:
    st.info("Click **Connect** to start your secure chat session.")
    st.stop()

render_history(chat.get_conversation_history())

if prompt := st.chat_input("Type your message …"):
    st.chat_message("user").markdown(prompt)

    if st.session_state.get("stream"):
        try:
            stream_response(chat, prompt)
        except Exception as exc:
            st.error(f"Streaming failed: {exc}")
    else:
        with st.spinner("Thinking …"):
            try:
                res = chat.send_message(prompt)
                st.chat_message("assistant").markdown(res["response"])
            except Exception as exc:
                st.error(f"Request failed: {exc}")
