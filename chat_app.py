# app.py – multi‑chat with automatic titling that back‑fills old conversations
import streamlit as st
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY

st.set_page_config(page_title="ChatGPT 🎈", page_icon="💬")
st.title("🗨️ between us")

# ── helper utilities ────────────────────────────────────────────────────────

def user_turns(conv):
    """Number of *user* messages in a conversation list."""
    return sum(m["role"] == "user" for m in conv)


def request_title(conv):
    """Ask the model for a concise title; return None if it declines."""
    sys_prompt = {
        "role": "system",
        "content": (
            "You create short, descriptive titles (≤ 6 words) for chat conversations. "
            "Respond with one such title, or the single word 'None' if uncertain."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[sys_prompt] + conv[-20:],  # last ~20 messages keeps prompt small
        )
        title = resp.choices[0].message.content.strip()
        return None if title.lower() == "none" else title
    except Exception:
        return None


def ensure_titles():
    """Back‑fill titles for all conversations that qualify but are untitled."""
    for i, conv in enumerate(st.session_state.convs):
        if st.session_state.titles[i] is None and user_turns(conv) >= 1:
            title = request_title(conv)
            if title:
                st.session_state.titles[i] = title

# ── state initialisation ────────────────────────────────────────────────────
if "convs" not in st.session_state:
    st.session_state.convs = [
        [{"role": "assistant", "content": "Hi — ask me anything!"}]
    ]
if "titles" not in st.session_state:
    st.session_state.titles = [None]
if "current" not in st.session_state:
    st.session_state.current = 0

# attempt to title any existing conversations even if user hasn't typed yet
ensure_titles()

# ── sidebar: pick or create a conversation ─────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 Conversations")
    labels = [t or f"Chat {i+1}" for i, t in enumerate(st.session_state.titles)]
    choice = st.radio("Choose a chat", labels, index=st.session_state.current)
    st.session_state.current = labels.index(choice)

    if st.button("➕  New chat"):
        st.session_state.convs.append([
            {"role": "assistant", "content": "Hi — ask me anything!"}
        ])
        st.session_state.titles.append(None)
        st.session_state.current = len(st.session_state.convs) - 1
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

# ── main area ───────────────────────────────────────────────────────────────
idx = st.session_state.current
messages = st.session_state.convs[idx]

for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Say something"):
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking…"):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    # title current conversation if still untitled and now long enough
    if st.session_state.titles[idx] is None and user_turns(messages) >= 3:
        title = request_title(messages)
        if title:
            st.session_state.titles[idx] = title
