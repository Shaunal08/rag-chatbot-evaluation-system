import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"   # FastAPI backend

st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 RAG Chatbot")
st.caption("Powered by FastAPI + ChromaDB + local embeddings")

# ---------------------------
# Session Utilities
# ---------------------------

def create_session():
    return str(uuid.uuid4())[:8]

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()

def reset_session():
    st.session_state.session_id = create_session()
    st.session_state.history = []

# ---------------------------
# UI Sidebar
# ---------------------------

st.sidebar.header("Session")
st.sidebar.write(f"Current session: `{st.session_state.session_id}`")

if st.sidebar.button("➕ New Session"):
    reset_session()

st.sidebar.markdown("---")
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K Retrieved Chunks", 1, 10, 4)

# ---------------------------
# Chat History
# ---------------------------

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.json(msg["sources"])

# ---------------------------
# Chat Input
# ---------------------------

user_input = st.chat_input("Ask something about your documents...")

if user_input:
    # add user message to local UI history
    st.session_state.history.append({"role": "user", "text": user_input})

    # display user message
    with st.chat_message("user"):
        st.write(user_input)

    # call the backend /chat endpoint
    payload = {
        "session_id": st.session_state.session_id,
        "question": user_input,
        "topk": top_k
    }

    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        data = response.json()
    except Exception as e:
        st.error(f"Failed to contact backend: {e}")
        st.stop()

    assistant_msg = data.get("answer", "[No answer returned]")
    sources = data.get("sources", [])

    # push assistant message to UI history
    st.session_state.history.append({
        "role": "assistant",
        "text": assistant_msg,
        "sources": sources
    })

    # display assistant reply
    with st.chat_message("assistant"):
        st.write(assistant_msg)
        if sources:
            with st.expander("Sources"):
                st.json(sources)
