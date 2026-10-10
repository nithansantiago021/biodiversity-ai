import os
import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Biodiversity AI Engine", page_icon="🌍")
st.title("Biodiversity & Ecological Restoration AI Engine")


def get_api_url() -> str:
    # 1. Respect explicit environment variable first (e.g., set via terminal or Docker)
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url.rstrip("/")

    # 2. Check Streamlit secrets (safely handled if secrets.toml doesn't exist)
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"].rstrip("/")
    except Exception:
        pass

    # 3. Check if local backend is reachable; default to local dev server
    local_url = "http://localhost:8000"
    try:
        # Quick 0.5s ping check to see if local Docker / FastAPI backend is running
        r = requests.get(f"{local_url}/health", timeout=0.5)
        if r.status_code == 200:
            return local_url
    except Exception:
        pass

    # 4. Fallback to production Render service
    return "https://biodiversity-backend.onrender.com"


BACKEND_URL: str = get_api_url()

# Sidebar indicator so you always know which environment you are connected to
with st.sidebar:
    st.caption(f"Connected Backend: `{BACKEND_URL}`")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"demo-{uuid.uuid4()}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Process user input
if user_input := st.chat_input("Ask about land metrics or restoration strategies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    payload = {"session_id": st.session_state.session_id, "message": user_input}

    with st.spinner("Analyzing ecological data..."):
        try:
            res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
            res.raise_for_status()
            res_json = res.json()

            # Handles common response key names
            bot_reply = (
                res_json.get("content")
                or res_json.get("response")
                or res_json.get("message")
                or "Error: Empty response body."
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": bot_reply}
            )
            with st.chat_message("assistant"):
                st.write(bot_reply)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend API ({BACKEND_URL}): {e}")
