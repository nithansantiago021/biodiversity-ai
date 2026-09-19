import os
import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Biodiversity AI Engine", page_icon="🌍")
st.title("Biodiversity & Ecological Restoration AI Engine")


def get_api_url() -> str:
    # 1. Try environment variable first
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url

    # 2. Try Streamlit secrets (safely handled if secrets.toml is missing)
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass

    # 3. Fallback to live Render service URL
    return "https://biodiversity-backend.onrender.com"


BACKEND_URL: str = get_api_url()

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
            res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=30)
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
            st.error(f"Failed to connect to backend API: {e}")
