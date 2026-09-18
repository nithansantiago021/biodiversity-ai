import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Biodiversity AI Engine", page_icon="🌍")
st.title("Biodiversity & Ecological Restoration AI Engine")

# Backend live URL (will be updated once Render deploys)
BACKEND_URL = st.sidebar.text_input(
    "Backend API URL", value="https://biodiversity-ai-backend.onrender.com"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = f"demo-{uuid.uuid4()}"
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about land metrics or restoration strategies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    payload = {"session_id": st.session_state.session_id, "message": user_input}

    with st.spinner("Analyzing ecological data..."):
        try:
            res = requests.post(f"{BACKEND_URL}/chat", json=payload).json()
            bot_reply = res.get("content", "Error processing request.")
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_reply}
            )
            with st.chat_message("assistant"):
                st.write(bot_reply)
        except Exception as e:
            st.error(f"Failed to connect to backend API: {e}")
