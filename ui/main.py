# ui/main.py
import streamlit as st
import requests
import os

st.set_page_config(page_title="Next-Gen Support Agent", page_icon="🤖", layout="wide")
st.title("🤖 Next-Gen Support Agent")
st.write("This is a 'dumb' echo-bot. The full wiring is complete!")

# It's good practice to get secrets from environment variables
API_KEY = os.environ.get("API_KEY")
API_URL = "http://api:8000/chat/stream"  # Docker Compose service name

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            headers = {"X-API-KEY": API_KEY}
            data = {"message": prompt}

            with requests.post(API_URL, json=data, headers=headers, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except requests.exceptions.RequestException as e:
            full_response = f"An error occurred: {e}"
            response_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
