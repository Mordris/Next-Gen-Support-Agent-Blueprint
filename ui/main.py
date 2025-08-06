import streamlit as st
import requests
import os

st.set_page_config(page_title="Next-Gen Support Agent", page_icon="🤖", layout="wide")
st.title("🤖 Next-Gen Support Agent")
st.write("I am an autonomous agent with a toolbelt!")

API_KEY = os.environ.get("API_KEY")
API_URL = "http://api:8000/chat/stream"

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages (but not the current streaming one)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle assistant's response
    with st.chat_message("assistant"):
        try:
            headers = {"X-API-KEY": API_KEY}
            data = {"message": prompt}

            # Use a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""
            is_thinking = True

            # Make the streaming request
            response = requests.post(API_URL, json=data, headers=headers, stream=True)
            response.raise_for_status()

            # Stream the response
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    # Handle the thinking phase
                    if is_thinking and chunk.startswith("🤔"):
                        # Show thinking animation
                        response_placeholder.markdown("🤔 **Thinking...**")
                        continue
                    elif chunk == "\r":
                        # Clear thinking and prepare for real response
                        is_thinking = False
                        full_response = ""
                        continue

                    # Handle the actual response streaming
                    if not is_thinking:
                        full_response += chunk
                        # Show the response with a typing cursor
                        response_placeholder.markdown(full_response + "▌")

            # After streaming is complete, show final response without cursor
            response_placeholder.markdown(full_response)

            # NOW add the complete response to session state
            if full_response.strip():
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            # Add error to session state too
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
