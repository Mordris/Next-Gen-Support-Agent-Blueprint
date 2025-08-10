# ui/main.py
import streamlit as st
import requests
import os
import uuid
import json

# --- Page Configuration ---
st.set_page_config(page_title="Next-Gen Support Agent", page_icon="🤖", layout="wide")
st.title("🤖 Next-Gen Support Agent")
st.write(
    "I am an autonomous agent with a toolbelt and memory! Ask me about the Transformer model or an order status."
)

# --- Environment & API Configuration ---
API_KEY = os.environ.get("API_KEY")
API_URL = "http://api:8000/chat/stream"
HEALTH_URL = "http://api:8000/health"

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Health Check ---
@st.cache_data(ttl=10)
def check_api_health():
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None


health_status = check_api_health()

# --- UI Rendering ---
with st.sidebar:
    st.header("Debug Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    if health_status and health_status.get("status"):
        st.success("API Status: Online")
        st.success(
            f"Redis Status: {'Connected' if health_status.get('redis_connected') else 'Disconnected'}"
        )
    else:
        st.error("API Status: Offline")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Ask me anything!", disabled=not health_status):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_expander = st.expander("🤔 Agent is thinking...", expanded=True)
        thinking_placeholder = thinking_expander.empty()
        response_placeholder = st.empty()

        full_response = ""
        thinking_steps = ""

        try:
            headers = {"X-API-KEY": API_KEY, "Accept": "text/event-stream"}
            data = {"message": prompt, "session_id": st.session_state.session_id}

            with requests.post(
                API_URL, json=data, headers=headers, stream=True, timeout=300
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith(b"data:"):
                        try:
                            json_str = line.decode("utf-8")[5:]
                            if not json_str:
                                continue
                            event_data = json.loads(json_str)
                            event_type = event_data.get("event")

                            if event_type == "tool_start":
                                tool_name = event_data.get("name", "unknown_tool")
                                # Map internal tool names to display names
                                display_name = {
                                    "retrieve_context": "Knowledge Retrieval",
                                    "get_order_status": "Order Status Check",
                                }.get(tool_name, tool_name)

                                thinking_steps += (
                                    f"**🔧 Tool Started:** `{display_name}`\n\n"
                                )
                                thinking_placeholder.markdown(thinking_steps)

                            elif event_type == "tool_end":
                                tool_name = event_data.get("name", "unknown_tool")
                                # Map internal tool names to display names
                                display_name = {
                                    "retrieve_context": "Knowledge Retrieval",
                                    "get_order_status": "Order Status Check",
                                }.get(tool_name, tool_name)

                                thinking_steps += (
                                    f"**✅ Tool Finished:** `{display_name}`\n\n"
                                )
                                tool_output = event_data.get("output", "")

                                if tool_output:
                                    # Format the output based on tool type
                                    if tool_name == "get_order_status":
                                        # For order status, show formatted JSON
                                        try:
                                            if isinstance(tool_output, str):
                                                # Try to parse if it's a JSON string
                                                if tool_output.startswith("{"):
                                                    import json

                                                    parsed = json.loads(tool_output)
                                                    formatted_output = json.dumps(
                                                        parsed, indent=2
                                                    )
                                                else:
                                                    formatted_output = tool_output
                                            else:
                                                formatted_output = json.dumps(
                                                    tool_output, indent=2
                                                )
                                            thinking_steps += (
                                                f"```json\n{formatted_output}\n```\n\n"
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            TypeError,
                                            ValueError,
                                        ):
                                            thinking_steps += f"```\n{str(tool_output)[:500]}\n```\n\n"
                                    else:
                                        # For other tools, show truncated text
                                        output_preview = str(tool_output)[:300]
                                        if len(str(tool_output)) > 300:
                                            output_preview += (
                                                "...\n[Output truncated for display]"
                                            )
                                        thinking_steps += (
                                            f"```\n{output_preview}\n```\n\n"
                                        )

                                thinking_placeholder.markdown(thinking_steps)

                            elif event_type == "final_token":
                                token = event_data.get("data", "")
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")

                            elif event_type == "stream_end":
                                response_placeholder.markdown(full_response)
                                break

                            elif event_type == "error":
                                error_msg = event_data.get(
                                    "data", "Unknown error occurred"
                                )
                                st.error(f"Agent Error: {error_msg}")
                                break

                        except json.JSONDecodeError:
                            continue

            if full_response:
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

        except requests.RequestException as e:
            st.error(f"Connection Error: {e}")
            st.error("Please check if the API server is running and accessible.")
