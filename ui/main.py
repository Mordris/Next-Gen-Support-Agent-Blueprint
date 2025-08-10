# ui/main.py
import streamlit as st
import requests
import os
import uuid
import json

# --- Page Configuration ---
st.set_page_config(page_title="Next-Gen Support Agent", page_icon="🤖", layout="wide")
st.title("🤖 Next-Gen Self-Correcting Support Agent")
st.write(
    "I am a self-correcting autonomous agent with advanced reasoning capabilities! "
    "Ask me about eBay policies, order statuses, or general questions. "
    "I can analyze my own responses and improve them if needed."
)

# --- Environment & API Configuration ---
API_KEY = os.environ.get("API_KEY")
API_URL = "http://api:8000/chat/stream"
HEALTH_URL = "http://api:8000/health"
STATUS_URL = "http://api:8000/agent/status"

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


@st.cache_data(ttl=30)
def check_agent_status():
    try:
        headers = {"X-API-KEY": API_KEY}
        response = requests.get(STATUS_URL, headers=headers, timeout=3)
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None


health_status = check_api_health()
agent_status = check_agent_status()

# --- UI Rendering ---
with st.sidebar:
    st.header("🔍 Debug Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")

    # API Health
    if health_status and health_status.get("status"):
        st.success("✅ API Status: Online")
        redis_status = (
            "Connected" if health_status.get("redis_connected") else "Disconnected"
        )
        status_color = "success" if health_status.get("redis_connected") else "warning"
        getattr(st, status_color)(f"📊 Redis Status: {redis_status}")
    else:
        st.error("❌ API Status: Offline")

    # Agent Status
    if agent_status:
        if agent_status.get("status") == "initialized":
            st.success("🧠 Agent Status: Ready")
            st.info(f"🛠️ Tools: {agent_status.get('tools_count', 0)}")
        else:
            st.warning("🧠 Agent Status: Not Ready")
    else:
        st.error("🧠 Agent Status: Unknown")

    if st.button("🗑️ Clear Chat History"):
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
        # Create expandable sections for different phases
        thinking_expander = st.expander("🧠 Agent Reasoning", expanded=False)
        thinking_placeholder = thinking_expander.empty()

        assessment_expander = st.expander(
            "🔍 Self-Assessment & Correction", expanded=False
        )
        assessment_placeholder = assessment_expander.empty()

        response_placeholder = st.empty()

        full_response = ""
        thinking_steps = ""
        assessment_steps = ""
        current_step = ""

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
                            if not json_str or json_str.strip() == "":
                                continue

                            event_data = json.loads(json_str)
                            event_type = event_data.get("event")

                            # Reasoning events
                            if event_type == "reasoning_start":
                                iteration = event_data.get("iteration", 0)
                                thinking_steps += (
                                    f"**🔄 Reasoning Cycle {iteration + 1}**\n\n"
                                )
                                thinking_placeholder.markdown(thinking_steps)
                                current_step = "reasoning"

                            elif event_type == "tool_start":
                                tool_name = event_data.get("name", "unknown_tool")
                                # Map internal tool names to display names
                                display_name = {
                                    "retrieve_context": "Knowledge Retrieval",
                                    "get_order_status": "Order Status Check",
                                    "get_current_time": "Current Time",
                                    "ask_for_order_id": "Request Order ID",
                                }.get(tool_name, tool_name)

                                thinking_steps += (
                                    f"**🔧 Tool Started:** `{display_name}`\n\n"
                                )
                                thinking_placeholder.markdown(thinking_steps)

                            elif event_type == "tool_end":
                                tool_name = event_data.get("name", "unknown_tool")
                                display_name = {
                                    "retrieve_context": "Knowledge Retrieval",
                                    "get_order_status": "Order Status Check",
                                    "get_current_time": "Current Time",
                                    "ask_for_order_id": "Request Order ID",
                                }.get(tool_name, tool_name)

                                thinking_steps += (
                                    f"**✅ Tool Finished:** `{display_name}`\n\n"
                                )
                                tool_output = event_data.get("output", "")

                                if tool_output:
                                    # Format the output based on tool type
                                    if tool_name == "get_order_status":
                                        try:
                                            if isinstance(
                                                tool_output, str
                                            ) and tool_output.startswith("{"):
                                                parsed = json.loads(tool_output)
                                                formatted_output = json.dumps(
                                                    parsed, indent=2
                                                )
                                            else:
                                                formatted_output = str(tool_output)
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

                            elif event_type == "response_start":
                                thinking_steps += "**📝 Generating Response...**\n\n"
                                thinking_placeholder.markdown(thinking_steps)

                            elif event_type == "self_assessment":
                                confidence = event_data.get("confidence", 0.0)
                                needs_correction = event_data.get(
                                    "needs_correction", False
                                )
                                reason = event_data.get("reason")

                                assessment_steps += (
                                    f"**🎯 Confidence Score:** {confidence:.2f}\n\n"
                                )

                                if needs_correction:
                                    assessment_steps += (
                                        f"**⚠️ Correction Needed:** {reason}\n\n"
                                    )
                                    assessment_expander.expanded = True
                                else:
                                    assessment_steps += (
                                        "**✅ Response Quality:** Acceptable\n\n"
                                    )

                                assessment_placeholder.markdown(assessment_steps)

                            elif event_type == "correction_start":
                                reason = event_data.get("reason", "Unknown reason")
                                assessment_steps += (
                                    f"**🔄 Applying Correction:** {reason}\n\n"
                                )
                                assessment_placeholder.markdown(assessment_steps)
                                assessment_expander.expanded = True

                            elif event_type == "final_response":
                                content = event_data.get("content", "")
                                full_response = content
                                response_placeholder.markdown(full_response)

                            elif event_type == "stream_end":
                                if full_response:
                                    response_placeholder.markdown(full_response)
                                break

                            elif event_type == "error":
                                error_msg = event_data.get(
                                    "data", "Unknown error occurred"
                                )
                                st.error(f"🚨 Agent Error: {error_msg}")
                                break

                        except json.JSONDecodeError:
                            # It's safe to ignore lines that are not valid JSON.
                            continue
                        except Exception as e:
                            st.error(f"Event processing error: {e}")
                            continue

            # Finalize the response
            if full_response:
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                # If no response was captured, show a fallback message
                fallback_msg = "I apologize, but I wasn't able to generate a complete response. Please try asking your question again."
                response_placeholder.markdown(fallback_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": fallback_msg}
                )

        except requests.RequestException as e:
            st.error(f"🔌 Connection Error: {e}")
            st.error("Please check if the API server is running and accessible.")
        except Exception as e:
            st.error(f"🐛 Unexpected Error: {e}")

# --- Footer Information ---
with st.expander("ℹ️ About This Agent", expanded=False):
    st.markdown("""
    ### 🧠 Self-Correcting Architecture

    This agent uses a **LangGraph state machine** with the following capabilities:

    **🔄 Reasoning Loop:**
    - **Reasoning:** Analyzes your question and decides what to do
    - **Tool Execution:** Uses specialized tools when needed
    - **Response Generation:** Creates an initial response
    - **Self-Assessment:** Evaluates response quality and confidence
    - **Correction:** Improves the response if needed

    **🛠️ Available Tools:**
    - **Knowledge Retrieval:** Access to eBay policies and documentation
    - **Order Status Check:** Real-time order status lookup (requires order ID)
    - **Current Time:** Date and time information
    - **Request Order ID:** Intelligently asks for missing order information

    **✨ Key Features:**
    - **Intelligent Conversations:** Asks clarifying questions instead of giving up
    - **Self-Correction:** Can identify and fix inadequate responses
    - **Confidence Scoring:** Evaluates its own certainty
    - **Iterative Improvement:** Multiple reasoning cycles if needed
    - **Memory:** Maintains conversation context with Redis
    - **Streaming:** Real-time response generation with step visibility

    **🎯 Best Use Cases:**
    - eBay policy questions (refunds, returns, privacy)
    - Order status inquiries
    - General support questions
    - Complex multi-step problems
    """)
