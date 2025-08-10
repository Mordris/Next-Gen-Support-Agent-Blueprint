# app/core/agent.py
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.core.tools import (
    retrieve_context,
    get_order_status,
    get_current_time,
    ask_for_order_id,
)

logger = logging.getLogger(__name__)


# --- State Definition (The "Perceiver" in MCP) ---
class AgentState(TypedDict):
    """
    The complete state of our agent across its reasoning cycle.
    This acts as the "Context" or "Perceiver" in the MCP architecture.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    error_count: int
    max_iterations: int
    iteration_count: int
    tool_results: Dict[str, Any]
    confidence_score: float
    needs_correction: bool
    correction_reason: Optional[str]


# --- Node Functions (The "Models" in MCP) ---
class AgentNodes:
    """
    Contains all the node functions that represent different actions
    the agent can take. These are the "Models" in MCP architecture.
    """

    def __init__(self, llm: ChatOpenAI, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.tool_node = ToolNode(tools)

        # Create a tool-enabled LLM
        self.llm_with_tools = llm.bind_tools(tools)

        # System prompts for different reasoning phases
        self.main_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert customer support agent for eBay.

Core Directives:
- Be conversational and helpful: If information is missing, ask clarifying questions instead of giving up.
- Prioritize Tools: For policy questions use retrieve_context, for order questions use get_order_status, for time questions use get_current_time.
- Handle incomplete requests intelligently: If a user asks about "my order" but doesn't provide an order ID, ask for it politely.
- Ground Your Answers: Base answers on tool outputs, but also provide helpful guidance when tools can't be used.
- If there is a tool you need to use but you can't use the tool due to missing information, explain what you need and ask for it.
- If no tools are relevant, provide a helpful response based on conversation context.

            Available tools:
- retrieve_context: for policy-related questions (refunds, returns, privacy, etc.)
- get_order_status: for order-related questions (requires order ID)
- get_current_time: for time/date questions
- ask_for_order_id: use when user asks about order status but hasn't provided order ID

Remember: It's better to ask for clarification than to give an incomplete or unhelpful response.

Current step: {current_step}
Iteration: {iteration_count}/{max_iterations}
""",
                ),
                ("placeholder", "{messages}"),
            ]
        )

        self.correction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are reviewing your previous response for accuracy and completeness.

Previous response analysis needed because: {correction_reason}

Please:
1. Identify what was wrong or incomplete in your previous response
2. If you gave up too easily, try a different approach:
   - For order questions without IDs: Ask politely for the order ID
   - For vague questions: Ask clarifying questions
   - For policy questions: Use the retrieve_context tool
3. If you need additional tool calls, make them
4. Provide a better, more helpful response

Remember: A good customer service agent asks clarifying questions rather than giving up!
""",
                ),
                ("placeholder", "{messages}"),
            ]
        )

    def reasoning_node(self, state: AgentState) -> AgentState:
        """
        Main reasoning node - decides what to do based on the current state.
        """
        logger.info(
            f"Reasoning node - Step: {state['current_step']}, Iteration: {state['iteration_count']}"
        )

        # Format the prompt with current state context
        prompt = self.main_prompt.format_messages(
            current_step=state["current_step"],
            iteration_count=state["iteration_count"],
            max_iterations=state["max_iterations"],
            messages=state["messages"],
        )

        try:
            response = self.llm_with_tools.invoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["messages"] = [response]
            if (
                isinstance(response, AIMessage)
                and hasattr(response, "tool_calls")
                and response.tool_calls
            ):
                new_state["current_step"] = "tool_execution"
            else:
                new_state["current_step"] = "response_generation"
            new_state["iteration_count"] += 1

            return new_state

        except Exception as e:
            logger.error(f"Error in reasoning node: {e}")
            new_state = state.copy()
            new_state["error_count"] += 1
            new_state["current_step"] = "error_handling"
            return new_state

    def tool_execution_node(self, state: AgentState) -> AgentState:
        """
        Executes tools and processes their results.
        """
        logger.info("Executing tools")

        # Execute tools using LangGraph's ToolNode
        result = self.tool_node.invoke(state)

        # Update our state with tool results
        new_state = state.copy()
        new_state["messages"] = result["messages"]
        new_state["current_step"] = "response_generation"

        # Store tool results for later analysis
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                tool_name = (
                    str(message.name)
                    if hasattr(message, "name") and message.name
                    else "unknown"
                )
                new_state["tool_results"][tool_name] = message.content

        return new_state

    def response_generation_node(self, state: AgentState) -> AgentState:
        """
        Generates the final response based on tool results and conversation.
        """
        logger.info("Generating response")

        # Create a prompt that includes tool results
        prompt = self.main_prompt.format_messages(
            current_step="response_generation",
            iteration_count=state["iteration_count"],
            max_iterations=state["max_iterations"],
            messages=state["messages"],
        )

        try:
            # Generate response without tools (final answer)
            response = self.llm.invoke(
                prompt
                + [
                    HumanMessage(
                        content="""Based on the conversation above, provide a helpful response to the user.

Guidelines:
- If you tried to use a tool but it failed due to missing information (like order ID), ask the user to provide that information instead of saying you can't help.
- If you successfully used tools, base your response on those results.
- If no tools were needed, provide a direct helpful answer.
- Always rate your confidence in this response from 0.0 to 1.0 at the end.

Remember: It's better to ask clarifying questions than to give up!"""
                    )
                ]
            )

            new_state = state.copy()
            new_state["messages"] = [response]
            new_state["current_step"] = "self_assessment"

            return new_state

        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            new_state = state.copy()
            new_state["error_count"] += 1
            new_state["current_step"] = "error_handling"
            return new_state

    def self_assessment_node(self, state: AgentState) -> AgentState:
        """
        Self-correction node - evaluates the response quality and decides if correction is needed.
        """
        logger.info("Performing self-assessment")

        # Extract confidence from the last message if possible
        last_message = state["messages"][-1] if state["messages"] else None
        confidence_score = self._extract_confidence(last_message)

        # Determine if correction is needed based on various factors
        needs_correction = self._should_correct(state, confidence_score)
        correction_reason = self._get_correction_reason(state, confidence_score)

        new_state = state.copy()
        new_state["confidence_score"] = confidence_score
        new_state["needs_correction"] = needs_correction
        new_state["correction_reason"] = correction_reason
        new_state["current_step"] = "correction" if needs_correction else "complete"

        logger.info(
            f"Self-assessment: confidence={confidence_score}, needs_correction={needs_correction}"
        )

        return new_state

    def correction_node(self, state: AgentState) -> AgentState:
        """
        Correction node - attempts to improve the response.
        """
        logger.info(f"Performing correction: {state['correction_reason']}")

        correction_prompt = self.correction_prompt.format_messages(
            correction_reason=state["correction_reason"], messages=state["messages"]
        )

        try:
            corrected_response = self.llm_with_tools.invoke(correction_prompt)

            new_state = state.copy()
            new_state["messages"] = [corrected_response]
            new_state["current_step"] = (
                "tool_execution"
                if isinstance(corrected_response, AIMessage)
                and hasattr(corrected_response, "tool_calls")
                and corrected_response.tool_calls
                else "complete"
            )
            new_state["needs_correction"] = False

            return new_state

        except Exception as e:
            logger.error(f"Error in correction: {e}")
            new_state = state.copy()
            new_state["error_count"] += 1
            new_state["current_step"] = "error_handling"
            return new_state

    def error_handling_node(self, state: AgentState) -> AgentState:
        """
        Handles errors and decides whether to retry or fail.
        """
        logger.error(f"Error handling - Error count: {state['error_count']}")

        new_state = state.copy()

        if state["error_count"] >= 3:
            # Too many errors, give up
            error_message = AIMessage(
                content="I'm sorry, I encountered technical difficulties and cannot process your request at this time. Please try again later."
            )
            new_state["messages"] = [error_message]
            new_state["current_step"] = "complete"
        else:
            # Retry from reasoning
            new_state["current_step"] = "reasoning"

        return new_state

    def _extract_confidence(self, message: Optional[BaseMessage]) -> float:
        """Extract confidence score from AI message."""
        if (
            not message
            or not isinstance(message, AIMessage)
            or not isinstance(message.content, str)
        ):
            return 0.5

        content = str(message.content).lower()

        # Simple confidence extraction - in production, this could be more sophisticated
        if "confidence: " in content:
            try:
                confidence_part = content.split("confidence: ")[1].split()[0]
                return float(confidence_part)
            except (IndexError, ValueError):
                pass

        # Heuristic confidence based on content
        if "i don't know" in content or "not sure" in content:
            return 0.3
        elif "based on the" in content and "policy" in content:
            return 0.8
        else:
            return 0.6

    def _should_correct(self, state: AgentState, confidence: float) -> bool:
        """Determine if correction is needed."""
        # Don't correct if we've already done too many iterations
        if state["iteration_count"] >= state["max_iterations"]:
            return False

        # Don't correct if we're asking for clarification (this is good behavior)
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and isinstance(last_message, AIMessage):
            content = str(last_message.content).lower()
            if any(
                phrase in content
                for phrase in [
                    "could you provide",
                    "please provide",
                    "what is your order",
                    "order id",
                    "order number",
                    "which order",
                ]
            ):
                return False  # Asking for clarification is good, don't correct

        # Correct if confidence is very low (but not if we're asking questions)
        if confidence < 0.4:
            return True

        # Correct if we seem to have given up without trying alternatives
        if (
            last_message
            and "wasn't able to generate" in str(last_message.content).lower()
        ):
            return True

        # Correct if no tools were used but they might be needed for factual questions
        if (
            not state["tool_results"]
            and self._might_need_tools(state)
            and confidence < 0.7
        ):
            return True

        return False

    def _get_correction_reason(
        self, state: AgentState, confidence: float
    ) -> Optional[str]:
        """Get the reason for correction."""
        last_message = state["messages"][-1] if state["messages"] else None

        if (
            last_message
            and isinstance(last_message, AIMessage)
            and isinstance(last_message.content, str)
        ):
            if "wasn't able to generate" in last_message.content.lower():
                return "Agent gave up instead of asking for clarification."

        if confidence < 0.4:
            return f"Very low confidence score ({confidence})"

        if (
            not state["tool_results"]
            and self._might_need_tools(state)
            and confidence < 0.7
        ):
            return "No tools used but they might be helpful for this factual question"

        return None

    def _might_need_tools(self, state: AgentState) -> bool:
        """Heuristic to determine if tools might be needed."""
        if not state["messages"]:
            return False

        # Get the original user message
        user_message = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_message = str(msg.content).lower()
                break

        if not user_message:
            return False

        # Keywords that suggest tool usage
        policy_keywords = [
            "refund",
            "return",
            "policy",
            "guarantee",
            "privacy",
            "agreement",
        ]
        order_keywords = ["order", "status", "tracking", "delivery", "shipped"]
        time_keywords = ["time", "date", "when", "current"]

        return any(
            keyword in user_message
            for keyword in policy_keywords + order_keywords + time_keywords
        )


# --- Edge Functions (The "Controller" in MCP) ---
def should_continue(state: AgentState) -> str:
    """
    Routing function that determines the next step.
    This is the "Controller" in MCP architecture.
    """
    current_step = state["current_step"]

    if current_step == "reasoning":
        return "reasoning"
    elif current_step == "tool_execution":
        return "tools"
    elif current_step == "response_generation":
        return "response_generation"
    elif current_step == "self_assessment":
        return "self_assessment"
    elif current_step == "correction":
        return "correction"
    elif current_step == "error_handling":
        return "error_handling"
    else:  # complete
        return END


# --- Graph Construction ---
def create_agent_graph(llm: ChatOpenAI, tools: List[BaseTool]) -> StateGraph:
    """
    Creates and compiles the LangGraph agent.
    """
    # Initialize components
    agent_nodes = AgentNodes(llm, tools)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("reasoning", agent_nodes.reasoning_node)
    workflow.add_node("tools", agent_nodes.tool_execution_node)
    workflow.add_node("response_generation", agent_nodes.response_generation_node)
    workflow.add_node("self_assessment", agent_nodes.self_assessment_node)
    workflow.add_node("correction", agent_nodes.correction_node)
    workflow.add_node("error_handling", agent_nodes.error_handling_node)

    # Set entry point
    workflow.set_entry_point("reasoning")

    # Add edges
    workflow.add_conditional_edges("reasoning", should_continue)
    workflow.add_conditional_edges("tools", should_continue)
    workflow.add_conditional_edges("response_generation", should_continue)
    workflow.add_conditional_edges("self_assessment", should_continue)
    workflow.add_conditional_edges("correction", should_continue)
    workflow.add_conditional_edges("error_handling", should_continue)

    # Compile the graph with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# --- Main Agent Class ---
class SelfCorrectingAgent:
    """
    Main agent class that wraps the LangGraph implementation.
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [
            retrieve_context,
            get_order_status,
            get_current_time,
            ask_for_order_id,
        ]
        self.graph = create_agent_graph(llm, self.tools)

    def get_initial_state(self, message: str, max_iterations: int = 5) -> AgentState:
        """Create the initial state for a new conversation turn."""
        return {
            "messages": [HumanMessage(content=message)],
            "current_step": "reasoning",
            "error_count": 0,
            "max_iterations": max_iterations,
            "iteration_count": 0,
            "tool_results": {},
            "confidence_score": 0.0,
            "needs_correction": False,
            "correction_reason": None,
        }

    def stream_response(self, message: str, session_id: str, max_iterations: int = 5):
        """
        Stream the agent's response with detailed step information.
        """
        initial_state = self.get_initial_state(message, max_iterations)
        config = {"configurable": {"thread_id": session_id}}

        try:
            for event in self.graph.stream(initial_state, config=config):
                yield event
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield {"error": str(e)}

    async def astream_response(
        self, message: str, session_id: str, max_iterations: int = 5
    ):
        """
        Async version of stream_response.
        """
        initial_state = self.get_initial_state(message, max_iterations)
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self.graph.astream(initial_state, config=config):
                yield event
        except Exception as e:
            logger.error(f"Error in astream_response: {e}")
            yield {"error": str(e)}
