from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime


def add(a: int, b: int) -> int:
    """Adding a and b.

    Use this tool to add two numbers together.

    Args:
        a: first int
        b: second int

    Returns:
        Result of addition

    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplying a and b.

    Use this tool to multiply two numbers together.

    Args:
        a: first int
        b: second int

    Returns:
        Result of multiplication

    """
    return a * b


def divide(a: int, b: int) -> float:
    """Dividing a and b.

    Use this tool to divide two numbers together.

    Args:
        a: first int
        b: second int

    Returns:
        Result of division

    """
    return a / b
    

tool_input_map = {
    "add": add,
    "multiply": multiply,
    "divide": divide,
}


class State(MessagesState):
    summary: str | None
    llm_input_messages: list | None


@dataclass
class ContextSchema:
    model: Literal["gpt-4.1-nano", "claude-3-haiku-20240307", "gemma3:1b"]
    temperature: float
    max_tokens: int
    messages_strategy: Literal["trim_count", "trim_tokens", "summarize"]
    message_strategy_keep: int
    message_strategy_summarize: int
    message_strategy_delete: int
    agentic_tools: list[str]
    workflow_tools: list[str] | None = None


model = init_chat_model(configurable_fields="any")


def get_llm_context(state: State, runtime: Runtime[ContextSchema]) -> list:
    """Compute which messages to send to LLM based on strategy.

    Does NOT modify state - pure computation.
    """
    messages = state["messages"]
    strategy = runtime.context.messages_strategy
    
    # Trim by count - keep last N messages
    if strategy == "trim_count":
        return trim_messages(
            messages,
            strategy="last",
            token_counter=len,
            max_tokens=runtime.context.message_strategy_keep,
            start_on="human",
            end_on=("human", "tool"),
        )
    # Trim by tokens - keep last N tokens
    elif strategy == "trim_tokens":
        return trim_messages(
            messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=runtime.context.message_strategy_keep,
            start_on="human",
            end_on=("human", "tool"),
        )
    
    # Default: all messages
    return messages


def call_llm(runtime, system_message, messages, use_system_message):
    tools = [
        tool_input_map[name] for name in runtime.context.agentic_tools if name in tool_input_map
    ] if runtime.context.agentic_tools else []
    call_model = model.bind_tools(tools) if tools else model

    if use_system_message:
        messages = system_message + messages
    if runtime.context.model in ["gpt-4.1-nano", "claude-3-haiku-20240307"]:
        return {
            "messages": [
                call_model.invoke(
                    messages,
                    config={
                        "configurable": {
                            "model": runtime.context.model,
                            "temperature": runtime.context.temperature,
                            "max_tokens": runtime.context.max_tokens
                        }
                    }
                )
            ]
        }
    elif runtime.context.model == "gemma3:1b":
        ollama_model = ChatOllama(
            model=runtime.context.model,
            temperature=runtime.context.temperature,
            num_predict=runtime.context.max_tokens,
            base_url="http://172.20.0.1:11434"
        )
        return {"messages": [ollama_model.invoke(system_message + messages)]}


def summarize_conversation(state: State, runtime: Runtime[ContextSchema]):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    response = call_llm(
        runtime, None, state["messages"] + [HumanMessage(content=summary_message)], use_system_message=False
    )
    summary_text = response["messages"][-1].content if response is not None else ""
    
    # After summarizing, optionally delete old messages from thread
    if runtime.context.message_strategy_delete:
        # Use trim_messages to intelligently decide what to keep
        messages_to_keep = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=runtime.context.message_strategy_delete,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,  # Preserve system messages if any
        )
        
        # Now find what to remove
        message_ids_to_keep = {m.id for m in messages_to_keep if m.id}
        delete_messages = [
            RemoveMessage(id=m.id) for m in state["messages"] if m.id and m.id not in message_ids_to_keep
        ]
        return {"messages": delete_messages, "summary": summary_text}
    else:
        return {"summary": summary_text}


def conversation(state: State, runtime: Runtime[ContextSchema]):
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of the conversation so far: {summary}"
    else:
        summary_message = ""

    system_message = [
        SystemMessage(
            content=(
                "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                f"{summary_message}"
            )
        )
    ]

    messages = get_llm_context(state, runtime)
    print(f"LEN OF MESSAGES: {len(messages)}")

    return call_llm(runtime, system_message, messages, use_system_message=True)



def route_after_conversation(state: State, runtime: Runtime[ContextSchema]) -> str:
    """Route after conversation based on tool calls and summary needs."""

    # Check if there are tool calls (similar to tools_condition)
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

    # Check if we should summarize
    if (
        runtime.context.messages_strategy == "summarize" and
        len(state["messages"]) > runtime.context.message_strategy_summarize
    ):
        return "summarize"

    return "end"


# Build workflow
agent_builder = StateGraph(State, context_schema=ContextSchema)


agent_builder.add_node("conversation", conversation)
agent_builder.add_node("tools", ToolNode([add, multiply, divide]))
agent_builder.add_node("summarize_conversation", summarize_conversation)

agent_builder.add_edge(START, "conversation")
agent_builder.add_conditional_edges(
    "conversation",
    route_after_conversation,  # Single routing function
    {
        "tools": "tools",
        "summarize": "summarize_conversation",
        "end": END,
    }
)
agent_builder.add_edge("tools", "conversation")
agent_builder.add_edge("summarize_conversation", END)
simple_agent = agent_builder.compile()
