from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langchain_community.tools import ShellTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

shell_tool = ShellTool()


class State(MessagesState):
    summary: str | None


@dataclass
class ContextSchema:
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2"] | None = None


model = init_chat_model(
    model="gpt-5.1",
    temperature=0.7,
    max_tokens=2048
)
model = model.bind_tools([shell_tool])


def conversation(state: State, runtime: Runtime[ContextSchema]):        
    system_message = [
        SystemMessage(
            content=(
                "You are a helpful assistant with access to shell commands via ShellTool. "
                "Important guidelines for file searches:\n"
                "- Avoid searching from root (/) as it takes too long\n"
                "- Use specific directories like /home/username or ~/Documents\n"
                "- Use 'find' with -maxdepth option to limit search depth\n"
                "- For file content, limit output with 'head' or 'tail' for large files\n"
                "- Consider using 'locate' command if the system has it (much faster than find)\n"
                "Commands have a 60-second timeout, so plan accordingly."
            )
        )
    ]

    return {
        "messages": [
            model.invoke(system_message + state["messages"])
        ]
    }
    
def should_continue(state: State, runtime: Runtime[ContextSchema]) -> Literal["tools", END]:
    """Route after conversation based on tool calls."""

    # Check if there are tool calls (similar to tools_condition)
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

    return END


# Build workflow
agent_builder = StateGraph(State, context_schema=ContextSchema)

agent_builder.add_node("conversation", conversation)
agent_builder.add_node("tools", ToolNode([shell_tool]))

agent_builder.add_edge(START, "conversation")
agent_builder.add_conditional_edges("conversation", should_continue)
agent_builder.add_edge("tools", "conversation")
agent_builder.add_edge("conversation", END)
tools_mcp_agent = agent_builder.compile()
