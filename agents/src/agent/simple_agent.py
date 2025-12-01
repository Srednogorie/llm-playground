from dataclasses import dataclass
from socket import timeout

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime


@dataclass
class ContextSchema:
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int


model = init_chat_model()


def llm_call(state: dict, runtime: Runtime[ContextSchema]):
    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"],
                config={"configurable": {"model": "gpt-4.1-nano", "temperature": runtime.context.temperature}}
            )
        ]
    }


# Build workflow
agent_builder = StateGraph(MessagesState, context_schema=ContextSchema)

# Add nodes
agent_builder.add_node("llm_call", llm_call)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")

# Compile the agent
agent = agent_builder.compile()


# from IPython.display import Image, display
# # Show the agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# # Invoke
# from langchain.messages import HumanMessage
# messages = [HumanMessage(content="Add 3 and 4.")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()
