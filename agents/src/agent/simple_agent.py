from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime


@dataclass
class ContextSchema:
    model: str = "gpt-4.1-nano"
    temperature: float = 0.7
    max_tokens: int = 1024


model = init_chat_model(configurable_fields="any")


def llm_call(state: dict, runtime: Runtime[ContextSchema]):
    # if runtime.context.model in ["gpt-4.1-nano", "claude-3-haiku-20240307"]:
    if True:
        return {
            "messages": [
                model.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"],
                    config={
                        "configurable": {
                            "model": "gpt-4.1-nano",
                            # "temperature": runtime.context.temperature,
                            # "max_tokens": runtime.context.max_tokens
                        }
                    }
                )
            ]
        }
    elif runtime.context.model == "ollama:gemma3:1b":
        ollama_model = ChatOllama(
            model="gemma3:1b",
            temperature=runtime.context.temperature,
            num_predict=runtime.context.max_tokens,
            base_url="http://172.20.0.1:11434"
        )
        
        return {
            "messages": [
                ollama_model.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"],
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
