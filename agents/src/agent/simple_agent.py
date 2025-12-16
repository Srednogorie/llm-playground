from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime


class State(MessagesState):
    summary: str | None
    llm_input_messages: list | None


@dataclass
class ContextSchema:
    model: Literal["gpt-4.1-nano", "claude-3-haiku-20240307", "gemma3:1b"]
    temperature: float
    max_tokens: int
    messages_strategy: Literal["delete", "trim_count", "trim_tokens", "summarize"]
    message_strategy_number: int


model = init_chat_model(configurable_fields="any")


def call_llm(runtime, summary_message, messages):
    system_message = [
        SystemMessage(
            content=(
                "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                f"{summary_message}"
            )
        )
    ]
    if runtime.context.model in ["gpt-4.1-nano", "claude-3-haiku-20240307"]:
        return {
            "messages": [
                model.invoke(
                    system_message + messages,
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
        return {
            "messages": [
                ollama_model.invoke(system_message + messages)
            ]
        }


def apply_messages_strategy(state: State, runtime: Runtime[ContextSchema]):
    """We separate the message state and the messages that are passed to the LLM.
    
    This is somehow important, especially if we want to keep the messages in the state/thread but pass to the model
    only subset of them, for example the last n.
    """
    # Delete all but the n most recent messages. This is raw, it's going to delete the messages from the state/thread.
    response = {"messages": state["messages"], "llm_input_messages": state.get("llm_input_messages", [])}
    if runtime.context.messages_strategy == "delete":
        messages = [
            RemoveMessage(id=m.id) if m.id else m
            for m in state["messages"][:-runtime.context.message_strategy_number]
        ]
        # We are deleting messages from the state/thread so we need to return the updated state
        response["messages"] = messages
        response["llm_input_messages"] = [m for m in state["messages"][-runtime.context.message_strategy_number:]]
    elif runtime.context.messages_strategy == "trim_tokens":
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=runtime.context.message_strategy_number,
            start_on="human",
            end_on=("human", "tool"),
        )
        response["llm_input_messages"] = messages
    elif runtime.context.messages_strategy == "trim_count":
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=runtime.context.message_strategy_number,
            start_on="human",
            end_on=("human", "tool"),
        )
        response["llm_input_messages"] = messages
    return response


def summarize_conversation(state: State, runtime: Runtime[ContextSchema]):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary} \n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    delete_messages = [
        RemoveMessage(id=m.id) if m.id else m for m in state["messages"][:-runtime.context.message_strategy_number - 1]
    ]
    return {"messages": delete_messages, "summary": response.content}


def should_summarize(state: State, runtime: Runtime[ContextSchema]):
    # messages = state["messages"]
    # if len(messages) > 6:
    #     return "summarize_conversation"
    return END


def conversation(state: State, runtime: Runtime[ContextSchema]):
    # elif runtime.context.messages_strategy == "summarize":
    #     messages = summarize_messages(
    #         state["messages"],
    #         strategy="last",
    #         token_counter=count_tokens_approximately,
    #         max_tokens=20,
    #         start_on="human",
    #         end_on=("human", "tool"),
    #     )
        
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of the conversation so far: {summary}"
    else:
        summary_message = ""

    messages = state["llm_input_messages"]
    
    return call_llm(runtime, summary_message, messages)


# Build workflow
agent_builder = StateGraph(State, context_schema=ContextSchema)

agent_builder.add_node("apply_messages_strategy", apply_messages_strategy)
agent_builder.add_node("conversation", conversation)
agent_builder.add_node("summarize_conversation", summarize_conversation)

agent_builder.add_edge(START, "apply_messages_strategy")
agent_builder.add_edge("apply_messages_strategy", "conversation")
agent_builder.add_conditional_edges("conversation", should_summarize)
agent_builder.add_edge("summarize_conversation", END)
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
