from dataclasses import dataclass
from enum import Enum
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field


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
    web_search_context: list[str]
    wiki_search_context: list[str]


@dataclass
class ContextSchema:
    model: Literal["gpt-5-nano", "claude-3-haiku-20240307", "gemma3:1b"]
    temperature: float
    max_tokens: int
    messages_strategy: Literal["trim_count", "trim_tokens", "summarize"]
    message_strategy_keep: int
    message_strategy_summarize: int
    message_strategy_delete: int
    agentic_tools: list[str] | None = None
    workflow_tools: list[str] | None = None
    
    
class SearchQuery(BaseModel):
    wikipedia_query: str | None = Field(None, description="Search query for retrieval.")
    web_query: str | None = Field(None, description="Search query for retrieval.")
    
    
def should_search(state: State, runtime: Runtime[ContextSchema]) -> Literal["search", "conversation"]:
    """Determine if we need to search or can answer directly."""
    
    # If no search tools configured, skip search
    if not runtime.context.workflow_tools:
        return "conversation"
    
    # On first message, always search if tools are available
    if len(state["messages"]) <= 1:
        return "search"
        
    web_search_context = state.get("web_search_context", "")
    if web_search_context:
        web_search_context_message = f"Web Search Context: {web_search_context}"
    else:
        web_search_context_message = ""
        
    wiki_search_context = state.get("wiki_search_context", "")
    if wiki_search_context:
        wiki_search_context_message = f"Wikipedia Search Context: {wiki_search_context}"
    else:
        wiki_search_context_message = ""
    
    # For follow-up questions, use LLM to decide
    decision_prompt: list = [SystemMessage(
        content="""
            Analyze the user's latest message, conversation history and external search contexts if available.
            
            The latest message from the user is the message that needs to be analyzed most because it may or may not be
            relevant to the current conversation. Pay special attention to the last message from the user.
            
            Determine if NEW information from external sources (web/wikipedia) is needed.
            
            Return "search" if:
            - The question asks about current events, facts, or topics not in the conversation or the context
            - The user explicitly asks to look something up
            
            Return "no search" if:
            - The question can be answered from the existing conversation context
            - It's a clarification or follow-up about information already retrieved
            - It's a simple calculation or reasoning task
            - Generally prefer to avoid searching unless absolutely necessary
        """
    )]
    
    class Decision(str, Enum):
        ANSWER_FROM_CONTEXT = "answer_from_context"
        ANSWER_FROM_HISTORY = "answer_from_history"
        NEEDS_NEW_SEARCH = "needs_new_search"
    
    class SearchDecision(BaseModel):
        decision: Decision = Field(
            description="""Choose ONE:
            - answer_from_context: The existing Web Search Context context or Wikipedia Search Context has the answer
            - answer_from_history: The conversation history has the answer
            - needs_new_search: Need to search external sources
            """
        )
        reasoning: str = Field(
            description="Quote the relevant part of context/history OR explain what's missing"
        )
        
    # Add conversation history
    decision_prompt.extend(state["messages"])

    if web_search_context_message or wiki_search_context_message:
        search_context_human_message = HumanMessage(
            content=f"""
                HISTORICAL SEARCH CONTEXT (from previous questions in this conversation):
                    - Web Search Context: {web_search_context_message}
                    - Wikipedia Search Context: {wiki_search_context_message}
                NOTE: The above context may NOT be relevant to the user's LATEST question below.
            """
        )
        decision_prompt.append(search_context_human_message)
    
    # Make the latest user message explicit
    decision_prompt.append(HumanMessage(content=f'''
        LATEST USER QUESTION TO ANALYZE: {state["messages"][-1].content}
        
        Does this LATEST question require a NEW search, or can it be answered from the historical context above?
    '''))
    
    decision_model = search_question_model.with_structured_output(SearchDecision)
    decision = decision_model.invoke(decision_prompt)

    return "search" if decision.decision == Decision.NEEDS_NEW_SEARCH else "conversation"
    
    
search_instructions = SystemMessage(
    content="""
        You will be given a conversation between an llm assistant and a user.
        Your goal is to generate search queries for different search engines.
        There is no need to analyze the full conversation.
        Pay particular attention to the final question posed by the user.
        
        RULES:
        - Wikipedia search is lexical and title-based.
        Return short queries (1–3 words) targeting canonical article titles.
        - Web search is semantic.
        Return descriptive, natural-language queries.
    """
)


model = init_chat_model(configurable_fields="any")
search_question_model = ChatOpenAI(model="gpt-5-nano", temperature=0)


def search_web(state: State, runtime: Runtime[ContextSchema]):
    """ Retrieve docs from web search """
    
    if runtime.context.workflow_tools and "tavily" in runtime.context.workflow_tools:

        # Search
        tavily_search = TavilySearch(max_results=1, include_raw_content=True)
    
        # Search query
        structured_llm = search_question_model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions] + state["messages"])
    
        # Search
        search_docs = tavily_search.invoke(search_query.web_query)
    
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["raw_content"]}\n</Document>'
                for doc in search_docs["results"]
            ],
        )
    
        return {"web_search_context": [formatted_search_docs]}


def search_wikipedia(state: State, runtime: Runtime[ContextSchema]):
    """ Retrieve docs from wikipedia """
    
    if runtime.context.workflow_tools and "wikipedia" in runtime.context.workflow_tools:

        # Search query
        structured_llm = search_question_model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions] + state["messages"])
        
        if search_query.wikipedia_query:
            # Search
            search_docs = WikipediaLoader(query=search_query.wikipedia_query, load_max_docs=2).load()
        
            # Format
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}"'
                    f'page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                    for doc in search_docs
                ],
            )
        
            return {"wiki_search_context": [formatted_search_docs]}


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

    if runtime.context.model in ["gpt-5-nano", "claude-3-haiku-20240307"]:
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
        
    web_search_context = state.get("web_search_context", "")
    if web_search_context:
        web_search_context_message = f"Web Search Context: {web_search_context}"
    else:
        web_search_context_message = ""
        
    wiki_search_context = state.get("wiki_search_context", "")
    if wiki_search_context:
        wiki_search_context_message = f"Wikipedia Search Context: {wiki_search_context}"
    else:
        wiki_search_context_message = ""
        
    system_message = [
        SystemMessage(
            content=(
                # "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                "You are a helpful assistant. Use the provided context to answer the user's question."
                f"{summary_message}"
                f"{web_search_context_message}"
                f"{wiki_search_context_message}"
            )
        )
    ]

    messages = get_llm_context(state, runtime)

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

agent_builder.add_node("search_web", search_web)
agent_builder.add_node("search_wikipedia", search_wikipedia)
agent_builder.add_node("conversation", conversation)
agent_builder.add_node("tools", ToolNode([add, multiply, divide]))
agent_builder.add_node("summarize_conversation", summarize_conversation)

# Route from START based on search decision
agent_builder.add_conditional_edges(
    START,
    should_search,
    {
        "search": "search_web",
        "conversation": "conversation",
    }
)
agent_builder.add_edge("search_web", "search_wikipedia")
agent_builder.add_edge("search_wikipedia", "conversation")
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
