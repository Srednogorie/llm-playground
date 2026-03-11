import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.store.postgres import PostgresStore
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


@dataclass
class ContextSchema:
    token: str
    main_model: str
    sql_model: str
    analyst_model: str


async def get_aws_docs_mcp_tools() -> list:
    client = MultiServerMCPClient(
        {
            "awslabs.aws-documentation-mcp-server": {
                "command": "awslabs.aws-documentation-mcp-server",
                "args": [],
                "env": {
                    "FASTMCP_LOG_LEVEL": "ERROR",
                    "AWS_DOCUMENTATION_PARTITION": "aws",
                    "MCP_USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                },
                "transport": "stdio"
            }
        }
    )
    
    return await client.get_tools()


def get_engine_for_chinook_db():
    return create_engine(
        "sqlite:///./Chinook_Sqlite.sqlite",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)
# The model is used for the QuerySQLCheckerTool tool of the toolkit and it's difficult to override it in the middleware
sql_model = init_chat_model(model_provider="groq", model="openai/gpt-oss-120b", streaming=False)
toolkit = SQLDatabaseToolkit(db=db, llm=sql_model)

initial_default_model = init_chat_model(model_provider="groq", model="llama-3.1-8b-instant", streaming=False)


mcp_servers = asyncio.run(get_aws_docs_mcp_tools())


# Use PostgresStore.from_conn_string as a context manager
store_ctx = PostgresStore.from_conn_string(os.environ["POSTGRES_URI"])
store = store_ctx.__enter__()
store.setup()


# This is connectivity test for the sql-agent agent to which you have access, please invoke it with a random message.
# SQL subagent
class SqlSubagentMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        sql_model = init_chat_model(
            model_provider="groq",
            model=request.runtime.context.sql_model,
            api_key=request.runtime.context.token,
            streaming=False
        )
        new_request = request.override(model=sql_model)

        return await handler(new_request)


sql_subagent = create_agent(
    # Default model which will be overridden by the middleware
    model=initial_default_model,
    name="sql-agent",
    system_prompt="""
        You are a SQL agent that can execute queries against the company's database. Run queries to retrieve
        information from the database. The company database is a SQLite database so you must use the SQLite syntax
        when writing queries.
    """,
    middleware=[SqlSubagentMiddleware()],
    tools=toolkit.get_tools(),
)
compiled_sql_subagent = CompiledSubAgent(
    name="sql-agent",
    description="A powerful SQL agent that can execute queries against the company's database",
    runnable=sql_subagent
)


# Analyst subagent
class AnalystSubagentMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        analyst_model = init_chat_model(
            model_provider="groq",
            model=request.runtime.context.analyst_model,
            api_key=request.runtime.context.token,
            streaming=False
        )
        new_request = request.override(model=analyst_model)

        return await handler(new_request)


analyst_subagent = create_agent(
    # Default model which will be overridden by the middleware
    model=initial_default_model,
    name="analyst-agent",
    system_prompt="""
        You are an analyst agent that can analyze data from the company's database.
        You will be given a data to analyze.
    """,
    middleware=[AnalystSubagentMiddleware()],
)
compiled_analyst_subagent = CompiledSubAgent(
    name="analyst-agent",
    description="You are an analyst agent that can analyze data from the company's database.",
    runnable=analyst_subagent
)

# Main agent
class MainAgentMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        main_model = init_chat_model(
            model_provider="groq",
            model=request.runtime.context.main_model,
            api_key=request.runtime.context.token,
            streaming=False
        )
        new_request = request.override(model=main_model)

        return await handler(new_request)


agent_with_subagents = create_deep_agent(
    model=initial_default_model,
    system_prompt=(
        "You are a helpful assistant helping the company employees with their tasks. Your role is to "
        "orchestrate the subagents you have access to. With their help, you can accomplish complex tasks and answer "
        "the user's questions."
    ),
    subagents=[compiled_sql_subagent, compiled_analyst_subagent],
    context_schema=ContextSchema,
    middleware=[MainAgentMiddleware()],
    store=store,
    backend=lambda rt: CompositeBackend(
        default=FilesystemBackend(root_dir="/home/app/application-data"),
        routes={
            "/memories/": StoreBackend(rt),
            # "/home": FilesystemBackend()
        }
    ),
    tools=mcp_servers
)
