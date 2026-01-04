import asyncio
import sqlite3

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_aws_docs_mcp_tools() -> list:
    client = MultiServerMCPClient(
        {
        "awslabs.aws-documentation-mcp-server": {
                "command": "uvx",
                "args": ["awslabs.aws-documentation-mcp-server@latest"],
                "env": {
                    "FASTMCP_LOG_LEVEL": "ERROR",
                    "AWS_DOCUMENTATION_PARTITION": "aws",
                    "MCP_USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                },
                # "disabled": False,
                # "autoApprove": [],
                "transport": "stdio"
            }
        }
    )
    
    return await client.get_tools()


def get_engine_for_chinook_db():
    return create_engine(
        "sqlite:////home/middlefour/Downloads/Chinook_Sqlite.sqlite",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()

db = SQLDatabase(engine)
sql_model = init_chat_model(model="gpt-5-mini")
toolkit = SQLDatabaseToolkit(db=db, llm=sql_model)

sql_subagent = {
    "name": "sql-agent",
    "description": "A powerful SQL agent that can execute queries against the company's database",
    "system_prompt": """
        You are a SQL agent that can execute queries against the company's database. Run queries to retrieve
        information from the database.
    """,
    "tools": toolkit.get_tools(),
    # "model": "gpt-5-mini",  # Optional override, defaults to main agent model
}
subagents = [sql_subagent]


all_tools = asyncio.run(get_aws_docs_mcp_tools())


model = init_chat_model(model="gpt-5.1")

deep_agent_subagent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful assistant helping the company employees with their tasks.",
    subagents=subagents,
    backend=FilesystemBackend(root_dir="/home"),
    tools=all_tools
)
