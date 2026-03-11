from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


@dataclass
class ContextSchema:
    token: str
    model: str


# Schema for structured output
class CodeChange(BaseModel):
    """A single code change to apply to a file."""
    
    start_line: int = Field(
        description="Starting line number (1-indexed) of the code to replace"
    )
    end_line: int = Field(
        description="Ending line number (1-indexed) of the code to replace (inclusive)"
    )
    new_content: str = Field(
        description="The new code to insert (can be multiple lines, will replace lines start_line to end_line)"
    )
    explanation: str = Field(
        description="Brief explanation of why this change is being made"
    )
    change_type: Literal["fix", "refactor", "add", "remove", "optimize"] = Field(
        description="Category of change being made"
    )


class FileEditProposal(BaseModel):
    """A proposal for editing a Python file with structured changes."""
    
    file_path: str = Field(
        description="Path to the file being edited"
    )
    summary: str = Field(
        description="High-level summary of all proposed changes"
    )
    changes: list[CodeChange] = Field(
        description="List of individual changes to apply, ordered by line number (ascending)"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Assessment of how risky these changes are"
    )
    requires_testing: bool = Field(
        description="Whether these changes should be tested before committing"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) in the proposed changes"
    )
    
    
class MainAgentMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        main_model = init_chat_model(
            model_provider="openai",
            model=request.runtime.context.model,
            api_key=request.runtime.context.token,
            disable_streaming=True
        )
        new_request = request.override(model=main_model)

        return await handler(new_request)


coding_assistant_agent = create_deep_agent(
    # Using default model which will be overridden in the middleware
    model=init_chat_model(model_provider="openai", model="gpt-5-nano", streaming=False),
    system_prompt=(
        "You are a software developer assistant suggesting new features and improvements for the codebase. "
        "You have been initialized to work on a specific, single project which is located under "
        "/home/app/agent-context."
    ),
    backend=FilesystemBackend(root_dir="/home/app/agent-context"),
    response_format=FileEditProposal,
    context_schema=ContextSchema,
    middleware=[MainAgentMiddleware()],
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "edit", "reject"]}
    },
)
