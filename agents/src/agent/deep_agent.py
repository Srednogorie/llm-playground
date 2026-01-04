from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


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


model = init_chat_model(model="gpt-5.1")
deep_agent = create_deep_agent(
    model=model,
    system_prompt="You are a software developer assistant suggesting new features and improvements for the codebase.",
    backend=FilesystemBackend(root_dir="/home"),
    response_format=FileEditProposal
)
