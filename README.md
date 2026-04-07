# LLM Playground

> A full-stack, hands-on exploration of the LangGraph / LangChain ecosystem —
> four production-patterned agents, a self-hosted API server, and a custom Next.js chat UI,
> built to systematically cover as many LLM agent capabilities as possible in one codebase.

This is a deliberate **breadth** project, not a finished product.
Each agent targets a distinct slice of the agent-building problem: runtime configurability, memory management, tool use, human oversight, multi-agent orchestration, MCP integrations, and persistent cross-session memory.
The goal was to learn these patterns by building them — and to leave a single reference codebase that demonstrates them together.

---

## Repository Layout

```
llm-playground/
├── agents/                        # Four LangGraph agents  (Python · uv)
│   ├── src/agent/
│   │   ├── simple_agent.py        # Multi-feature conversational agent
│   │   ├── tools_mcp_agent.py     # Shell / MCP tool-use agent
│   │   ├── coding_assistant.py    # HITL coding assistant
│   │   └── agent_with_subagents.py# Multi-agent orchestrator
│   ├── Chinook_Sqlite.sqlite      # Bundled SQL sample database
│   ├── langgraph.json             # Graph entrypoint registry
│   └── pyproject.toml
├── backend/
│   └── docker-compose.yaml        # Langgraph API · Redis · PostgreSQL
├── frontend/                      # Next.js 15 chat UI  (TypeScript · pnpm)
│   └── src/
│       ├── app/                   # One route per agent
│       └── components/thread/
│           └── agents-custom-threads/
└── bin/
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Next.js 15  Frontend                                │
│                                                                                  │
│    /simple-agent · /tools-mcp-agent · /coding-assistant-agent · /subagents       │
└─────────────────────────────────────┬────────────────────────────────────────────┘
                                      │  HTTP + SSE  (token streaming)
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  Self-Hosted LangGraph API  ·  :8123                             │
│                                                                                  │
│         simple-agent · tools-mcp-agent · coding-assistant · agent-with-subagents │
└───────────────────────────────┬──────────────────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
┌───────────────────────────┐    ┌──────────────────────────────────────────────────┐
│         Redis  6          │    │                   PostgreSQL  16                 │
│  pub/sub · SSE streaming  │    │   LangGraph state checkpoints                    │
└───────────────────────────┘    │   PostgresStore  (cross-session agent memories)  │
                                 └──────────────────────────────────────────────────┘
```

---

## The Four Agents

### 1. `simple_agent.py` — Multi-Feature Conversational Agent

The most feature-rich agent. Built as an explicit exercise in covering as many core LangGraph patterns as possible within a single compiled graph.

**Graph flow**

```
START
  └─► should_search?
           ├─[search needed]──► search_web ──► search_wikipedia ─┐
           └─[context ok]────────────────────────────────────────►┤
                                                                  ▼
                                                            conversation
                                                                 │
                                              ┌──────────────────┼──────────────────┐
                                              ▼                  ▼                  ▼
                                           [tools]          [summarize]           [END]
                                              │                  │
                                         ToolNode     summarize_conversation
                                              │                  │
                                         conversation           END
```

**What it explores**

- **Multi-provider LLM** — `init_chat_model` with Groq backend; model swapped per-request at runtime (`llama-3.3-70b-versatile`, `qwen/qwen3-32b`, `moonshotai/kimi-k2-instruct`, `openai/gpt-oss-120b`, `meta-llama/llama-4-maverick`)
- **LangGraph Runtime Context** — `ContextSchema` dataclass injected via `Runtime[ContextSchema]`; every parameter (model, temperature, max_tokens, strategy, tool selections) is fully dynamic per-request with no graph recompilation
- **Configurable agentic tools** — math tools (`add`, `multiply`, `divide`) toggled by the user at runtime; bound to the model only when selected
- **Configurable workflow tools** — Tavily web search and Wikipedia, independently selectable per conversation
- **Intelligent search routing** — a dedicated LLM call using `with_structured_output` (`SearchDecision`) decides whether existing context is sufficient or a new external search is warranted before each response
- **Three conversation memory strategies**, selected at runtime:

  | Strategy | Mechanism |
  |---|---|
  | `trim_count` | `trim_messages` keeping the last *N* messages by count |
  | `trim_tokens` | `trim_messages` keeping the last *N* tokens via `count_tokens_approximately` |
  | `summarize` | LLM-generated rolling summary + selective `RemoveMessage` pruning of old messages |

- **Structured output** — `SearchQuery` and `SearchDecision` Pydantic models via `with_structured_output` drive search query generation and the routing decision

---

### 2. `tools_mcp_agent.py` — Shell / MCP Tools Agent

A deliberately minimal agent focused entirely on tool use and real OS access.
Named `tools-mcp-agent` because it is the designated surface for Model Context Protocol integrations.

**Graph flow**

```
START ──► conversation ─────────────────────────────────────── END
               ▲              │  [tool calls present]
               │              ▼
               └──────── ToolNode  (ShellTool)
```

**What it explores**

- **`ShellTool`** from `langchain_community` — the agent executes real OS commands (`find`, `grep`, `head`, `ls`, etc.) directly on the container host
- **ReAct loop** — standard `conversation → should_continue? → tools → conversation` cycle kept intentionally lean to isolate the tool-use pattern
- **MCP-ready surface** — the graph is structurally prepared to receive additional tools from a `MultiServerMCPClient` alongside or in place of the shell tool

---

### 3. `coding_assistant.py` — Coding Assistant with Human-in-the-Loop

Explores higher-level agent abstractions, richly structured file-editing proposals, and mandatory human oversight before any write operation lands on disk.

**Graph flow** *(managed internally by `deepagents`)*

```
START ──► agent loop
               ├──► read_file                       (autonomous)
               │
               └──► write_file ──────► INTERRUPT
                                             │
                                  ┌──────────┼──────────┐
                                  ▼          ▼          ▼
                              [approve]   [edit]    [reject]
                              apply as-is  apply     discard
                                           edit
```

**What it explores**

- **`deepagents` / `create_deep_agent`** — higher-level abstraction over LangGraph with built-in filesystem access and structured response format support
- **`FilesystemBackend`** — agent reads from and proposes writes to a live codebase mounted at `/home/app/agent-context`
- **Structured output (`FileEditProposal`)** — every proposed change carries full metadata via a Pydantic model:
  - `start_line`, `end_line`, `new_content` — precise line-level targeting
  - `change_type`: `fix | refactor | add | remove | optimize`
  - `risk_level`, `confidence` (0.0–1.0), `requires_testing`
- **Human-in-the-loop** — `interrupt_on={"write_file": {"allowed_decisions": ["approve", "edit", "reject"]}}` pauses the graph before any file mutation; the frontend surfaces the full proposal through the `AgentInbox` component
- **`AgentMiddleware` (`MainAgentMiddleware`)** — intercepts every model call to inject the runtime-configured OpenAI model; the agent definition never changes to swap models

---

### 4. `agent_with_subagents.py` — Multi-Agent Orchestration

The most architecturally complex agent. Explores multi-agent delegation, per-subagent model injection, live external knowledge via MCP, and persistent cross-session memory.

**Graph flow** *(managed internally by `deepagents`)*

```
START ──► main orchestrator (deepagents)
               │
               ├──► sql-agent ──► SQLDatabaseToolkit ──► Chinook SQLite DB
               │         (SqlSubagentMiddleware injects sql_model per-call)
               │
               ├──► analyst-agent ──► structured data analysis
               │         (AnalystSubagentMiddleware injects analyst_model)
               │
               ├──► MCP tools ──► awslabs.aws-documentation-mcp-server (stdio)
               │
               └──► PostgresStore ──► persistent cross-session memories
```

**What it explores**

- **`create_deep_agent` + `CompiledSubAgent`** — two independently compiled subagents registered with the main orchestrator and invoked as first-class tools
- **SQL subagent** — `create_agent` + `SQLDatabaseToolkit` against the bundled [Chinook](https://github.com/lerocha/chinook-database) SQLite music database; handles schema introspection, query generation, and execution autonomously
- **Analyst subagent** — receives raw query results from the SQL subagent and performs structured analysis; isolated so it can be swapped or scaled independently
- **Per-subagent model injection** — three `AgentMiddleware` subclasses (`MainAgentMiddleware`, `SqlSubagentMiddleware`, `AnalystSubagentMiddleware`) each intercept their own model call and swap in the runtime-configured Groq model, driven by `ContextSchema.main_model / sql_model / analyst_model`
- **MCP via `MultiServerMCPClient`** — connects to `awslabs.aws-documentation-mcp-server` over `stdio` transport at startup; the orchestrator can query live AWS documentation as a native tool
- **`PostgresStore`** — LangGraph's cross-session persistent store backed by PostgreSQL; agent memories survive across conversation threads and server restarts
- **`CompositeBackend`** — routes path prefixes to different backends: `FilesystemBackend` for general application files, `StoreBackend` under `/memories/` for the PostgreSQL-backed persistent store

---

## Infrastructure

All services are orchestrated by `backend/docker-compose.yaml` with Docker Compose healthchecks.

| Service | Image | Role |
|---|---|---|
| `langgraph-api` | `local-langgraph-api` | Self-hosted LangGraph API server |
| `langgraph-redis` | `redis:6` | Pub/sub and SSE streaming support for the API server |
| `langgraph-postgres` | `postgres:16` | LangGraph state checkpoints + `PostgresStore` memories |

- All four agents are registered in `agents/langgraph.json` and served as independent graph endpoints under the same API server
- The API container mounts `${BASE_PATH}/${PROJECT_NAME}` → `/home/app/agent-context/${PROJECT_NAME}`, giving the coding assistant and subagents live read/write access to a local project directory
- Fully environment-variable-driven; no secrets in source

---

## Frontend

Based on LangChain's open-source [`agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui), with substantially extended per-agent components built on top.

**Per-agent thread panels** (`src/components/thread/agents-custom-threads/`)

| Agent | Key UI Controls |
|---|---|
| Simple Agent | Model selector (Groq), memory strategy + parameters, agentic tool toggles (add/multiply/divide), workflow tool toggles (Tavily/Wikipedia), temperature & max-tokens sliders, API key input |
| Coding Assistant | Model selector (OpenAI), API key input, HITL interrupt panel (approve / edit / reject) |
| Agent with Subagents | Three independent model selectors (main / SQL / analyst), API key input, subagent tool-call rendering |
| Tools / MCP Agent | Chat input + hide-tool-calls toggle |

**Key frontend patterns**

- **Streaming** via `@langchain/langgraph-sdk`'s `useStreamContext` — tokens arrive token-by-token from the Langgraph API; `streamSubgraphs: true` surfaces subagent activity events in the message stream in real time
- **`ArtifactProvider` / `ArtifactContent`** — large outputs (file-edit proposals, data analysis, code) render in a collapsible side panel rather than inline in the message flow
- **`AgentInbox`** — dedicated HITL component that surfaces a pending `write_file` interrupt with the full `FileEditProposal` and approve / edit / reject actions
- **Thread history sidebar** — per-agent thread list with Framer Motion entrance/exit animations
- **Radix UI** primitives (Select, Switch, Tooltip, Dialog) + **Tailwind CSS v4** + **shadcn/ui**
- **React 19**, `nuqs` for URL-synced state, `zod` v4 for schema validation, `recharts` for data visualisation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.0, LangChain 1.1 |
| High-level agent abstraction | deepagents |
| LLM providers | Groq (Llama 3.3 · Qwen3 · Kimi K2 · Llama 4 · GPT-OSS), OpenAI |
| Tool integrations | LangChain Community, MCP (`langchain-mcp-adapters`) |
| Search | Tavily, Wikipedia |
| State / memory | PostgreSQL 16 (LangGraph checkpoints + `PostgresStore`) |
| Caching / streaming | Redis 6 |
| API server | self-hosted LangGraph API |
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS v4, Radix UI |
| Python package manager | uv |
| Node package manager | pnpm |
| Linting / formatting | ruff, mypy · Prettier, ESLint |
| Testing | pytest (unit + integration), LangSmith tracing |
| Containerisation | Docker Compose |

---

*This project is a personal learning exercise. LangGraph, LangChain, and all associated logos are trademarks of LangChain, Inc.*