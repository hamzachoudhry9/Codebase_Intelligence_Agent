# Codebase Intelligence Agent

A fully local agent that ingests any software repository, understands it at the function level, and answers developer questions conversationally. No cloud APIs. No data leaves the machine.

**Q&A eval results (50 test cases):**

| Metric | Score |
|--------|-------|
| Task completion | **100%** (50/50, 0 errors, 0 timeouts) |
| LLM faithfulness | **0.856** |
| Keyword faithfulness | 0.769 |
| Tool precision | 0.543 |
| Tool recall | 0.770 |
| Avg latency | **30s** (first query) / **22s** (subsequent) |

**Benchmark results (5 tasks, execution-based):**

| Model | Resolve rate | Solved | bugfix | mm_debug | testgen | Tool prec. | Latency (s) |
|-------|-------------|--------|--------|----------|---------|-----------|-------------|
| `oracle` | **100%** | 5/5 | 1.0 | 1.0 | 1.0 | 1.0 | 2.2 |
| `llama3.1:8b` | **0%** | 0/5 | 0.0 | 0.0 | 0.0 | 0.75 | 42.0 |

The 0% resolve rate for llama3.1:8b on agentic bug-fixing tasks is expected and honest. A 7-8B model editing files and passing tests end-to-end is a hard bar. The oracle row (gold patches applied directly) confirms the harness is correct and all 5 tasks are solvable.

---

## What is this

The project is two things:

1. **A codebase Q&A agent.** You ask questions about a repository in plain English. The agent searches the indexed codebase semantically, retrieves past sessions from memory, runs code if needed, and returns answers with exact file names, line numbers, and working code snippets.

2. **An execution-based benchmark for agentic coding tools** (`bench/`). A task is resolved only when its fail-to-pass tests pass and its pass-to-pass tests still pass after the agent acts. This is the SWE-bench methodology, runnable locally with no API keys or GPU.

---

## What it does (Q&A mode)

You type `"my CUDA kernel is giving an out-of-memory error"` and the agent:

1. Retrieves any past session where this error was solved
2. Decides which tools to use and in what order
3. Searches the codebase semantically, runs code, searches the web if needed
4. Replans up to 2 times if a tool step fails
5. Returns an answer with exact file names, function names, and working code
6. Saves the session to memory so repeat questions are faster

### How it differs from basic RAG tools

| Capability | Most RAG tools | This agent |
|-----------|---------------|------------|
| Code understanding | Document-level | Function-level (tree-sitter) |
| Memory | Per-session only | Persistent across all sessions |
| LLM provider | Cloud APIs required | 100% local via Ollama |
| IDE integration | None | MCP server (Cursor, VS Code) |
| Data privacy | Sent to cloud | Never leaves the machine |

---

## What's new: agentic SDLC system + benchmark

**Actor mode (`agent/actor.py`):** Beyond Q&A, the agent localizes a bug, patches files, runs the tests, and retries until they pass. It reads exact file contents, writes full new files, and feeds failing test names back into the next edit iteration.

**Benchmark (`bench/`):** Four categories of tasks:
- Bug fixing (3 tasks)
- Multimodal debugging -- the bug report is a screenshot of a traceback (1 task)
- Mutation-scored test generation -- generated tests must kill seeded mutants (1 task)

**Complexity router (`agent/router.py`):** Simple lookups go through a single-retrieval fast path. Multi-step tasks use the full plan-execute-replan loop. This raises tool precision and cuts latency on common queries.

**Multimodal input:** `/query` accepts an image. A screenshot of a stack trace is converted to text by a local vision model (OCR fallback) before the agent reasons over it.

**Verify it in under a minute, no GPU, no API keys:**

```bash
pip install -r requirements-bench.txt
python -m bench.selfcheck   # every seeded bug fails before, every gold patch resolves after
python -m bench.smoke       # actor loop: solve, retry-then-fix, multimodal
```

---

## Architecture

```
User Query
    |
    v
+-------------------------------------------------------------+
|                    LangGraph Agent                          |
|                                                             |
|  [route_entry]                                              |
|      |                                                      |
|   "simple" --> fast_path (1 search + synthesis)             |
|      |                                                      |
|   "complex" --> memory_retrieval --> planning               |
|                                         |                   |
|                                     execution <----------+  |
|                                         |                |  |
|                                  [should_replan?]        |  |
|                                   /    |    \            |  |
|                               exec  replan  synthesize   |  |
|                                |      +------------------+  |
|                                v                            |
|                           synthesis --> save_memory         |
+-------------------------------------------------------------+
         |                                    |
         v                                    v
   FastAPI Server                       ChromaDB
   /query (blocking)                  +--------------+
   /query/stream (SSE)                | project_docs | <- code chunks
   /health                            | session_mem  | <- past sessions
   /sessions                          | stackoverflow| <- Q&A pairs
         |                            +--------------+
         v
   MCP Server --> Cursor / VS Code / Claude Desktop
         |
         v
   Streamlit UI (localhost:8501)
```

---

## Tools

| Tool | Description | Backed by |
|------|-------------|-----------|
| `search_docs` | Semantic search over ingested codebase | ChromaDB + HuggingFace BGE |
| `web_search` | External docs and Stack Overflow | DuckDuckGo (3-attempt retry) |
| `execute_code` | Sandboxed Python subprocess | subprocess.run (10s timeout) |
| `retrieve_memory` | Past debugging sessions | ChromaDB session_memory collection |

---

## Code-aware chunking

Files are chunked at the function and class level using tree-sitter, not at fixed token counts. Each chunk carries structured metadata:

```json
{
  "file": "agent/nodes.py",
  "function": "planning_node",
  "start_line": 63,
  "end_line": 108,
  "type": "function",
  "language": "python",
  "docstring": "Generate a minimal, tool-annotated plan..."
}
```

Supported languages: Python, C, C++, CUDA (.cu, .cuh), Markdown, RST.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | [python.org](https://python.org) |
| Ollama | Latest | [ollama.com](https://ollama.com) |
| Git | Any | For cloning |
| Docker (optional) | 24+ | For production deploy |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hamzachoudhry9/codebase-intelligence-agent.git
cd codebase-intelligence-agent
```

### 2. Install Ollama and pull the model

**macOS:**
```bash
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl start ollama
ollama pull llama3.1:8b
```

**Windows:**
```
# Download from https://ollama.com/download/windows
# After installing, open Ollama from the Start menu, then in PowerShell:
ollama pull llama3.1:8b
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# macOS / Linux:
source .venv/bin/activate

# Windows PowerShell:
.venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-bench.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Defaults work for local development, no changes needed
```

### 5. Build the knowledge base

```bash
# Index this repository (source code + docs):
python ingest/build_index.py --repo . --docs docs/

# Optionally scrape Stack Overflow Q&A (cached after first run):
python ingest/build_index.py --repo . --docs docs/ --scrape-so

# Index a different codebase:
python ingest/build_index.py --repo /path/to/your/repo
```

Expected output:
```
Loading embedding model (BAAI/bge-small-en-v1.5)... Ready in 3.2s
Found 37 files in .
Total chunks: 228
  python: 120 chunks
  markdown: 108 chunks
'project_docs' -> 228 chunks
```

---

## Running the agent

### Development mode (3 terminals)

**Terminal 1 -- API server:**
```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for:
```
[info] preload_graph_complete
[info] ollama_warmup_complete
[info] all_components_ready
INFO: Application startup complete.
```

**Terminal 2 -- Streamlit UI:**
```bash
source .venv/bin/activate
streamlit run ui/app.py
# Opens at http://localhost:8501
```

**Terminal 3 (optional) -- MCP server:**
```bash
source .venv/bin/activate
python mcp_server/server.py
```

### Production mode (Docker)

```bash
docker-compose up --build

# On first run, build the knowledge base inside the container:
docker-compose exec agent python ingest/build_index.py --repo . --docs docs/
```

Services:
- API: http://localhost:8000
- UI: http://localhost:8501
- Ollama: http://localhost:11434

---

## Running the benchmark

```bash
# Validate the task suite -- no model required:
python -m bench.selfcheck

# Run the actor loop deterministically -- no model required:
python -m bench.smoke

# Harness self-test with gold patches (expect 100%):
python -m bench.run_bench --oracle

# Run against a real local model:
python -m bench.run_bench --models llama3.1:8b

# Generate the leaderboard:
python -m bench.leaderboard
```

On Windows, `make` is not available. Use `python -m bench.selfcheck` etc. directly.

---

## API reference

### POST /query

Blocking endpoint. Returns the complete result.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "What does the planning_node function do?"}'
```

Response:
```json
{
  "answer": "planning_node is in agent/nodes.py (lines 63-108)...",
  "plan": ["[search_docs] What does the planning_node function do?"],
  "tools_used": ["search_docs"],
  "replan_count": 0,
  "latency_s": 22.3
}
```

### POST /query/stream

Server-Sent Events. Returns events as the agent works.

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "Debug a CUDA out-of-memory error"}' \
  --no-buffer
```

Event types:
```
data: {"type": "plan",        "data": ["[search_docs] ...", "[execute_code] ..."]}
data: {"type": "tool_call",   "data": {"tool": "search_docs", "task": "..."}}
data: {"type": "tool_result", "data": {"tool": "search_docs", "result": "...", "success": true}}
data: {"type": "replan",      "data": {"count": 1}}
data: {"type": "answer",      "data": "Full answer text..."}
data: {"type": "done",        "data": {"latency_s": 22.3}}
```

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "2.1.0",
  "ollama_warmed_up": true,
  "index": {
    "project_docs_chunks": 228,
    "session_memory_sessions": 12
  }
}
```

### GET /sessions

Returns the 20 most recent memory sessions.

```bash
curl http://localhost:8000/sessions \
  -H "X-API-Key: dev-key-change-in-production"
```

---

## IDE integration via MCP

### Cursor

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "codebase-intelligence": {
      "command": "python",
      "args": ["/absolute/path/to/project/mcp_server/server.py"]
    }
  }
}
```

Restart Cursor. The tools appear in the tool panel.

### VS Code + Continue extension

Add to `~/.continue/config.json`:

```json
{
  "mcpServers": [
    {
      "name": "codebase-intelligence",
      "command": "python",
      "args": ["/absolute/path/to/project/mcp_server/server.py"]
    }
  ]
}
```

### Available MCP tools

| Tool | Description |
|------|-------------|
| `mcp_search_docs` | Semantic search over ingested codebase |
| `mcp_web_search` | DuckDuckGo web search |
| `mcp_execute_code` | Sandboxed Python execution |
| `mcp_retrieve_memory` | Past session retrieval |
| `mcp_full_query` | Full agent pipeline via API |

---

## Running evaluations (Q&A harness)

```bash
# API server must be running
python eval/evaluator.py --cases eval/test_cases.json --out eval/results.json

# Keyword-only mode (faster):
python eval/evaluator.py --no-llm-judge
```

### Results (50 test cases, 3 categories)

```
=== Evaluation Summary ===
task_completion_rate : 1.000   (50/50)
avg_llm_faithfulness : 0.856
avg_kw_faithfulness  : 0.769
avg_tool_precision   : 0.543
avg_tool_recall      : 0.770
avg_latency_s        : 30.02
n_errors             : 0
n_timeouts           : 0
n_low_faithfulness   : 0

By category:
  documentation_lookup             n=24  faithfulness=0.85  precision=0.729
  code_generation_and_verification n=20  faithfulness=0.85  precision=0.217
  debugging                        n=6   faithfulness=0.90  precision=0.889
```

Tool precision on code generation (0.217) is low because the agent over-called tools speculatively on those tasks. The complexity router introduced in this version reduces that by sending simple queries through a single-retrieval fast path instead.

---

## Adding your own codebase

```bash
python ingest/build_index.py \
  --repo /path/to/your/repo \
  --docs /path/to/your/docs

# Append without wiping the existing index:
python ingest/build_index.py --repo . --no-wipe
```

---

## Project structure

```
.
+-- agent/
|   +-- graph.py          # LangGraph StateGraph with fast path + full loop
|   +-- nodes.py          # memory, planning, execution, replan, synthesis, save
|   +-- actor.py          # perceive -> localize -> edit -> test -> retry loop
|   +-- actor_tools.py    # workspace-scoped read/write/list + run_tests
|   +-- router.py         # complexity classifier (simple vs complex)
|   +-- vision.py         # image to text (local VLM, OCR fallback)
|   +-- llm.py            # model factory + FakeLLM for CI
|   +-- state.py          # AgentState TypedDict
|   +-- tools.py          # search_docs, web_search, execute_code, retrieve_memory
+-- api/
|   +-- main.py           # FastAPI server (blocking + streaming + multimodal)
+-- bench/
|   +-- selfcheck.py      # validates task suite without a model
|   +-- smoke.py          # exercises actor loop with FakeLLM
|   +-- run_bench.py      # runs benchmark against Ollama models
|   +-- leaderboard.py    # generates LEADERBOARD.md
|   +-- tasks/            # 5 benchmark tasks across 4 categories
|   +-- runs/             # JSON results per model
+-- ingest/
|   +-- build_index.py    # knowledge base builder (code + docs + Stack Overflow)
|   +-- code_chunker.py   # tree-sitter chunking (Python, C++, CUDA, Markdown)
+-- memory/
|   +-- session_store.py  # ChromaDB session memory with BGE embeddings
+-- mcp_server/
|   +-- server.py         # FastMCP server (5 tools for Cursor / VS Code)
+-- ui/
|   +-- app.py            # Streamlit interface with SSE streaming
+-- eval/
|   +-- evaluator.py      # LLM-as-judge + keyword evaluation harness
|   +-- test_cases.json   # 50 test cases across 3 categories
+-- docs/                 # Documentation indexed into the knowledge base
+-- Dockerfile            # API server container
+-- Dockerfile.ui         # Streamlit container
+-- docker-compose.yml    # Ollama + agent + UI
+-- requirements.txt      # Core dependencies
+-- requirements-bench.txt # pytest + Pillow for benchmark
+-- .env.example          # Environment variable template
```

---

## Technical decisions

**Why Ollama instead of OpenAI/Anthropic?**
Production codebases cannot go to external APIs. Ollama runs llama3.1:8b on-device. No tokens leave the machine.

**Why direct ChromaDB queries instead of LlamaIndex?**
The index uses raw ChromaDB inserts with custom metadata. LlamaIndex's ChromaVectorStore wrapper expects its own internal node format and returns 0 results against externally-inserted embeddings. Direct queries are simpler and always work.

**Why tree-sitter instead of fixed-token chunking?**
Fixed-token chunking splits functions in the middle. A search for `planning_node` can return the bottom half of `execution_node` and the top half of `planning_node`, neither of which is complete. Tree-sitter parses the AST and chunks at function/class boundaries so every chunk is a complete, usable unit.

**Why execution-based scoring in the benchmark?**
LLM-as-judge scores answer quality. The benchmark scores whether the code actually compiles and passes tests. A task is resolved only when its fail-to-pass tests pass and its pass-to-pass tests still pass. No judge involved.

**Why is the benchmark resolve rate 0% for llama3.1:8b?**
Patching files correctly and passing tests end-to-end is harder than answering questions. The oracle row (gold patches applied directly) shows the harness works and all tasks are solvable. A larger or code-specialized model would score higher.

---

## License

AGPL-3.0
