# Developer Productivity Agent - Knowledge Base

## Project Overview

**Developer Productivity Agent** is a LangGraph-orchestrated autonomous AI agent
that helps engineers debug code, look up documentation, execute verification
snippets, and recall past debugging sessions. It is designed to map directly onto
the workflow of a professional software engineer: receive a question, plan a
multi-step investigation, call the right tools in the right order, replan if
something fails, and synthesise a structured answer.

---

## Project Folder Structure

```
dev-agent/
├── ingest/
│   ├── __init__.py
│   ├── loader.py          # LlamaIndex SimpleDirectoryReader wrapper
│   └── build_index.py     # One-time ChromaDB population script
├── agent/
│   ├── __init__.py
│   ├── state.py           # AgentState TypedDict - flows through every node
│   ├── tools.py           # Four @tool functions registered with the agent
│   ├── nodes.py           # LangGraph node functions + should_replan router
│   └── graph.py           # StateGraph construction and compilation
├── memory/
│   ├── __init__.py
│   └── session_store.py   # ChromaDB session_memory collection
├── mcp_server/
│   ├── __init__.py
│   └── server.py          # FastMCP server exposing all four tools via stdio
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI POST /query endpoint
├── eval/
│   ├── test_cases.json    # 50 annotated developer queries with ground truth
│   └── evaluator.py       # Evaluation harness - 3 metrics
├── docs/                  # Documentation files ingested into ChromaDB
├── chroma_db/             # Persisted vector index (gitignored)
├── requirements.txt
├── .env
└── .gitignore
```

---

## Agent Architecture - Plan-and-Execute Loop

The agent uses a **plan-and-execute** pattern with replanning, not pure ReAct.

### LangGraph Node Sequence

```
memory_retrieval → planning → execution → [should_replan] → synthesis → save_memory → END
                                  ↑              |
                                  └── replan ←───┘ (on failure, max 2x)
```

### Node Responsibilities

| Node | Input | Output | Description |
|------|-------|--------|-------------|
| `memory_retrieval` | query | past_context | Queries ChromaDB `session_memory` for past similar sessions |
| `planning` | query + past_context | plan (list of steps) | Decomposes query into 2-5 `[tool_name]`-prefixed sub-tasks |
| `execution` | plan[idx] | tool_outputs | Calls the tool named in the step prefix, records result + success flag |
| `replan` | failed step + recent outputs | revised plan | Revises remaining steps when a step fails; max 2 replans |
| `synthesis` | all tool_outputs | final_answer | Assembles structured answer from all collected results |
| `save_memory` | final state | (persists to DB) | Saves session summary to ChromaDB for future retrieval |

### AgentState Schema

```python
class AgentState(TypedDict):
    query: str                  # original user question
    past_context: str           # summaries from similar past sessions
    plan: list[str]             # e.g. ["[search_docs] ...", "[execute_code] ..."]
    current_step_index: int     # which plan step we are executing
    messages: Sequence[BaseMessage]
    tool_outputs: list[dict]    # [{step, task, tool, args, result, success}]
    final_answer: str           # synthesised response
    replan_count: int           # how many times we have replanned (max 2)
    done: bool                  # True after synthesis_node completes
```

---

## The Four Tools

### Tool 1: search_docs
- **Purpose**: Semantic similarity search over the ChromaDB `project_docs` collection
- **Implementation**: LlamaIndex `VectorStoreIndex.as_retriever(similarity_top_k=4)`
- **Embedding model**: `BAAI/bge-small-en-v1.5` (local HuggingFace, no API key)
- **Chunking**: SentenceSplitter, 512 tokens, 64-token overlap
- **Use when**: The query is about internal project documentation, APIs, patterns

### Tool 2: web_search
- **Purpose**: Live web search for external library docs, known bugs, Stack Overflow
- **Implementation**: DuckDuckGo DDGS with 3-attempt retry and exponential backoff
- **Use when**: The topic is not covered by internal docs, or you need current info

### Tool 3: execute_code
- **Purpose**: Run Python snippets to verify fixes and test hypotheses
- **Implementation**: `subprocess.run([sys.executable, "-c", code], timeout=10)`
- **Safety**: Blocks `os`, `sys`, `subprocess`, `shutil`, `eval`, `exec` imports
- **Use when**: Need to verify a code example actually works

### Tool 4: retrieve_memory
- **Purpose**: Look up summaries of past debugging sessions
- **Implementation**: ChromaDB `session_memory` collection, same BGE embedding model
- **Use when**: Always call this first - reduces redundant work on recurring problems

---

## Memory System Design

Two separate ChromaDB collections:

1. **`project_docs`** - static documentation index (built once, rebuild when docs change)
2. **`session_memory`** - dynamic session history (grows with every completed query)

Session summaries are compact by design (≤400 chars of result text) so retrieved
past context does not flood the LLM's context window.

```python
summary = (
    f"Query: {query}\n"
    f"Plan: {'; '.join(plan)}\n"
    f"Result summary: {result[:400]}"
)
```

---

## Common Python Debugging Patterns

### Pattern 1: Import errors - ModuleNotFoundError

**Symptom**: `ModuleNotFoundError: No module named 'X'`

**Causes and fixes**:
1. Package not installed → `pip install X`
2. Wrong virtual environment active → `which python` / `where python` to confirm
3. Module in wrong directory → check `sys.path` with `python -c "import sys; print(sys.path)"`
4. Circular imports → reorganise module dependencies

**Diagnostic code**:
```python
import sys
print("Python:", sys.executable)
print("Path entries:")
for p in sys.path:
    print(" ", p)
```

### Pattern 2: Type errors in Python 3.11+

**Symptom**: `TypeError: X() argument 'Y' must be Z, not W`

**Common causes**:
- Passing `str` where `bytes` expected in network/file operations
- Mixing `int` and `float` in NumPy operations
- Passing `None` where a required argument is expected

**Fix pattern**:
```python
# Instead of letting it crash, validate at the boundary
def process(data: str) -> bytes:
    if not isinstance(data, str):
        raise TypeError(f"Expected str, got {type(data).__name__}")
    return data.encode("utf-8")
```

### Pattern 3: ChromaDB collection errors

**Symptom**: `InvalidCollectionException` or collection not found

**Cause**: Trying to `get_collection()` before the collection was created.

**Fix**: Always use `get_or_create_collection()` in application code.
```python
# Wrong - crashes if collection doesn't exist yet
collection = client.get_collection("my_collection")

# Right - safe in any order
collection = client.get_or_create_collection("my_collection")
```

**Symptom**: `n_results` warning - "Number of requested results N is greater than number of elements in index M"

**Cause**: Querying for more results than documents in the collection.

**Fix**: Cap `n_results` to the collection count:
```python
n = min(top_k, collection.count())
if n == 0:
    return []
results = collection.query(query_embeddings=[emb], n_results=n)
```

### Pattern 4: LangGraph state KeyError

**Symptom**: `KeyError: 'plan'` or similar inside a node function

**Cause**: `initial_state` dict passed to `graph.invoke()` is missing required keys.

**Fix**: Always initialise all AgentState keys explicitly:
```python
initial_state = {
    "query": query,
    "past_context": "",
    "plan": [],
    "current_step_index": 0,
    "messages": [],
    "tool_outputs": [],
    "final_answer": "",
    "replan_count": 0,
    "done": False,
}
```

### Pattern 5: FastAPI request validation errors

**Symptom**: `422 Unprocessable Entity` response

**Cause**: Request body does not match the Pydantic model.

**Fix**: Add explicit error handling and check the response detail:
```python
# Client side - inspect the 422 detail
response = requests.post(url, json=payload)
if response.status_code == 422:
    print(response.json()["detail"])

# Server side - ensure your model matches the request shape
class QueryRequest(BaseModel):
    query: str  # must be a non-empty string
```

---

## Python Performance Patterns

### Use generators for large datasets
```python
# Memory-hungry: loads everything into RAM
results = [process(item) for item in large_dataset]

# Memory-efficient: processes one at a time
results = (process(item) for item in large_dataset)
```

### Profile before optimising
```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    your_function()

stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
stats.print_stats(10)  # top 10 slowest functions
```

### Avoid repeated attribute lookups in tight loops
```python
# Slow - Python looks up .append on every iteration
for item in data:
    results.append(transform(item))

# Fast - cache the method reference
append = results.append
for item in data:
    append(transform(item))
```

---

## API Design Best Practices

### FastAPI endpoint structure
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    max_results: int = 10

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    # process...
    return {"result": "..."}
```

### Health check pattern (required for production)
```python
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

### Structured error responses
```python
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": str(exc), "type": "ValidationError"}
    )
```

---

## Vector Database Concepts

### Embedding similarity
Vector databases store documents as dense floating-point vectors.
Retrieval finds documents whose vectors are closest to the query vector
by cosine similarity or dot product.

The quality of retrieval depends on:
1. **Embedding model quality** - BGE-small is compact and fast; ada-002 is stronger
2. **Chunk size** - 512 tokens with 64-token overlap balances context and precision
3. **Collection size** - more documents = richer retrieval but slower indexing

### ChromaDB persistence
```python
# Persistent client - survives process restarts
client = chromadb.PersistentClient(path="./chroma_db")

# In-memory client - fast for testing, lost on exit
client = chromadb.Client()
```

### Choosing similarity_top_k
- `top_k=1` - precise, low recall, good for exact lookups
- `top_k=4` - balanced, suitable for most RAG applications
- `top_k=10` - high recall, floods context window, increases LLM cost

---

## Evaluation Metrics

### Task completion rate
`completed_without_low_faithfulness / total_cases`
Measures whether the agent produced a substantive, keyword-faithful answer.

### Keyword faithfulness score
`keywords_found_in_answer / total_expected_keywords`
Proxy for answer faithfulness. Low score indicates the model ignored key concepts.
Note: this is keyword coverage, not a true hallucination detector.

### Tool selection precision
`(tools_used ∩ expected_tools) / tools_used`
Measures whether the agent called appropriate tools - penalises unnecessary tool calls.

### Tool selection recall
`(tools_used ∩ expected_tools) / expected_tools`
Measures whether the agent called all needed tools - penalises missed tool calls.
