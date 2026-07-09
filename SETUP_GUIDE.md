# Complete Setup Guide: VS Code → GitHub

Follow every step in order. Do not skip ahead.

---

## Part 1 - One-time machine setup (do once)

### Step 1 - Install Python 3.11+

- **Windows:** https://www.python.org/downloads/  
  During install: check **"Add Python to PATH"**
- **macOS:** `brew install python@3.11`
- **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv python3-pip`

Verify:
```bash
python --version   # should print 3.11.x or 3.12.x
```

### Step 2 - Install Git

- **Windows:** https://git-scm.com/download/win  
- **macOS:** `brew install git`  
- **Linux:** `sudo apt install git`

Configure your identity (required for commits):
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Step 3 - Install VS Code

https://code.visualstudio.com/  
Recommended extensions to install from the Extensions panel (Ctrl+Shift+X):
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **GitLens** (optional but helpful)
- **REST Client** (lets you test your API from inside VS Code)

---

## Part 2 - Project setup

### Step 4 - Open the project in VS Code

```bash
# If you downloaded/cloned the project:
code /path/to/dev-agent

# Or: File → Open Folder → select dev-agent/
```

### Step 5 - Create and activate a virtual environment

Open the VS Code terminal (Ctrl+` on Windows/Linux, Cmd+` on macOS):

```bash
# From the dev-agent/ root:
python -m venv venv
```

Activate it:
```bash
# macOS/Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd.exe)
venv\Scripts\activate.bat
```

You should see `(venv)` in your terminal prompt.

**Select the venv interpreter in VS Code:**  
Ctrl+Shift+P → "Python: Select Interpreter" → choose `./venv/bin/python` (macOS/Linux) or `.\venv\Scripts\python.exe` (Windows)

### Step 6 - Install dependencies

```bash
pip install -r requirements.txt
```

This installs ~15 packages. It will take 2-4 minutes.

### Step 7 - Create your .env file

```bash
# macOS/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Open `.env` in VS Code and replace `your_openai_api_key_here` with your actual key:
```
OPENAI_API_KEY=sk-proj-...
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./docs
```

> **NEVER commit .env** - it is already in .gitignore but double-check before pushing.

---

## Part 3 - Build and test incrementally

### Step 8 - Build the ChromaDB index

```bash
python ingest/build_index.py
```

Expected output:
```
Loaded 1 document(s) from ./docs
Index built. ChromaDB 'project_docs' now has 12 chunk(s).
```

A `chroma_db/` directory appears in the project root. This is where all vector data lives.

### Step 9 - Test each tool in isolation

Open a Python interactive shell or create a scratch file:

```python
from dotenv import load_dotenv
load_dotenv()

from agent.tools import search_docs, web_search, execute_code, retrieve_memory

# Tool 1 - should return doc chunks
print(search_docs.invoke({"query": "FastAPI path parameters"}))

# Tool 2 - should return web snippets
print(web_search.invoke({"query": "Python LangGraph tutorial 2024"}))

# Tool 3 - should print 4
print(execute_code.invoke({"code": "print(2 + 2)"}))

# Tool 4 - will say "no sessions" on first run
print(retrieve_memory.invoke({"query": "FastAPI validation error"}))
```

**Do not proceed to Step 10 until all four tools return non-empty output.**

The most common failure at this stage:
- `search_docs` returns empty → `chroma_db/` does not exist or count = 0 → re-run `build_index.py`
- `web_search` returns empty → DuckDuckGo rate limiting → wait 30s and retry

### Step 10 - Test individual nodes

```python
from dotenv import load_dotenv
load_dotenv()

from agent.nodes import memory_retrieval_node, planning_node

dummy_state = {
    "query": "How does FastAPI handle request body validation?",
    "past_context": "",
    "plan": [],
    "current_step_index": 0,
    "messages": [],
    "tool_outputs": [],
    "final_answer": "",
    "replan_count": 0,
    "done": False,
}

# Should return {"past_context": "..."}
print(memory_retrieval_node(dummy_state))

# Should return {"plan": [...], "current_step_index": 0, ...}
dummy_state["past_context"] = "No relevant past sessions found."
print(planning_node(dummy_state))
```

### Step 11 - Run the full graph end-to-end

```python
from dotenv import load_dotenv
load_dotenv()

from agent.graph import agent_graph

result = agent_graph.invoke({
    "query": "How does FastAPI handle request body validation?",
    "past_context": "",
    "plan": [],
    "current_step_index": 0,
    "messages": [],
    "tool_outputs": [],
    "final_answer": "",
    "replan_count": 0,
    "done": False,
})

print("=== PLAN ===")
for step in result["plan"]:
    print(" -", step)

print("\n=== ANSWER ===")
print(result["final_answer"])
```

This will take ~30 seconds and make several API calls. Confirm the answer is coherent before continuing.

### Step 12 - Start the FastAPI server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

In a **second terminal** (keep the server running), test with curl:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does FastAPI handle request body validation?"}'
```

Or open http://localhost:8000/docs in your browser for the interactive Swagger UI.

### Step 13 - Test the MCP server (optional)

```bash
python mcp_server/server.py
```

The server waits for MCP protocol input over stdio. To connect it to Claude Desktop:

1. Find your Claude Desktop config file:
   - **macOS:** `~/.config/claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Add:
```json
{
  "mcpServers": {
    "dev-agent": {
      "command": "python",
      "args": ["/absolute/path/to/dev-agent/mcp_server/server.py"]
    }
  }
}
```

3. Restart Claude Desktop. You should see "dev-agent" in the MCP tools panel.

### Step 14 - Run evaluation

With the FastAPI server running on port 8000:
```bash
python eval/evaluator.py
```

This runs all 50 test cases, prints per-case results, and saves `eval/results.json`.  
Fill in your actual numbers from the summary into the README resume bullets.

---

## Part 4 - Publish to GitHub

### Step 15 - Create a GitHub repository

1. Go to https://github.com/new
2. Repository name: `dev-agent`
3. Description: `LangGraph developer productivity agent with planning, memory, tool use, and MCP`
4. Set to **Public** (so recruiters can see it)
5. **Do NOT** check "Add a README" - you already have one
6. Click **Create repository**

### Step 16 - Initialize git and make first commit

In your VS Code terminal (from the `dev-agent/` root):

```bash
git init
git add .
git status   # verify .env is NOT listed (should be ignored)
```

> **Critical:** if `.env` appears in `git status`, stop and run `git rm --cached .env` before committing.

```bash
git commit -m "Initial commit: LangGraph developer productivity agent

- Plan-and-execute loop with replanning (LangGraph)
- ChromaDB session memory with semantic retrieval
- Four tools: search_docs, web_search, execute_code, retrieve_memory
- FastMCP server exposing all tools
- FastAPI POST /query endpoint
- 50-case evaluation harness"
```

### Step 17 - Connect to GitHub and push

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/dev-agent.git
git branch -M main
git push -u origin main
```

If prompted, authenticate with your GitHub username and a **Personal Access Token** (not your password):
- Generate one at: https://github.com/settings/tokens/new
- Scopes needed: `repo`

### Step 18 - Verify the repo on GitHub

Open https://github.com/YOUR_USERNAME/dev-agent  
Confirm:
- README renders with the architecture table
- `.env` is NOT visible (only `.env.example`)
- `chroma_db/` is NOT committed (it should be gitignored)

### Step 19 - Add GitHub repository topics (improves discoverability)

On your repo page → gear icon next to "About" → add topics:
```
python langchain langgraph llm rag chromadb llama-index mcp fastapi openai agent
```

---

## Part 5 - Keeping the project up to date

### Making changes

```bash
# After any code change:
git add .
git commit -m "Fix: lazy init retriever to prevent import crash"
git push
```

### Pinning the eval results

After running `eval/evaluator.py`, fill your actual numbers into README.md, then commit:
```bash
git add README.md eval/results.json
git commit -m "Add evaluation results: X% completion, X% faithfulness"
git push
```

> Recruiters and interviewers will look for actual numbers. "Evaluated on 50 queries" without metrics is weak. "Achieved 84% task completion and 91% tool precision across 50 queries" is strong.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CollectionNotFoundError` | Run `python ingest/build_index.py` first |
| `search_docs` returns empty | Verify `chroma_db/` exists and `count > 0` |
| `openai.AuthenticationError` | Check `.env` has a valid `OPENAI_API_KEY` |
| `ModuleNotFoundError` | Activate venv: `source venv/bin/activate` |
| `uvicorn` port in use | Change port: `--port 8001` |
| Web search returns empty | DuckDuckGo rate limiting - wait 30s, retry |
| Graph loops infinitely | Check `should_replan` returns `"synthesize"` when `idx >= len(plan)` |
| `.env` committed accidentally | `git rm --cached .env && git commit -m "Remove .env from tracking"` |
