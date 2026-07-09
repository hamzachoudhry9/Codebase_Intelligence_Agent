"""
mcp_server/server.py - FastMCP server exposing agent tools as MCP primitives.

Compatible with any MCP client: Cursor, VS Code + Continue, Zed, Claude Desktop.

── HOW TO CONNECT ───────────────────────────────────────────────────────────

CURSOR  (.cursor/mcp.json in project root  OR  ~/.cursor/mcp.json globally):
{
  "mcpServers": {
    "codebase-intelligence": {
      "command": "python",
      "args": ["C:/absolute/path/to/project/mcp_server/server.py"]
    }
  }
}

VS CODE + Continue (~/.continue/config.json):
{
  "mcpServers": [
    {
      "name": "codebase-intelligence",
      "command": "python",
      "args": ["C:/absolute/path/to/project/mcp_server/server.py"]
    }
  ]
}

CLAUDE DESKTOP
  macOS  : ~/Library/Application Support/Claude/claude_desktop_config.json
  Windows: %APPDATA%/Claude/claude_desktop_config.json
  Linux  : ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "codebase-intelligence": {
      "command": "python",
      "args": ["C:/absolute/path/to/project/mcp_server/server.py"]
    }
  }
}

── STANDALONE TEST ───────────────────────────────────────────────────────────
    python mcp_server/server.py
    # Waits silently on stdin - correct behaviour. Press Ctrl+C to exit.

── TOOL TEST ─────────────────────────────────────────────────────────────────
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python mcp_server/server.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agent.tools import execute_code, retrieve_memory, search_docs, web_search

load_dotenv()

mcp = FastMCP("Codebase Intelligence Agent")


@mcp.tool()
def mcp_search_docs(query: str) -> str:
    """Search the ingested codebase and documentation using semantic vector retrieval.
    Returns exact function names, file paths, and line numbers.
    Use this first for any question about the codebase."""
    return search_docs.invoke({"query": query})


@mcp.tool()
def mcp_web_search(query: str) -> str:
    """Search the web for external library docs, known bugs, and Stack Overflow answers.
    Use only when internal docs don't have the answer."""
    return web_search.invoke({"query": query})


@mcp.tool()
def mcp_execute_code(code: str) -> str:
    """Execute a Python snippet in a sandboxed subprocess (10 second timeout).
    os, sys, subprocess, shutil imports are blocked for safety.
    Returns stdout and stderr."""
    return execute_code.invoke({"code": code})


@mcp.tool()
def mcp_retrieve_memory(query: str) -> str:
    """Retrieve summaries of past debugging sessions semantically similar to this query.
    The agent learns from every session - use this to check if a bug was solved before."""
    return retrieve_memory.invoke({"query": query})


@mcp.tool()
def mcp_full_query(query: str) -> str:
    """Run the full Codebase Intelligence Agent pipeline:
    memory retrieval → planning → multi-step tool execution → synthesis.
    Returns a complete structured answer with code examples.
    Use for complex debugging or code explanation tasks."""
    api_base = os.getenv("AGENT_API_URL", "http://localhost:8000")
    api_key  = os.getenv("AGENT_API_KEY", "dev-key-change-in-production")
    try:
        import requests
        resp = requests.post(
            f"{api_base}/query",
            json={"query": query},
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "No answer returned")
    except Exception as e:
        return (
            f"Agent query failed: {e}\n"
            "Make sure the API server is running:\n"
            "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
