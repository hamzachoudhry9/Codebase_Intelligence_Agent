import os, subprocess, sys, textwrap, time
from chroma_settings import get_chroma_client
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
_DOCS_COLLECTION = "project_docs"

_chroma_client = None

def _get_embed_model():
    """Shared embedding singleton from ingest.embed (BUG-13: eliminates double 130MB load)."""
    from ingest.embed import get_embed_model
    return get_embed_model()

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = get_chroma_client(path=_CHROMA_DIR)
    return _chroma_client

def _search_chromadb(query, collection_name, top_k=5):
    try:
        embed_model = _get_embed_model()
        client = _get_chroma_client()
        collection = client.get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []
        query_embedding = embed_model.get_text_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({"text": doc, "metadata": meta, "distance": dist})
        return items
    except Exception as exc:
        raise RuntimeError(f"ChromaDB search failed: {exc}") from exc

# _is_safe lives in agent/sandbox.py to enable lightweight importing
from agent.sandbox import _is_safe


@tool
def search_docs(query: str) -> str:
    """Search the ingested codebase using semantic similarity. Returns function names,
    file paths, line numbers, and code context. Use this FIRST for any codebase question."""
    try:
        items = _search_chromadb(query, _DOCS_COLLECTION, top_k=5)
    except RuntimeError as e:
        return f"Search failed: {e}"
    if not items:
        return "No relevant documentation found."
    parts = []
    for item in items:
        meta = item["metadata"]
        file_ = meta.get("file", "?")
        func = meta.get("function", "?")
        lang = meta.get("language", "?")
        start = meta.get("start_line", "")
        end = meta.get("end_line", "")
        dist = item.get("distance", 0)
        location = f"{file_}:{func}"
        if start and end:
            location += f" (lines {start}-{end})"
        location += f" [{lang}] similarity={round(1 - dist, 2)}"
        parts.append(f"[Source: {location}]\n{item['text']}")
    return "\n\n---\n\n".join(parts)

@tool
def web_search(query: str) -> str:
    """Search the web for external library docs or known bugs not found internally.
    Use only when search_docs returned nothing relevant."""
    from duckduckgo_search import DDGS
    try:
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        RatelimitException = Exception
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
            if not results:
                return "No web results found."
            return "\n\n---\n\n".join(
                f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                for r in results
            )
        except RatelimitException:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return "Web search rate limited. Use search_docs instead."
        except Exception as e:
            return f"Web search error: {e}"
    return "Web search failed."

@tool
def execute_code(code: str) -> str:
    """Execute a Python snippet in a sandboxed subprocess (10s timeout).
    os, sys, subprocess, shutil imports are blocked for safety."""
    ok, reason = _is_safe(code)
    if not ok:
        return f"Code execution blocked: {reason}"
    try:
        res = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True, text=True, timeout=10,
            # Inherit the full environment so Python stdlib works on all platforms,
            # then override PYTHONPATH to empty to prevent the agent's own packages
            # leaking into the sandboxed subprocess.
            env={**os.environ, "PYTHONPATH": ""},
        )
        out = res.stdout or ""
        err = res.stderr or ""
        if err:
            return f"stdout:\n{out}\nstderr:\n{err}"
        return f"stdout:\n{out}" if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out after 10 seconds."
    except Exception as e:
        return f"Execution error: {e}"

@tool
def retrieve_memory(query: str) -> str:
    """Retrieve past debugging sessions similar to this query.
    Use this to check if a bug was solved before."""
    from memory.session_store import get_session_store
    store = get_session_store()
    sessions = store.retrieve_relevant_sessions(query, top_k=3)
    if not sessions:
        return "No relevant past sessions found."
    return "\n\n---\n\n".join(
        f"Past session ({s['metadata'].get('timestamp', '?')}):\n{s['summary']}"
        for s in sessions
    )

TOOLS = [search_docs, web_search, execute_code, retrieve_memory]
