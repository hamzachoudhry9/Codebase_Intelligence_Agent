import json, os, sys, time, traceback
from typing import AsyncIterator

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog, uvicorn
from chroma_settings import get_chroma_client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

load_dotenv()
log = structlog.get_logger()

_API_KEY = os.getenv("AGENT_API_KEY", "dev-key-change-in-production")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

import warnings as _warnings

_AUTH_DISABLED = os.getenv("DISABLE_AUTH", "false").lower() == "true"

def verify_api_key(key: str = Security(_api_key_header)):
    if _AUTH_DISABLED:
        return
    if _API_KEY == "dev-key-change-in-production":
        _warnings.warn(
            "AGENT_API_KEY is not set. API is OPEN. "
            "Set AGENT_API_KEY in .env or set DISABLE_AUTH=true to suppress.",
            stacklevel=2,
        )
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header.")

app = FastAPI(
    title="Codebase Intelligence Agent",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000",
        ).split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent_graph = None
# B5 FIX: track warmup state so /health can expose ollama_warmed_up
_ollama_warmed = False

def get_graph():
    global _agent_graph
    if _agent_graph is None:
        from agent.graph import agent_graph
        _agent_graph = agent_graph
    return _agent_graph

@app.on_event("startup")
async def preload_components():
    global _ollama_warmed
    import httpx
    log.info("preload_graph_started")
    get_graph()
    log.info("preload_graph_complete")
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            log.info("ollama_warmup_started", model="llama3.1:8b")
            await client.post(
                f"{base}/api/generate",
                json={"model": "llama3.1:8b", "prompt": "ready", "stream": False},
            )
        _ollama_warmed = True
        log.info("ollama_warmup_complete")
    except Exception as e:
        log.warning("ollama_warmup_failed", error=str(e))
    log.info("all_components_ready")

class QueryRequest(BaseModel):
    query: str
    image_base64: str | None = None   # optional screenshot (stack trace / diagram)

class QueryResponse(BaseModel):
    answer: str
    plan: list
    tools_used: list
    replan_count: int
    latency_s: float

def _make_initial_state(query: str) -> dict:
    return {
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


def _augment_with_image(query: str, image_base64: str | None) -> str:
    """If a screenshot was supplied, extract its text and prepend it to the query.

    Lets engineers paste an image of a stack trace or diagram; the vision
    module (local VLM, OCR fallback) turns it into text the agent reasons over.
    """
    if not image_base64:
        return query
    import base64 as _b64
    import tempfile

    from agent.vision import extract_text_from_image
    try:
        raw = _b64.b64decode(image_base64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(raw)
            path = fh.name
        extracted = extract_text_from_image(path)
        log.info("image_ingested", chars=len(extracted))
        return f"{query}\n\n--- Text extracted from attached screenshot ---\n{extracted}"
    except Exception as e:
        log.warning("image_ingest_failed", error=str(e))
        return query

@app.post("/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest, _: None = Depends(verify_api_key)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    t0 = time.time()
    log.info("query_received", query=req.query[:120], has_image=bool(req.image_base64))
    effective_query = _augment_with_image(req.query, req.image_base64)
    try:
        result = get_graph().invoke(_make_initial_state(effective_query))
    except Exception as exc:
        log.error("agent_failed", error=str(exc), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")
    tools_used = list({o["tool"] for o in result.get("tool_outputs", []) if o["tool"] != "none"})
    latency = round(time.time() - t0, 2)
    log.info("query_complete", latency_s=latency, tools_used=tools_used,
             replan_count=result.get("replan_count", 0))
    return QueryResponse(
        answer=result.get("final_answer", ""),
        plan=result.get("plan", []),
        tools_used=tools_used,
        replan_count=result.get("replan_count", 0),
        latency_s=latency,
    )

@app.post("/query/stream")
async def query_stream(req: QueryRequest, _: None = Depends(verify_api_key)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    t0 = time.time()

    async def generate() -> AsyncIterator[str]:
        def emit(type_: str, data) -> str:
            return f"data: {json.dumps({'type': type_, 'data': data})}\n\n"
        try:
            # BUG-04 fix: augment with image before streaming
            _effective = _augment_with_image(req.query, req.image_base64)
            for event in get_graph().stream(_make_initial_state(_effective)):
                for node_name, output in event.items():
                    if node_name == "fast_path" and "final_answer" in output:
                        plan = output.get("plan", ["[fast_path] direct lookup"])
                        yield emit("plan", plan)
                        for to in output.get("tool_outputs", []):
                            yield emit("tool_call", {"tool": to["tool"], "task": to.get("task", "")})
                            yield emit("tool_result", {
                                "tool": to["tool"],
                                "result": to.get("result", "")[:300],
                                "success": to.get("success", True),
                            })
                        yield emit("answer", output["final_answer"])

                    elif node_name == "planning" and "plan" in output:
                        yield emit("plan", output["plan"])
                    elif node_name == "execution" and "tool_outputs" in output:
                        outs = output.get("tool_outputs", [])
                        if outs:
                            last = outs[-1]
                            yield emit("tool_call", {"tool": last["tool"], "task": last.get("task", "")})
                            yield emit("tool_result", {
                                "tool": last["tool"],
                                "result": last.get("result", "")[:300],
                                "success": last.get("success", True),
                            })
                    elif node_name == "replan":
                        yield emit("replan", {"count": output.get("replan_count", 1)})
                    elif node_name == "synthesis" and "final_answer" in output:
                        yield emit("answer", output["final_answer"])
            yield emit("done", {"latency_s": round(time.time() - t0, 2)})
        except Exception as exc:
            yield emit("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
async def health():
    # B5 FIX: include ollama_warmed_up so the UI "Warmed" indicator works
    status = {"status": "ok", "version": "2.1.0", "ollama_warmed_up": _ollama_warmed, "index": {}}
    http_status = 200
    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = get_chroma_client(path=chroma_dir)
        docs = client.get_collection("project_docs")
        mem  = client.get_or_create_collection("session_memory")
        count = docs.count()
        status["index"] = {
            "project_docs_chunks": count,
            "session_memory_sessions": mem.count(),
        }
        if count == 0:
            status["status"] = "degraded"
            status["detail"] = "Index is empty. Run /ingest or: python ingest/build_index.py --repo ."
            http_status = 503
    except Exception as e:
        status["status"] = "error"
        status["detail"] = str(e)
        http_status = 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=status, status_code=http_status)

@app.get("/sessions")
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    _: None = Depends(verify_api_key),
):
    """List past debugging sessions with pagination (BUG-23 fix)."""
    try:
        from memory.session_store import get_session_store
        store = get_session_store()
        sessions = store.list_recent_sessions(limit=limit, offset=offset)
        return {
            "sessions": sessions,
            "total": store.count(),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Ingest router (BUG-17 fix: /ingest endpoint) ────────────────────────────
from api.ingest_router import router as _ingest_router
app.include_router(_ingest_router, dependencies=[Depends(verify_api_key)])


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "dev",
        log_level="warning",
    )
