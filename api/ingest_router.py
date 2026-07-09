"""api/ingest_router.py - POST /ingest endpoint (BUG-17 fix).

Triggers knowledge base rebuilding via API so SSH access is no longer needed.
Runs ingest as a background task and returns a job_id to poll for status.

Endpoints:
    POST /ingest            -> start ingest job, returns {job_id, status: "pending"}
    GET  /ingest/{job_id}   -> check job status
    GET  /ingest            -> list recent jobs
"""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ingest", tags=["ingest"])
_jobs: dict[str, dict] = {}


class IngestRequest(BaseModel):
    repo_path: str = "."
    extra_docs_dir: Optional[str] = None
    wipe: bool = True
    scrape_so: bool = False


class IngestStatus(BaseModel):
    job_id: str
    status: str           # "pending" | "running" | "done" | "failed"
    started_at: float
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


def _run_ingest(job_id: str, req: IngestRequest) -> None:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.build_index import build_index
    _jobs[job_id]["status"] = "running"
    try:
        result = build_index(
            repo_root=req.repo_path,
            extra_docs_dir=req.extra_docs_dir,
            wipe=req.wipe,
            scrape_so=req.scrape_so,
        )
        _jobs[job_id].update({"status": "done", "result": result, "finished_at": time.time()})
    except Exception as e:
        _jobs[job_id].update({"status": "failed", "error": str(e), "finished_at": time.time()})


@router.post("", response_model=IngestStatus, status_code=202)
async def trigger_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger an async knowledge base rebuild. Poll GET /ingest/{job_id} for status.

    Example:
        curl -X POST http://localhost:8000/ingest \\
          -H "X-API-Key: your-key" -H "Content-Type: application/json" \\
          -d '{"repo_path": ".", "wipe": true}'
    """
    from pathlib import Path
    resolved = Path(req.repo_path).resolve()
    if not resolved.exists():
        raise HTTPException(400, f"repo_path not found: {req.repo_path}")
    if not resolved.is_dir():
        raise HTTPException(400, f"repo_path is not a directory: {req.repo_path}")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id, "status": "pending",
        "started_at": time.time(), "finished_at": None,
        "result": None, "error": None,
    }
    background_tasks.add_task(_run_ingest, job_id, req)
    return IngestStatus(**_jobs[job_id])


@router.get("/{job_id}", response_model=IngestStatus)
async def get_ingest_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return IngestStatus(**_jobs[job_id])


@router.get("", response_model=list[IngestStatus])
async def list_ingest_jobs():
    jobs = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
    return [IngestStatus(**j) for j in jobs[:20]]
