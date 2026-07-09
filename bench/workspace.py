"""bench/workspace.py - Isolated, disposable copy of a task's code.

Every benchmark run gets a fresh temp copy of the task's src/ and tests/.
The agent edits the copy; the originals under bench/tasks/ are never touched.
This is what makes runs reproducible and re-runnable.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .schema import Task


@contextmanager
def workspace(task: Task):
    """Yield a temp dir containing src/ and tests/ copied from the task.

    Layout inside the workspace mirrors the task so pytest node ids and
    imports resolve identically:

        <ws>/src/...
        <ws>/tests/...
    """
    ws = Path(tempfile.mkdtemp(prefix=f"bench_{task.id}_"))
    try:
        shutil.copytree(task.src_dir, ws / "src")
        shutil.copytree(task.tests_dir, ws / "tests")
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def apply_gold(task: Task, ws: Path) -> None:
    """Copy the reference solution over the workspace src/ (selfcheck only)."""
    if not task.gold_dir.is_dir():
        raise FileNotFoundError(f"{task.id}: no gold/ dir to apply")
    for f in task.gold_dir.rglob("*.py"):
        rel = f.relative_to(task.gold_dir)
        dest = ws / "src" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
