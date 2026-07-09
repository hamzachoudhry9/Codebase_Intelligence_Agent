"""agent/actor_tools.py - Tools the actor uses to act on a real workspace.

Fixed in this version:
  FIX-5  list_repo: always returns POSIX (forward-slash) paths regardless of
         OS. On Windows, Path.relative_to() returns backslash paths which
         confuse both the model's localize response and the fallback path check.
"""

from __future__ import annotations

from pathlib import Path

from bench.test_runner import run_tests as _run_tests

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".venv", "venv", "node_modules"}


def _safe(root: Path, rel: str) -> Path:
    """Resolve `rel` under `root`, refusing to escape the workspace."""
    root = root.resolve()
    # Normalize separator so "src/foo.py" works on Windows too
    rel_norm = rel.replace("\\", "/")
    target = (root / rel_norm).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path '{rel}' escapes the workspace")
    return target


def list_repo(root: Path) -> str:
    """Return a newline-separated tree of source/test files.

    FIX-5: Always uses forward slashes in the returned paths, even on Windows.
    This ensures the model's JSON response uses "src/foo.py" format which
    the fallback localize path check and write_file both expect.
    """
    root = Path(root)
    lines = []
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # FIX-5: .as_posix() gives forward slashes on all platforms
        lines.append(p.relative_to(root).as_posix())
    return "\n".join(lines) if lines else "(empty)"


def read_file(root: Path, rel: str) -> str:
    p = _safe(Path(root), rel)
    if not p.exists():
        return f"ERROR: file not found: {rel}"
    return p.read_text(encoding="utf-8", errors="replace")


def write_file(root: Path, rel: str, content: str) -> str:
    p = _safe(Path(root), rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {rel} ({len(content)} chars)"


def run_tests(root: Path, test_files: list[str]) -> dict[str, bool]:
    return _run_tests(Path(root), test_files)
