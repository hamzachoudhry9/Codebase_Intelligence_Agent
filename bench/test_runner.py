"""bench/test_runner.py - Run pytest in a workspace, report per-test status.

We run the task's test files once under a workspace, emit a JUnit XML
report, and parse it into {test_name: passed}. Test *function names* are
unique within a task (enforced by task design), so keying by name avoids
all the path/package fragility of matching full node ids across machines.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def run_tests(ws: Path, test_files: list[str], timeout: int = 120) -> dict[str, bool]:
    """Return {test_function_name: passed_bool} for every collected test.

    `test_files` are paths relative to the workspace, e.g. "tests/test_x.py".
    A test counts as passed only if it neither failed nor errored.
    """
    xml_path = Path(tempfile.mktemp(suffix=".xml"))
    cache = Path(tempfile.mkdtemp(prefix="pytest_cache_"))
    targets = [str(ws / tf) for tf in test_files]

    cmd = [
        sys.executable, "-m", "pytest",
        *targets,
        f"--junitxml={xml_path}",
        "-o", f"cache_dir={cache}",
        "-p", "no:cacheprovider",
        "-q", "--no-header", "--tb=no",
    ]
    try:
        subprocess.run(
            cmd, cwd=str(ws), capture_output=True, text=True, timeout=timeout,
            # Make `import src...` resolve from the workspace root.
            env=_env(ws),
        )
    except subprocess.TimeoutExpired:
        return {}  # treated as "nothing passed" by callers

    if not xml_path.exists():
        return {}

    results: dict[str, bool] = {}
    tree = ET.parse(xml_path)
    for case in tree.iter("testcase"):
        name = case.attrib.get("name", "")
        classname = case.attrib.get("classname", "")
        failed = any(child.tag in ("failure", "error") for child in case)
        # Detect pytest collection errors: name looks like a module path (no "test_")
        # e.g. name="tests.test_retry" classname="" means the file failed to import
        is_collection_error = (
            not name.startswith("test_")
            and "." in name
            and classname == ""
            and failed
        )
        if is_collection_error:
            # Extract error message for better feedback
            for child in case:
                if child.tag in ("failure", "error"):
                    results["__collection_error__"] = False
                    results[f"__collection_msg__{name}"] = False
                    break
        else:
            results[name] = not failed
    xml_path.unlink(missing_ok=True)
    return results


def _env(ws: Path) -> dict:
    import os
    env = dict(os.environ)
    # Workspace root on PYTHONPATH so `from src.x import y` works.
    env["PYTHONPATH"] = str(ws) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def resolved(fail_to_pass: list[str], pass_to_pass: list[str],
             results: dict[str, bool]) -> bool:
    """A task is resolved iff every fail->pass test passes AND every
    pass->pass test still passes."""
    f2p_ok = all(results.get(t, False) for t in fail_to_pass)
    p2p_ok = all(results.get(t, False) for t in pass_to_pass)
    return f2p_ok and p2p_ok
