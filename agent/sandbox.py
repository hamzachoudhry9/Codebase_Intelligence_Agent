"""agent/sandbox.py - Code safety checker with no heavy dependencies.

Extracted from tools.py so tests can import _is_safe without triggering
the llama-index embedding model load.
"""
import re

_BLOCKED = [
    # System access
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b", r"\bfrom\s+os\b", r"\bfrom\s+sys\b",
    r"\bfrom\s+subprocess\b", r"\bfrom\s+shutil\b",
    # Code injection
    r"\b__import__\s*\(", r"\beval\s*\(", r"\bexec\s*\(",
    # Network - BUG-16: these were missing, allowing data exfiltration
    r"\bimport\s+socket\b", r"\bimport\s+http\b", r"\bimport\s+urllib\b",
    r"\bimport\s+ftplib\b", r"\bimport\s+smtplib\b",
    r"\bimport\s+requests\b", r"\bimport\s+httpx\b",
    r"\bimport\s+aiohttp\b", r"\bimport\s+websockets\b",
    r"\bfrom\s+socket\b", r"\bfrom\s+http\b", r"\bfrom\s+urllib\b",
    r"\bfrom\s+requests\b", r"\bfrom\s+httpx\b", r"\bfrom\s+aiohttp\b",
    # File system
    r"\bopen\s*\(", r"\bpathlib\b", r"\bfrom\s+pathlib\b",
]


def _is_safe(code: str) -> tuple[bool, str]:
    for p in _BLOCKED:
        if re.search(p, code):
            return False, p
    return True, ""
