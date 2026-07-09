"""agent/vision.py - The one multimodal capability.

Developers paste screenshots of errors constantly. This turns an image of a
stack trace (or an architecture diagram) into text the rest of the pipeline
can reason over. It degrades gracefully:

    1. local vision model via Ollama (e.g. `llava`)   - best
    2. OCR via pytesseract                            - fallback
    3. a clear message                                - if neither is present

Nothing here needs the cloud; it stays consistent with the project's
fully-local design.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

VISION_MODEL = os.getenv("AGENT_VISION_MODEL", "llava")

_VLM_PROMPT = (
    "This image is a screenshot from a developer's screen - usually a stack "
    "trace, error message, or a diagram. Transcribe all text exactly, and if "
    "it is a traceback, state the exception type, message, and the file and "
    "line where it was raised. Be concise."
)


def extract_text_from_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return f"[vision] image not found: {image_path}"

    text = _try_ollama_vision(p)
    if text:
        return text

    text = _try_ocr(p)
    if text:
        return f"[OCR fallback]\n{text}"

    return ("[vision] No local vision model or OCR engine available. "
            "Install a vision model (`ollama pull llava`) or pytesseract.")


def _try_ollama_vision(p: Path) -> str | None:
    try:
        import ollama  # type: ignore
    except Exception:
        return None
    try:
        b64 = base64.b64encode(p.read_bytes()).decode()
        resp = ollama.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": _VLM_PROMPT, "images": [b64]}],
        )
        return resp["message"]["content"].strip()
    except Exception:
        return None


def _try_ocr(p: Path) -> str | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception:
        return None
    try:
        return pytesseract.image_to_string(Image.open(p)).strip() or None
    except Exception:
        return None
