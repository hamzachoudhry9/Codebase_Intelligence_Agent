"""agent/llm.py - Single place to construct the chat model.

Centralising this lets every component (Q&A graph, actor graph, router,
benchmark) share one configuration, and lets tests/CI inject a deterministic
fake so the agent logic can be exercised without a running Ollama.
"""

from __future__ import annotations

import os

_DEFAULT_MODEL = os.getenv("AGENT_MODEL", "llama3.1:8b")


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Return a ChatOllama bound to the given (or default) local model."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model or _DEFAULT_MODEL,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout=180,
    )


def count_tokens(response) -> int:
    """Best-effort token count from an Ollama response.

    langchain-ollama surfaces eval_count / prompt_eval_count in
    response_metadata when available; fall back to a chars/4 estimate.
    """
    meta = getattr(response, "response_metadata", {}) or {}
    toks = (meta.get("eval_count") or 0) + (meta.get("prompt_eval_count") or 0)
    if toks:
        return int(toks)
    text = getattr(response, "content", "") or ""
    return max(1, len(text) // 4)


class FakeLLM:
    """Deterministic stand-in used by tests and `make smoke`.

    Constructed with a list of canned responses; each `.invoke()` returns
    the next one wrapped in a minimal object exposing `.content` and
    `.response_metadata`, matching what the nodes read.
    """

    class _Resp:
        def __init__(self, content):
            self.content = content
            self.response_metadata = {"eval_count": 1, "prompt_eval_count": 1}

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0

    def invoke(self, _messages):
        if self._i >= len(self._responses):
            resp = self._responses[-1] if self._responses else ""
        else:
            resp = self._responses[self._i]
            self._i += 1
        return self._Resp(resp)
