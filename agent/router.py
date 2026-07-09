"""agent/router.py - Complexity-aware routing for the Q&A graph.

The eval/ numbers show two weaknesses: tool precision ~0.54 (the agent
over-calls tools) and high latency (every query pays for planning + multi-step
execution). Most lookups ("what does X do", "where is Y defined") need a single
retrieval, not a plan. This classifier sends simple queries down a cheap fast
path (one search + one synthesis) and reserves the full plan-execute-replan
loop for genuinely multi-step work (debugging, fixing, running code).

Effect: fewer spurious tool calls (higher precision) and lower latency on the
majority of queries, without weakening the agent on hard ones.
"""

from __future__ import annotations

import re

# Queries that clearly need multi-step reasoning / acting.
_COMPLEX_HINTS = (
    "debug", "fix", "error", "traceback", "exception", "stack trace",
    "why is", "why does", "reproduce", "failing", "crash", "race condition",
    "step by step", "root cause", "optimize", "refactor", "compare",
    "run ", "execute", "benchmark", "and then", "after that",
)

# Queries that are almost always single-hop lookups.
_SIMPLE_HINTS = (
    "what is", "what does", "where is", "where are", "which file",
    "list ", "show me", "define", "definition of", "how many",
    "does this", "is there", "summarize", "explain what",
)


def classify_complexity(query: str) -> str:
    """Return 'simple' (fast path) or 'complex' (full loop)."""
    q = query.lower().strip()

    if any(h in q for h in _COMPLEX_HINTS):
        return "complex"

    # Multiple questions / conjoined asks -> treat as complex.
    if q.count("?") > 1 or " and " in q and len(q.split()) > 18:
        return "complex"

    if any(q.startswith(h) or h in q for h in _SIMPLE_HINTS):
        return "simple"

    # Short, single-sentence queries default to simple; long ones to complex.
    words = len(re.findall(r"\w+", q))
    return "simple" if words <= 16 else "complex"
