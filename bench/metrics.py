"""bench/metrics.py - Scoring for benchmark runs.

The headline metric is the execution-based RESOLVE RATE (fraction of tasks
whose fail->pass and pass->pass tests all pass after the agent acts). We also
keep the project's existing tool-precision / tool-recall idea so the new
benchmark stays continuous with the eval/ harness, plus efficiency metrics
(latency, tokens) - the axis the JD's team cares about for agent workflows.
"""

from __future__ import annotations

from collections import defaultdict


def tool_scores(used: list[str], expected: list[str]) -> tuple[float, float]:
    u, e = set(used), set(expected)
    precision = len(u & e) / len(u) if u else 0.0
    recall = len(u & e) / len(e) if e else 1.0
    return round(precision, 3), round(recall, 3)


def summarize(per_task: list[dict]) -> dict:
    n = len(per_task)
    if n == 0:
        return {"n_tasks": 0}

    def avg(key):
        vals = [r[key] for r in per_task if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "n_tasks": n,
        "resolve_rate": round(sum(r["resolved"] for r in per_task) / n, 3),
        "n_resolved": sum(r["resolved"] for r in per_task),
        "avg_tool_precision": avg("tool_precision"),
        "avg_tool_recall": avg("tool_recall"),
        "avg_latency_s": avg("latency_s"),
        "avg_tokens": round(avg("tokens")),
        "avg_iters": avg("iters"),
        "by_category": {},
    }
    cats = defaultdict(list)
    for r in per_task:
        cats[r["category"]].append(r)
    for cat, items in cats.items():
        k = len(items)
        summary["by_category"][cat] = {
            "n": k,
            "resolve_rate": round(sum(i["resolved"] for i in items) / k, 3),
            "avg_latency_s": round(sum(i.get("latency_s", 0) for i in items) / k, 2),
        }
    return summary
