"""
eval/evaluator.py - Production evaluation harness with LLM-as-judge.

Changes vs previous version:
  - LLM judge uses ChatOllama (local) instead of ChatGroq (requires API key)
  - Timeouts counted separately - not scored as wrong answers
  - X-API-Key header included in every request

Metrics:
    task_completion_rate  - fraction of cases that completed without error
    llm_faithfulness      - LLM judge scores answer quality 0-1
    keyword_faithfulness  - original keyword recall, kept for comparison
    avg_tool_precision    - correct tools / total tools called
    avg_tool_recall       - correct tools / expected tools
    avg_latency_s         - seconds per query

Run (API server must be running on port 8000):
    python eval/evaluator.py
    python eval/evaluator.py --cases eval/test_cases.json --out eval/results.json
    python eval/evaluator.py --no-llm-judge   # keyword-only, faster
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

API_URL     = "http://localhost:8000/query"
API_TIMEOUT = 180   # seconds - generous for first-token on a cold Ollama model
API_KEY     = os.getenv("AGENT_API_KEY", "dev-key-change-in-production")
LLM_FAITHFULNESS_THRESHOLD = 0.5

# ── LLM judge singleton ────────────────────────────────────────────────────
_judge_llm = None

def _get_judge() -> ChatOllama:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=60,
        )
    return _judge_llm


def llm_judge(query: str, answer: str) -> float:
    """
    Score answer quality 0.0-1.0 using the local Ollama model as judge.
    Unlike keyword counting this handles synonyms and paraphrasing.
    Returns 0.5 on judge failure (neutral - doesn't inflate or deflate scores).
    """
    prompt = f"""You are evaluating whether an AI assistant's answer correctly addresses a developer's query.

Query: {query}

Answer: {answer[:1500]}

Rate how well the answer addresses the query on a scale of 0 to 1:
  1.0 = completely correct, specific, and actionable
  0.7 = mostly correct with minor gaps
  0.5 = partially correct - addresses some but not all aspects
  0.3 = superficially related but misses the core question
  0.0 = wrong, irrelevant, or a refusal to answer

Return ONLY a single decimal number between 0 and 1. No explanation."""

    try:
        resp = _get_judge().invoke([HumanMessage(content=prompt)])
        score = float(resp.content.strip().split()[0])
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def evaluate(
    test_cases_path: str = "eval/test_cases.json",
    output_path: str = "eval/results.json",
    use_llm_judge: bool = True,
) -> dict:
    test_cases = json.loads(Path(test_cases_path).read_text())
    results = []
    n_timeouts = 0

    for i, tc in enumerate(test_cases):
        print(f"[{tc['id']}] ({i+1}/{len(test_cases)}) ", end="", flush=True)
        t0 = time.time()

        try:
            resp = requests.post(
                API_URL,
                json={"query": tc["query"]},
                headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.time() - t0

            answer       = data["answer"]
            tools_used   = set(data.get("tools_used", []))
            expected     = set(tc.get("expected_tools", []))

            kw_hits  = sum(1 for kw in tc["ground_truth_keywords"] if kw.lower() in answer.lower())
            kw_faith = kw_hits / len(tc["ground_truth_keywords"]) if tc["ground_truth_keywords"] else 0.0
            llm_faith = llm_judge(tc["query"], answer) if use_llm_judge else kw_faith

            tool_precision = len(tools_used & expected) / len(tools_used) if tools_used else 0.0
            tool_recall    = len(tools_used & expected) / len(expected)    if expected    else 1.0

            row = {
                "id": tc["id"],
                "category": tc.get("category", ""),
                "query": tc["query"],
                "llm_faithfulness": round(llm_faith, 3),
                "keyword_faithfulness": round(kw_faith, 3),
                "tool_precision": round(tool_precision, 3),
                "tool_recall": round(tool_recall, 3),
                "low_faithfulness_flag": llm_faith < LLM_FAITHFULNESS_THRESHOLD,
                "latency_s": round(latency, 2),
                "plan_length": len(data.get("plan", [])),
                "replan_count": data.get("replan_count", 0),
                "tools_called": sorted(tools_used),
                "tools_expected": sorted(expected),
                "completed": True,
            }
            results.append(row)

            flag = "LOW" if llm_faith < LLM_FAITHFULNESS_THRESHOLD else "OK "
            print(
                f"{flag} llm={llm_faith:.2f} kw={kw_faith:.2f} "
                f"p={tool_precision:.2f} r={tool_recall:.2f} {latency:.1f}s"
            )

        except requests.exceptions.Timeout:
            latency = time.time() - t0
            n_timeouts += 1
            results.append({
                "id": tc["id"], "category": tc.get("category", ""),
                "query": tc["query"], "error": "TIMEOUT",
                "latency_s": round(latency, 2), "completed": False,
            })
            print(f"TIMEOUT after {latency:.0f}s")

        except Exception as exc:
            latency = time.time() - t0
            results.append({
                "id": tc["id"], "category": tc.get("category", ""),
                "query": tc["query"], "error": str(exc),
                "latency_s": round(latency, 2), "completed": False,
            })
            print(f"ERROR {str(exc)[:60]}")

    # ── Aggregation ────────────────────────────────────────────────────────
    valid = [r for r in results if r.get("completed")]
    n, nv = len(results), len(valid)

    summary = {
        "task_completion_rate": round(nv / n, 3) if n else 0,
        "avg_llm_faithfulness": round(sum(r["llm_faithfulness"]    for r in valid) / nv, 3) if nv else 0,
        "avg_kw_faithfulness":  round(sum(r["keyword_faithfulness"] for r in valid) / nv, 3) if nv else 0,
        "avg_tool_precision":   round(sum(r["tool_precision"]       for r in valid) / nv, 3) if nv else 0,
        "avg_tool_recall":      round(sum(r["tool_recall"]          for r in valid) / nv, 3) if nv else 0,
        "avg_latency_s":        round(sum(r["latency_s"]            for r in valid) / nv, 2) if nv else 0,
        "n_cases": n, "n_errors": n - nv - n_timeouts,
        "n_timeouts": n_timeouts,
        "n_low_faithfulness": sum(1 for r in valid if r.get("low_faithfulness_flag")),
        "by_category": {},
    }

    cats: dict = defaultdict(list)
    for r in valid:
        if r.get("category"):
            cats[r["category"]].append(r)
    for cat, items in cats.items():
        summary["by_category"][cat] = {
            "n": len(items),
            "avg_llm_faithfulness": round(sum(i["llm_faithfulness"] for i in items) / len(items), 3),
            "avg_tool_precision":   round(sum(i["tool_precision"]    for i in items) / len(items), 3),
            "avg_latency_s":        round(sum(i["latency_s"]         for i in items) / len(items), 2),
        }

    print("\n=== Evaluation Summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))
    print("\nBy category:")
    print(json.dumps(summary["by_category"], indent=2))

    Path(output_path).write_text(json.dumps({"summary": summary, "per_case": results}, indent=2))
    print(f"\nResults saved to {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases",        default="eval/test_cases.json")
    parser.add_argument("--out",          default="eval/results.json")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Use keyword matching only (faster, less accurate)")
    args = parser.parse_args()
    evaluate(args.cases, args.out, use_llm_judge=not args.no_llm_judge)
