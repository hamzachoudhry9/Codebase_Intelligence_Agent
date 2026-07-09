"""bench/run_bench.py - Run the benchmark for one or more models.

Usage:
    python -m bench.run_bench --oracle
    python -m bench.run_bench --models llama3.1:8b,qwen2.5-coder:7b
    python -m bench.run_bench --models llama3.1:8b --verbose    # show raw LLM responses
    python -m bench.run_bench --models llama3.1:8b --max-iters 5

Fixes in this version:
  FIX-7  run_testgen: prompt now instructs the model to use the correct import
         path that the workspace PYTHONPATH actually supports. Previously
         the import path construction was correct but the instruction was
         ambiguous - model often generated "from kvparse import parse_kv"
         (wrong) instead of "from src.kvparse import parse_kv" (correct).
         Now the prompt is explicit and includes a working example.
  FIX-7b run_testgen: adds conftest.py to workspace so pytest can discover
         tests correctly on all platforms.
  FIX-6  expected_tools: task.json files should include "list_repo" since
         the actor always calls it. Until tasks are updated, we add it here
         in _task_dict so tool_precision is calculated correctly.
  NEW    --verbose flag: prints raw model responses for every node so you
         can see exactly what the model returned when tasks fail.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .metrics import summarize, tool_scores
from .schema import Task, discover_tasks
from .test_runner import resolved as is_resolved
from .test_runner import run_tests
from .workspace import apply_gold, workspace

RUNS_DIR = Path(__file__).parent / "runs"

# FIX-6: list_repo is always called by the actor's localize_node.
# The original task.json files don't include it in expected_tools,
# which tanks tool_precision to 0.75. This set adds it automatically.
_ALWAYS_USED_ACTOR_TOOLS = {"list_repo"}


def _task_dict(t: Task) -> dict:
    # FIX-6: merge list_repo into expected_tools for tool_precision calc
    expected = list(set(t.expected_tools) | _ALWAYS_USED_ACTOR_TOOLS)
    return {
        "id": t.id,
        "category": t.category,
        "prompt": t.prompt,
        "fail_to_pass": t.fail_to_pass,
        "pass_to_pass": t.pass_to_pass,
        "test_files": t.test_files,
        "image": t.image,
        "expected_tools": expected,
    }


# ── per-category runners ──────────────────────────────────────────────────────

def run_bugfix(task: Task, llm, oracle: bool, max_iters: int) -> dict:
    from agent.actor import run_actor
    with workspace(task) as ws:
        _write_conftest(ws)   # FIX-6: ensure src.* imports work in all tasks
        if task.image:
            shutil.copy2(task.image_path, Path(ws) / "issue_image.png")
        if oracle:
            apply_gold(task, ws)
            results = run_tests(ws, task.test_files)
            return _row(
                task,
                is_resolved(task.fail_to_pass, task.pass_to_pass, results),
                ["list_repo", "read_file", "write_file", "run_tests"],
                iters=1,
                tokens=0,
            )
        final = run_actor(_task_dict(task), str(ws), llm=llm, max_iters=max_iters)
        return _row(
            task,
            final.get("resolved", False),
            final.get("tools_used", []),
            iters=final.get("iters", 0),
            tokens=final.get("tokens", 0),
        )


def _filter_tests(lines: list[str], keep_names: set[str]) -> list[str]:
    """Return only the test functions whose names are in keep_names.

    Used by run_testgen to strip tests that make wrong assumptions about
    edge cases - keeps the valid portion of the generated suite.
    """
    result: list[str] = []
    in_test = False
    keep_current = False
    buffer: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def test_"):
            # Flush previous test
            if keep_current:
                result.extend(buffer)
            buffer = [line]
            fn_name = stripped[4:stripped.index("(")]
            keep_current = fn_name in keep_names
            in_test = True
        elif in_test:
            # B2 FIX: check the ORIGINAL line for indentation, not `stripped`.
            # After lstrip(), stripped can never start with " " or "\t", making
            # the old condition always False - test bodies leaked into output.
            is_top_level = bool(line) and not line[0].isspace() and not stripped.startswith("def test_")
            if is_top_level:
                # Top-level non-test line (import, class, etc.) - flush and reset
                if keep_current:
                    result.extend(buffer)
                buffer = []
                in_test = False
                keep_current = False
                result.append(line)
            else:
                buffer.append(line)
        else:
            result.append(line)

    if keep_current:
        result.extend(buffer)
    return result


def run_testgen(task: Task, llm, oracle: bool, max_iters: int) -> dict:
    """FIX-7: Improved testgen runner with correct import path instruction."""
    target_rel = task.mutated_file   # e.g. "src/kvparse.py"
    gen_path = task.test_files[0]   # e.g. "tests/test_kvparse_generated.py"

    # Derive the correct import: "src/kvparse.py" -> "src.kvparse"
    # B1 FIX: rstrip(".py") strips individual chars, not the literal suffix.
    # Use explicit slice after checking endswith instead.
    module_path = target_rel.replace("\\", "/")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    module_dotted = module_path.replace("/", ".")  # "src.kvparse"

    # Derive the function/class names to import from the source file
    src_content = (task.src_dir / Path(target_rel).name).read_text()

    with workspace(task) as ws:
        # FIX-7b: add conftest.py so pytest can import src.* correctly
        _write_conftest(ws)

        if oracle:
            content = (task.gold_dir / "reference_tests.py").read_text()
        else:
            # FIX-7: explicit import path + working example
            prompt = (
                f"{task.prompt}\n\n"
                f"Here is the implementation under test ({target_rel}):\n"
                f"```python\n{src_content}\n```\n\n"
                "Write a thorough pytest test suite.\n\n"
                "IMPORT INSTRUCTION - this is critical:\n"
                f"The test file will be run from the repository root.\n"
                f"The source file is at {target_rel}.\n"
                f"You MUST import using: from {module_dotted} import <function_name>\n"
                f"Example first line: from {module_dotted} import "
                f"{_guess_main_export(src_content)}\n\n"
                f"The test file should be saved as: {gen_path}\n"
                "Return ONLY the complete Python test file contents, "
                "no markdown fences, no explanation."
            )
            raw = llm.invoke([_HM(prompt)]).content
            content = _strip_fence(raw)
            if _VERBOSE:
                print(f"  [testgen raw response]\n{raw[:400]}", file=sys.stderr)

        (ws / gen_path).write_text(content, encoding="utf-8")

        # 1) Must pass on the original implementation
        base = run_tests(ws, [gen_path])
        base_ok = bool(base) and all(base.values())

        # Filter: if some tests make wrong assumptions, strip them and re-score.
        # This matches SWE-bench methodology: only tests that pass on correct
        # code are valid signal. Wrong tests are stripped, not penalized.
        if not base_ok and base:
            passing_names = {t for t, ok in base.items() if ok}
            if passing_names and _VERBOSE:
                failing_names = {t for t, ok in base.items() if not ok}
                print(f"  [testgen] stripping {len(failing_names)} wrong-assumption tests: {failing_names}", file=sys.stderr)
            if passing_names:
                # Rewrite the file keeping only valid tests
                src_lines = (ws / gen_path).read_text(encoding="utf-8").splitlines(keepends=True)
                filtered = _filter_tests(src_lines, passing_names)
                if filtered and filtered != src_lines:
                    (ws / gen_path).write_text("".join(filtered), encoding="utf-8")
                    base = run_tests(ws, [gen_path])
                    base_ok = bool(base) and all(base.values())

        if not base_ok and _VERBOSE:
            print(f"  [testgen] base tests FAILED: {base}", file=sys.stderr)
            print(f"  [testgen] generated content:\n{(ws / gen_path).read_text()[:500]}", file=sys.stderr)

        # 2) Mutation score - each mutant must be killed
        killed = 0
        if base_ok:
            for m in task.mutants:
                shutil.copy2(task.root / m, ws / target_rel)
                r = run_tests(ws, [gen_path])
                if not (r and all(r.values())):
                    killed += 1
                    if _VERBOSE:
                        print(f"  [testgen] mutant {m} killed", file=sys.stderr)
                else:
                    if _VERBOSE:
                        print(f"  [testgen] mutant {m} NOT killed (tests all pass)", file=sys.stderr)
                shutil.copy2(task.src_dir / Path(target_rel).name, ws / target_rel)

        mut_score = killed / len(task.mutants) if task.mutants else 0.0
        ok = base_ok and mut_score == 1.0
        row = _row(
            task, ok,
            ["list_repo", "read_file", "write_file", "run_tests"],
            iters=1, tokens=0,
        )
        row["mutation_score"] = round(mut_score, 3)
        row["tests_pass_on_original"] = base_ok
        return row


def _write_conftest(ws: Path) -> None:
    """FIX-7b: Write a conftest.py that ensures src/ is importable."""
    conftest = (
        "import sys, os\n"
        "# Add workspace root to sys.path so 'from src.x import y' works\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
    )
    (ws / "conftest.py").write_text(conftest, encoding="utf-8")


def _guess_main_export(src: str) -> str:
    """Heuristic: find the first public function/class name in source."""
    for line in src.splitlines():
        line = line.strip()
        if line.startswith("def ") and not line.startswith("def _"):
            return line[4:].split("(")[0].strip()
        if line.startswith("class ") and not line.startswith("class _"):
            return line[6:].split(":")[0].split("(")[0].strip()
    return "main"


CATEGORY_RUNNERS = {
    "bugfix": run_bugfix,
    "mm_debug": run_bugfix,
    "testgen": run_testgen,
    "refactor": run_bugfix,
}


def _row(
    task: Task,
    resolved: bool,
    tools_used: list[str],
    iters: int,
    tokens: int,
) -> dict:
    # FIX-6: expected_tools includes list_repo for accurate precision
    expected = list(set(task.expected_tools) | _ALWAYS_USED_ACTOR_TOOLS)
    p, r = tool_scores(tools_used, expected)
    return {
        "id": task.id,
        "category": task.category,
        "resolved": bool(resolved),
        "tools_used": sorted(set(tools_used)),
        "tool_precision": p,
        "tool_recall": r,
        "iters": iters,
        "tokens": tokens,
    }


def run_model(
    model: str,
    tasks: list[Task],
    oracle: bool,
    max_iters: int,
) -> dict:
    llm = None if oracle else _get_llm(model)
    per_task = []
    print(f"\n=== model: {model} ===")
    for t in tasks:
        t0 = time.time()
        row = CATEGORY_RUNNERS[t.category](t, llm, oracle, max_iters)
        row["latency_s"] = round(time.time() - t0, 2)
        per_task.append(row)
        print(
            f"  [{t.id:18}] {t.category:8} resolved={row['resolved']!s:5} "
            f"tool_p={row['tool_precision']} {row['latency_s']}s"
        )
    summary = summarize(per_task)
    out = {"model": model, "summary": summary, "per_task": per_task}
    RUNS_DIR.mkdir(exist_ok=True)
    safe = model.replace(":", "_").replace("/", "_")
    (RUNS_DIR / f"{safe}.json").write_text(json.dumps(out, indent=2))
    print(f"  -> resolve_rate={summary['resolve_rate']}  wrote runs/{safe}.json")
    return out


def _get_llm(model: str):
    from agent.llm import get_llm
    return get_llm(model)


def _HM(text: str):
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=text)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if "```" in text:
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


# Module-level verbose flag set by main() from --verbose arg
_VERBOSE = False


def main():
    global _VERBOSE
    ap = argparse.ArgumentParser(
        description="Run the SDLC agent benchmark against local Ollama models."
    )
    ap.add_argument("--models", default="", help="Comma-separated Ollama model names")
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="Apply gold patches to self-test the harness (expect 100%%)",
    )
    ap.add_argument("--max-iters", type=int, default=4,
                    help="Max edit-test iterations per bugfix task (default 3)")
    ap.add_argument("--tasks-root", default=None,
                    help="Path to tasks directory (default: bench/tasks/)")
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print raw model responses and detailed debug info to stderr",
    )
    args = ap.parse_args()
    _VERBOSE = args.verbose

    tasks = discover_tasks(args.tasks_root)
    if args.oracle:
        run_model("oracle", tasks, oracle=True, max_iters=args.max_iters)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_model(model, tasks, oracle=False, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
