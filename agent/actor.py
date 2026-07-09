"""agent/actor.py - The actor agent: turns an issue into a verified patch.

Flow:  perceive -> localize -> edit -> test -> (retry edit | stop)

Fixes in this version vs the original:
  FIX-1  _json_block: 3-strategy extractor handles literal newlines in JSON
         string values - the #1 failure mode for 8B models.
  FIX-2  edit_node: structured prompt that explicitly forbids multi-line
         strings and shows the exact escape format. 8B models follow examples.
  FIX-3  edit_node: on JSON parse failure, log the raw response AND tell the
         model on the next retry that the format was wrong (not that the code
         was wrong). Prevents the "silent no-op" loop.
  FIX-4  localize_node: fallback uses os.sep-agnostic check so it works on
         Windows (src\\file.py path separator fix).
  FIX-5  list_repo: always returns forward-slash paths so the model's JSON
         responses match regardless of OS.
  FIX-6  expected_tools in task.json: list_repo is an actor tool -
         run_bench.py task dicts need it added (see bench/tasks/*/task.json).
  FIX-7  testgen: handled in run_bench.py - see that file.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from . import actor_tools as fs
from .llm import count_tokens


class ActorState(TypedDict, total=False):
    task: dict
    workspace: str
    issue_text: str
    repo_tree: str
    files: dict           # rel_path -> current contents the model has seen
    iters: int
    max_iters: int
    results: dict         # test_name -> passed
    resolved: bool
    tools_used: Annotated[list, lambda a, b: a + b]   # append-only log
    tokens: int
    last_parse_error: str  # FIX-3: track JSON parse failures for retry feedback


# ── JSON extraction ───────────────────────────────────────────────────────────

def _normalize_json_strings(text: str) -> str:
    """FIX-1: Escape literal control characters inside JSON string values.

    llama3.1:8b (and most 8B models) frequently return Python file contents
    with ACTUAL newline characters inside JSON string values instead of the
    required \\n escape sequences. This causes json.loads() to raise
    JSONDecodeError: Invalid control character.

    This function fixes that by scanning character-by-character, tracking
    whether we're inside a JSON string, and escaping literal \\n \\r \\t.
    """
    result: list[str] = []
    in_string = False
    escaped = False
    bs = chr(92)  # backslash - avoids triggering linters

    for ch in text:
        if escaped:
            result.append(ch)
            escaped = False
            continue
        if ch == bs and in_string:
            result.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == '\n':
                result.append(bs + 'n')
            elif ch == '\r':
                result.append(bs + 'r')
            elif ch == '\t':
                result.append(bs + 't')
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def _json_block(text: str):
    """FIX-1: Multi-strategy JSON extractor tolerating 8B model formatting.

    Strategy 1: Strip markdown fences, try direct parse + normalized parse.
    Strategy 2: Depth-tracking bracket scan, try direct + normalized parse.
    Strategy 3: Greedy regex fallback (original behaviour), normalized.

    Raises ValueError if all strategies fail.
    """
    text = text.strip()

    # Strategy 1: strip markdown fences
    fence_content = text
    if "```" in text:
        m = re.search(r"```(?:json|python)?\s*([\s\S]*?)```", text, re.DOTALL)
        if m:
            fence_content = m.group(1).strip()
            for candidate in [fence_content, _normalize_json_strings(fence_content)]:
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, Exception):
                    pass

    # Strategy 2: bracket-depth scanning (works for nested structures)
    sources = [text] if text == fence_content else [text, fence_content]
    for source in sources:
        for start_ch, end_ch in [('{', '}'), ('[', ']')]:
            start = source.find(start_ch)
            if start == -1:
                continue
            depth, in_str, esc = 0, False, False
            for i, ch in enumerate(source[start:], start):
                if esc:
                    esc = False
                    continue
                if ch == chr(92) and in_str:
                    esc = True
                    continue
                if ch == '"' and not esc:
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == start_ch:
                        depth += 1
                    elif ch == end_ch:
                        depth -= 1
                        if depth == 0:
                            cand = source[start: i + 1]
                            for c in [cand, _normalize_json_strings(cand)]:
                                try:
                                    return json.loads(c)
                                except (json.JSONDecodeError, Exception):
                                    pass
                            break  # try next start_ch

    raise ValueError(f"Could not extract JSON from model response. First 200 chars: {text[:200]!r}")


# ── Node helpers ──────────────────────────────────────────────────────────────

def _cfg(config, key):
    return config["configurable"][key]


def _HM(text: str):
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=text)


def _log(msg: str) -> None:
    """Simple stderr logging so progress is visible without adding structlog dep."""
    print(f"[actor] {msg}", file=sys.stderr, flush=True)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def perceive_node(state: ActorState, config) -> dict:
    task = state["task"]
    issue = task["prompt"]
    tools = []
    image = task.get("image")
    if image:
        vision = _cfg(config, "vision")
        ws_img = Path(state["workspace"]) / "issue_image.png"
        img_path = Path(state["workspace"]) / ".." / image
        target = ws_img if ws_img.exists() else img_path
        extracted = vision(str(target))
        issue = f"{issue}\n\n--- Error shown in screenshot ---\n{extracted}"
        tools = ["read_image"]
        _log(f"perceive: extracted {len(extracted)} chars from image")
    return {
        "issue_text": issue,
        "tools_used": tools,
        "iters": 0,
        "tokens": 0,
        "files": {},
        "last_parse_error": "",
    }


def localize_node(state: ActorState, config) -> dict:
    llm = _cfg(config, "llm")
    ws = Path(state["workspace"])
    tree = fs.list_repo(ws)  # FIX-5: list_repo now returns forward-slash paths
    _log(f"localize: repo tree has {len(tree.splitlines())} files")

    prompt = (
        "You are fixing a bug in a small Python repository.\n"
        f"ISSUE:\n{state['issue_text']}\n\n"
        f"REPOSITORY FILES:\n{tree}\n\n"
        "Which source files under src/ must you read to fix this?\n"
        "Return ONLY a JSON array of file paths using forward slashes.\n"
        'Example: ["src/foo.py"]\n'
        "Do not list test files. Do not add any explanation."
    )
    resp = llm.invoke([_HM(prompt)])
    _log(f"localize raw response: {resp.content[:120]!r}")

    try:
        paths = _json_block(resp.content)
        paths = [p for p in paths if isinstance(p, str)][:5]
        # FIX-4: normalize separators so paths work on both Windows and Linux
        paths = [p.replace("\\", "/") for p in paths]
        _log(f"localize: model selected {paths}")
    except Exception as e:
        _log(f"localize: JSON parse failed ({e}), falling back to all src/ files")
        # FIX-4: use os-agnostic check for fallback - works on Windows AND Linux
        paths = [
            ln.replace("\\", "/")
            for ln in tree.splitlines()
            if ln.replace("\\", "/").startswith("src/")
        ]
        _log(f"localize fallback: {paths}")

    files = {}
    for p in paths:
        content = fs.read_file(ws, p)
        files[p] = content
        _log(f"localize: read {p} ({len(content)} chars)")

    return {
        "repo_tree": tree,
        "files": files,
        "tokens": state.get("tokens", 0) + count_tokens(resp),
        "tools_used": ["list_repo"] + ["read_file"] * len(files),
    }


def edit_node(state: ActorState, config) -> dict:
    """FIX-2 + FIX-3: Better prompt for 8B models + handle parse failures."""
    llm = _cfg(config, "llm")
    ws = Path(state["workspace"])

    file_blob = "\n\n".join(
        f"### FILE: {path}\n```python\n{content}\n```"
        for path, content in state["files"].items()
    )

    # FIX-3: If last iteration failed to parse JSON, tell the model explicitly
    parse_error_hint = ""
    if state.get("last_parse_error"):
        parse_error_hint = (
            "\n\nIMPORTANT: Your PREVIOUS response could not be parsed as JSON. "
            f"Parsing error: {state['last_parse_error'][:120]}. "
            "Make sure your response is ONLY valid JSON with NO prose before or after it. "
            "All newlines inside string values MUST be written as \\n (not actual line breaks)."
        )

    # FIX-2 + FIX-3: collection error gets distinct feedback from test failure
    feedback = ""
    if state.get("results"):
        results = state["results"]
        if "__collection_error__" in results:
            # The test file couldn't even be imported - syntax or import error in written code
            collection_msgs = [k.replace("__collection_msg__", "") 
                               for k in results if k.startswith("__collection_msg__")]
            feedback = (
                "\nCRITICAL: Your code caused a Python import/syntax error. "
                "The test file could not be collected by pytest. "
                f"Module that failed: {collection_msgs}\n"
                "Check for: syntax errors, incomplete code (truncation), bad indentation, "
                "or missing imports. Return the COMPLETE, syntactically valid file.\n"
            )
            _log(f"test feedback: collection error for {collection_msgs}")
        else:
            failing = [t for t, ok in results.items() if not ok]
            passing = [t for t, ok in results.items() if ok]
            if failing:
                feedback = (
                    f"\nThe following tests are FAILING: {failing}\n"
                    f"The following tests are PASSING: {passing}\n"
                    "Fix the code so all failing tests pass without breaking passing tests.\n"
                )

    # FIX-2: structured prompt with explicit JSON format example
    prompt = (
        "Fix the bug described below.\n"
        f"ISSUE:\n{state['issue_text']}\n"
        f"{feedback}"
        f"{parse_error_hint}\n"
        f"CURRENT FILES:\n{file_blob}\n\n"
        "OUTPUT FORMAT - follow this EXACTLY:\n"
        "Return ONLY a JSON object. The keys are file paths, the values are the COMPLETE new file contents.\n"
        "CRITICAL RULES:\n"
        "1. Use \\n for newlines inside the JSON string (not actual line breaks)\n"
        "2. Use \\ to escape any backslashes\n"
        "3. Use \\\" to escape any double quotes inside the file content\n"
        "4. Include the COMPLETE file - not a diff, not a snippet\n"
        "5. No explanation text before or after the JSON\n"
        "6. No markdown fences\n\n"
        'EXAMPLE (correct format): {"src/foo.py": "def add(a, b):\\n    return a + b\\n"}\n'
    )

    resp = llm.invoke([_HM(prompt)])
    raw = resp.content
    _log(f"edit iter={state.get('iters', 0)} raw response ({len(raw)} chars): {raw[:200]!r}")

    parse_error = ""
    edits = {}
    try:
        parsed = _json_block(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}: {str(parsed)[:80]}")
        edits = parsed
        _log(f"edit: parsed {list(edits.keys())}")
    except Exception as e:
        parse_error = str(e)
        _log(f"edit: JSON parse FAILED: {e}")
        edits = {}

    written = []
    syntax_errors = []
    new_files = dict(state["files"])
    for path, content in edits.items():
        if not isinstance(content, str):
            _log(f"edit: skipping {path} - value is {type(content).__name__}, not str")
            continue
        # FIX-B: unescape double-escaped \n (qwen2.5-coder emits \\\\n in JSON values)
        if path.endswith(".py") and "\\" + "n" in content:
            unescaped = content.replace("\\" + "n", "\n").replace("\\" + "t", "\t")
            if unescaped != content:
                _log(f"edit: unescaped double-escaped sequences in {path}")
                content = unescaped
        # FIX: validate Python syntax before writing - catches truncated model output
        if path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                err = f"SyntaxError in {path} at line {e.lineno}: {e.msg}"
                _log(f"edit: {err} - NOT writing (model output was truncated or malformed)")
                syntax_errors.append(err)
                parse_error = err
                continue
        fs.write_file(ws, path, content)
        new_files[path] = content
        written.append(path)
        _log(f"edit: wrote {path} ({len(content)} chars)")

    if not written and not parse_error and not syntax_errors:
        _log("edit: model returned empty edits dict - no files modified")
        parse_error = "Model returned an empty JSON object (no file edits)"

    return {
        "files": new_files,
        "iters": state.get("iters", 0) + 1,
        "tokens": state.get("tokens", 0) + count_tokens(resp),
        "tools_used": ["write_file"] * len(written),
        "last_parse_error": parse_error,
    }


def test_node(state: ActorState, config) -> dict:
    task = state["task"]
    ws = Path(state["workspace"])
    results = fs.run_tests(ws, task["test_files"])
    f2p_ok = all(results.get(t, False) for t in task.get("fail_to_pass", []))
    p2p_ok = all(results.get(t, False) for t in task.get("pass_to_pass", []))
    resolved = bool(f2p_ok and p2p_ok)
    _log(f"test: results={results} f2p_ok={f2p_ok} p2p_ok={p2p_ok} resolved={resolved}")
    return {
        "results": results,
        "resolved": resolved,
        "tools_used": ["run_tests"],
    }


def _route(state: ActorState) -> str:
    if state.get("resolved"):
        return "stop"
    if state.get("iters", 0) >= state.get("max_iters", 3):
        _log(f"route: budget exhausted after {state.get('iters')} iters")
        return "stop"
    return "retry"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_actor_graph():
    g = StateGraph(ActorState)
    g.add_node("perceive", perceive_node)
    g.add_node("localize", localize_node)
    g.add_node("edit", edit_node)
    g.add_node("test", test_node)
    g.set_entry_point("perceive")
    g.add_edge("perceive", "localize")
    g.add_edge("localize", "edit")
    g.add_edge("edit", "test")
    g.add_conditional_edges("test", _route, {"retry": "edit", "stop": END})
    return g.compile()


actor_graph = build_actor_graph()


def run_actor(
    task: dict,
    workspace: str,
    llm,
    vision=None,
    max_iters: int = 3,
) -> dict:
    """Drive one task to resolution (or budget exhaustion). Returns final state."""
    if vision is None:
        from .vision import extract_text_from_image as vision  # type: ignore
    init: ActorState = {
        "task": task,
        "workspace": workspace,
        "max_iters": max_iters,
        "tools_used": [],
        "last_parse_error": "",
    }
    cfg = {"configurable": {"llm": llm, "vision": vision}, "recursion_limit": 50}
    return actor_graph.invoke(init, cfg)
