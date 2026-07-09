"""
agent/nodes.py — Production LangGraph nodes.

Fixes in this version:
  - ANONYMIZED_TELEMETRY silenced at module load
  - Planning prompt includes a RELEVANCE RULE to stop off-topic steps
  - past_context capped at 500 chars and only used if directly relevant
  - Ollama warmup is handled by api/main.py startup event
"""

import json
import os
import re
import time

import structlog
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from .state import AgentState
from .tools import TOOLS

log = structlog.get_logger()

# ── LLM singleton ──────────────────────────────────────────────────────────
_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url=_OLLAMA_BASE,
    timeout=120,
)

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a Codebase Intelligence Agent for a large engineering organisation.
Help engineers debug errors, understand code, and look up documentation.
Always cite the exact file name and function name when available.
Include working, copy-paste-ready code examples. Never refuse — always attempt the query."""


def _parse_tool_and_task(step: str) -> tuple:
    match = re.match(r"^\[(\w+)\]\s*(.*)", step, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, step.strip()


def _extract_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        code = parts[1] if len(parts) > 1 else text
        if code.startswith("python"):
            code = code[6:]
        return code.strip()
    return text




def memory_retrieval_node(state: AgentState) -> dict:
    from memory.session_store import get_session_store
    t0 = time.time()
    store = get_session_store()
    sessions = store.retrieve_relevant_sessions(state["query"])
    past_ctx = (
        "\n\n".join(s["summary"] for s in sessions)
        if sessions else "No relevant past sessions found."
    )
    log.info("memory_retrieval_complete",
             sessions_found=len(sessions),
             latency_ms=round((time.time() - t0) * 1000))
    return {"past_context": past_ctx}


def planning_node(state: AgentState) -> dict:
    t0 = time.time()
    query = state["query"]

    # Cap past context and only surface it if non-trivial
    past_ctx_raw = state.get("past_context", "")
    past_ctx = (
        past_ctx_raw[:600]
        if past_ctx_raw and past_ctx_raw != "No relevant past sessions found."
        else "None"
    )

    planning_prompt = f"""You are a planning agent for a Codebase Intelligence assistant.

Produce a JSON array of 2-3 sub-task strings to answer the query below.

TOOL RULES — follow exactly:
  [search_docs]     — ALWAYS use this first. Searches the ingested codebase by
                      semantic similarity. Use it for ANY question about this
                      project's code, functions, classes, or errors.
  [execute_code]    — Only use to RUN code and verify a hypothesis. Never use
                      to "check if something exists" — use search_docs for that.
  [retrieve_memory] — Only use if past sessions below are directly about this
                      exact query. Skip if past sessions are unrelated.
  [web_search]      — Only use if search_docs found NOTHING and the question is
                      about an external library (not this codebase).

RELEVANCE RULE: Every step must directly address this specific query:
  "{query}"
  Do NOT generate generic Python tutorial steps. Do NOT explain concepts
  unrelated to this query. If unsure what to do, use [search_docs].

MINIMALISM RULE: 2 steps beats 3. Only add a step if essential.

Past sessions (only relevant if they directly match the query above):
{past_ctx}

Output ONLY a valid JSON array. No preamble, no markdown, no explanation.
Query: {query}
"""
    response = llm.invoke([HumanMessage(content=planning_prompt)])
    content = response.content.strip()

    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        plan = json.loads(content)
        if not isinstance(plan, list) or not plan:
            raise ValueError
        normalized = []
        for step in plan:
            if not re.match(r"^\[\w+\]", step):
                step = f"[search_docs] {step}"
            normalized.append(step)
        plan = normalized
    except Exception:
        plan = [f"[search_docs] {query}"]

    log.info("planning_complete", n_steps=len(plan), plan=plan,
             latency_ms=round((time.time() - t0) * 1000))
    return {"plan": plan, "current_step_index": 0, "replan_count": 0,
            "tool_outputs": [], "done": False}


def execution_node(state: AgentState) -> dict:
    plan = state["plan"]
    idx  = state["current_step_index"]
    if idx >= len(plan):
        return {"done": True}

    current_step = plan[idx]
    tool_name, task_desc = _parse_tool_and_task(current_step)
    tool_outputs = list(state["tool_outputs"])
    step_success = True
    args: dict = {}
    t0 = time.time()

    if tool_name and tool_name in TOOL_MAP:
        tool_fn = TOOL_MAP[tool_name]
        try:
            if tool_name == "execute_code":
                code_prompt = (
                    f"Write Python code to accomplish:\n{task_desc}\n\n"
                    f"Context:\n{json.dumps(state['tool_outputs'][-2:], indent=2)}\n\n"
                    "Return ONLY Python code. No explanation, no markdown fences."
                )
                code_resp = llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=code_prompt),
                ])
                code = _extract_code(code_resp.content)
                result_text = str(tool_fn.invoke({"code": code}))
                args = {"code": code}
            else:
                result_text = str(tool_fn.invoke({"query": task_desc}))
                args = {"query": task_desc}

            step_success = (
                bool(result_text.strip())
                and "Code execution blocked" not in result_text
                and "Execution error" not in result_text
                and "Execution timed out" not in result_text
                and result_text != "(no output)"
                and "No relevant documentation found" not in result_text
                and "Search failed" not in result_text
            )
        except Exception as exc:
            result_text = f"Tool raised exception: {exc}"
            step_success = False
    else:
        direct_prompt = (
            f"Answer directly:\n{task_desc}\n\n"
            f"Original query: {state['query']}\n"
            f"Previous results: {json.dumps(state['tool_outputs'][-2:], indent=2)}"
        )
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=direct_prompt),
        ])
        result_text = response.content
        tool_name = "none"

    lat = round((time.time() - t0) * 1000)
    log.info("step_executed", step=idx, tool=tool_name,
             success=step_success, latency_ms=lat)

    tool_outputs.append({
        "step": idx, "task": current_step,
        "tool": tool_name or "none", "args": args,
        "result": result_text[:2000], "success": step_success,
        "latency_ms": lat,
    })
    return {"current_step_index": idx + 1, "tool_outputs": tool_outputs, "messages": []}


def replan_node(state: AgentState) -> dict:
    if state["replan_count"] >= 2:
        log.warning("replan_limit_reached")
        return {"done": True}

    current_idx   = state["current_step_index"]
    last_outputs  = state["tool_outputs"][-3:] if state["tool_outputs"] else []

    replan_prompt = (
        f"A step failed. Revise ONLY the remaining steps.\n\n"
        f"Query: {state['query']}\n"
        f"Completed: {current_idx}/{len(state['plan'])} steps\n"
        f"Failed: {state['plan'][current_idx-1] if current_idx > 0 else 'none'}\n"
        f"Recent output:\n{json.dumps(last_outputs, indent=2)}\n\n"
        "Output ONLY a JSON array. Every step MUST start with "
        "[search_docs], [web_search], [execute_code], or [retrieve_memory]. "
        "No preamble, no markdown."
    )
    response = llm.invoke([HumanMessage(content=replan_prompt)])
    content  = response.content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        new_steps = json.loads(content)
        if not isinstance(new_steps, list):
            raise ValueError
        new_steps = [
            s if re.match(r"^\[\w+\]", s) else f"[search_docs] {s}"
            for s in new_steps
        ]
    except Exception:
        return {"done": True}

    new_plan = state["plan"][:current_idx] + new_steps
    log.info("replanning_complete", replan_count=state["replan_count"] + 1)
    return {"plan": new_plan, "replan_count": state["replan_count"] + 1}


def synthesis_node(state: AgentState) -> dict:
    t0 = time.time()
    synthesis_prompt = (
        "Synthesise the investigation results into a clear, structured answer.\n"
        "Rules:\n"
        "  1. Cite the exact file name and function name when tool outputs include them.\n"
        "  2. Include working, copy-paste-ready code examples.\n"
        "  3. If a tool returned an error, acknowledge it and give your best answer.\n"
        "  4. Format: brief summary → detailed explanation → code example.\n\n"
        f"Query: {state['query']}\n\n"
        f"Tool outputs:\n{json.dumps(state['tool_outputs'], indent=2)}"
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ])
    log.info("synthesis_complete",
             latency_ms=round((time.time() - t0) * 1000),
             answer_chars=len(response.content))
    return {"final_answer": response.content, "done": True}


def save_memory_node(state: AgentState) -> dict:
    from memory.session_store import get_session_store
    tools_used = list({o["tool"] for o in state["tool_outputs"] if o["tool"] != "none"})
    try:
        store = get_session_store()
        store.save_session(
            query=state["query"],
            plan=state["plan"],
            result=state["final_answer"],
            tools_used=tools_used,
        )
        log.info("session_saved", tools_used=tools_used)
    except Exception as e:
        log.warning("session_save_failed", error=str(e))
    return {"done": True}


def should_replan(state: AgentState) -> str:
    if state.get("done"):
        return "synthesize"
    idx = state["current_step_index"]
    if idx >= len(state["plan"]):
        return "synthesize"
    if state["tool_outputs"]:
        last = state["tool_outputs"][-1]
        if not last.get("success", True) and state["replan_count"] < 2:
            return "replan"
    return "execute"


# ── Fast path + entry router (added in upgrade) ─────────────────────────────
def fast_path_node(state: AgentState) -> dict:
    """Single retrieval + synthesis for simple lookups.

    Skips planning, replanning, and multi-step execution. This is what raises
    tool precision (only search_docs is called, when appropriate) and cuts
    latency on the majority of queries that are plain lookups.
    """
    t0 = time.time()
    query = state["query"]
    search = TOOL_MAP["search_docs"]
    result_text = str(search.invoke({"query": query}))
    success = (
        bool(result_text.strip())
        and "No relevant documentation found" not in result_text
        and "Search failed" not in result_text
    )
    tool_outputs = [{
        "step": 0, "task": f"[search_docs] {query}", "tool": "search_docs",
        "args": {"query": query}, "result": result_text[:2000],
        "success": success, "latency_ms": round((time.time() - t0) * 1000),
    }]
    synth_prompt = (
        "Answer the developer's question using the retrieved code context.\n"
        "Cite exact file and function names. Add a short code example if useful.\n\n"
        f"Question: {query}\n\nRetrieved context:\n{result_text[:4000]}"
    )
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=synth_prompt),
    ])
    log.info("fast_path_complete", latency_ms=round((time.time() - t0) * 1000))
    return {
        "plan": [f"[fast_path] {query}"],
        "tool_outputs": tool_outputs,
        "current_step_index": 1,
        "replan_count": 0,
        "final_answer": resp.content,
        "done": True,
    }


def route_entry(state: AgentState) -> str:
    """Send simple lookups to the fast path, complex tasks to the full loop."""
    from agent.router import classify_complexity
    return "fast" if classify_complexity(state["query"]) == "simple" else "full"
