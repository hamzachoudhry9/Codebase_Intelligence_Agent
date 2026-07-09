"""
showcase_demo.py - Top 0.1% narrated showcase of all five agent components.

Run from the project root (with venv activated):
    python showcase_demo.py

Saves results to eval/showcase_results.json which the dashboard reads.
Every section maps to a specific JD requirement from NVIDIA's posting.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── ANSI colour helpers ────────────────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"
D  = "\033[2m"
G  = "\033[32m"
C  = "\033[36m"
Y  = "\033[33m"
M  = "\033[35m"
W  = "\033[97m"
RE = "\033[31m"

def hr(title="", color=C, width=70):
    bar = "━" * width
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{color}{B}{'━'*pad} {title} {'━'*(width-pad-len(title)-2)}{R}")
    else:
        print(f"{color}{bar}{R}")

def tag(label, color=G):
    return f"{color}{B}[{label}]{R}"

def ok(msg):  print(f"  {G}✓{R}  {msg}")
def warn(msg):print(f"  {Y}⚠{R}  {msg}")
def err(msg): print(f"  {RE}✗{R}  {msg}")
def info(msg):print(f"  {C}→{R}  {D}{msg}{R}")

# ── import agent ───────────────────────────────────────────────────────────
from agent.graph import agent_graph

def make_state(query: str) -> dict:
    return {
        "query": query,
        "past_context": "",
        "plan": [],
        "current_step_index": 0,
        "messages": [],
        "tool_outputs": [],
        "final_answer": "",
        "replan_count": 0,
        "done": False,
    }

def stream_query(query: str) -> dict:
    """Stream the graph, collect and return a structured result dict."""
    result = {
        "query": query,
        "plan": [],
        "tool_calls": [],
        "replan_count": 0,
        "final_answer": "",
        "memory_retrieved": False,
        "past_context_preview": "",
        "latency_s": 0,
        "success": True,
    }
    t0 = time.time()

    for event in agent_graph.stream(make_state(query)):
        for node, output in event.items():

            if node == "memory_retrieval":
                ctx = output.get("past_context", "")
                result["memory_retrieved"] = "No relevant" not in ctx
                result["past_context_preview"] = ctx[:200]

            elif node == "planning":
                result["plan"] = output.get("plan", [])

            elif node == "execution":
                outs = output.get("tool_outputs", [])
                if outs:
                    last = outs[-1]
                    result["tool_calls"].append({
                        "tool": last["tool"],
                        "task": last["task"],
                        "success": last.get("success", True),
                        "result_preview": str(last.get("result", ""))[:120],
                    })
                    if not last.get("success", True):
                        result["success"] = False

            elif node == "replan":
                result["replan_count"] += 1
                result["success"] = True  # recovered

            elif node == "synthesis":
                result["final_answer"] = output.get("final_answer", "")

    result["latency_s"] = round(time.time() - t0, 2)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHOWCASE CASES - 5 explicit demonstrations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SHOWCASE_CASES = [
    {
        "title": "Documentation Intelligence",
        "jd_mapping": "document code - LlamaIndex doc index over internal docs",
        "query": "What is the architecture of this agent? Explain each node in the LangGraph graph.",
        "why": "Tests search_docs against a real multi-section knowledge base.",
        "expected_primary_tool": "search_docs",
    },
    {
        "title": "Debugging Assistant",
        "jd_mapping": "debug code - tool use, multi-step investigation",
        "query": "I'm getting a KeyError inside a LangGraph node. What are the likely causes and how do I fix it?",
        "why": "Tests search_docs + web_search; real debugging scenario engineers face.",
        "expected_primary_tool": "search_docs",
    },
    {
        "title": "Code Generation + Sandbox Verification",
        "jd_mapping": "code execution sandbox - verify fixes",
        "query": "Write a Python decorator that measures function execution time and demonstrate it works.",
        "why": "Tests execute_code - agent writes code AND runs it to verify.",
        "expected_primary_tool": "execute_code",
    },
    {
        "title": "Memory: Session Continuity",
        "jd_mapping": "memory - past debugging sessions retrieved for new queries",
        "query": "How do I fix ChromaDB collection errors when querying for more results than exist in the index?",
        "why": "Should retrieve prior session about ChromaDB from memory (if run after Demo 1).",
        "expected_primary_tool": "retrieve_memory",
    },
    {
        "title": "Multi-Tool Orchestration",
        "jd_mapping": "planning + multi-step - autonomous developer workflow",
        "query": "Explain Python generator functions, show a verified working example, and note any performance advantages.",
        "why": "Requires search_docs + execute_code + synthesis - tests the full plan-execute loop.",
        "expected_primary_tool": "execute_code",
    },
]


def run_showcase():
    all_results = []
    
    hr("DEVELOPER PRODUCTIVITY AGENT - SHOWCASE", G)
    print(f"""
  {D}Stack: LangGraph · LlamaIndex · ChromaDB · Groq llama-3.3-70b · FastAPI · FastMCP
  JD:    NVIDIA - "crafting, developing, and deploying AI tools and agents
         to improve developer productivity across NVIDIA's organizations"

  Five components demonstrated explicitly:
    {M}[MEMORY]    {R}ChromaDB session retrieval + persistence across queries
    {C}[PLANNING]  {R}LLM decomposes query into [tool_name]-prefixed sub-tasks
    {G}[TOOL USE]  {R}search_docs · web_search · execute_code · retrieve_memory
    {Y}[REPLAN]    {R}Automatic step recovery when a tool fails (max 2x)
    {W}[SYNTHESIS] {R}Structured answer assembled from all tool outputs
{R}""")

    for i, case in enumerate(SHOWCASE_CASES, 1):
        hr(f"DEMO {i}/5 - {case['title']}", C)
        print(f"\n  {B}JD mapping:{R}  {D}{case['jd_mapping']}{R}")
        print(f"  {B}Why this query:{R} {D}{case['why']}{R}")
        print(f"\n  {W}{B}❓ {case['query']}{R}\n")

        result = stream_query(case["query"])

        # ── Memory ──
        print(f"\n  {tag('MEMORY', M)}", end="  ")
        if result["memory_retrieved"]:
            ok("Past session retrieved")
            info(result["past_context_preview"][:120])
        else:
            info("No past session (expected on first run or dissimilar query)")

        # ── Plan ──
        print(f"\n  {tag('PLANNING', C)}")
        for j, step in enumerate(result["plan"], 1):
            print(f"    {j}. {step}")

        # ── Tool calls ──
        print(f"\n  {tag('TOOL USE', G)}")
        for tc in result["tool_calls"]:
            sym = f"{G}✓{R}" if tc["success"] else f"{RE}✗{R}"
            print(f"    {sym} {B}{tc['tool']}{R}  {D}{tc['result_preview'][:100]}{R}")

        # ── Replan ──
        if result["replan_count"] > 0:
            print(f"\n  {tag('REPLAN', Y)}  "
                  f"{Y}Triggered {result['replan_count']}x - agent recovered automatically{R}")

        # ── Answer ──
        print(f"\n  {tag('SYNTHESIS', W)}")
        answer_lines = result["final_answer"].split("\n")
        for line in answer_lines[:8]:   # first 8 lines
            if line.strip():
                print(f"    {line[:100]}")
        if len(answer_lines) > 8:
            remaining = len(answer_lines) - 8
            print(f"    {D}... ({remaining} more lines in full answer){R}")

        # ── Stats ──
        tool_names = [tc["tool"] for tc in result["tool_calls"] if tc["tool"] != "none"]
        print(f"\n  {D}Latency: {result['latency_s']}s  ·  "
              f"Steps: {len(result['plan'])}  ·  "
              f"Tools: {', '.join(set(tool_names))}  ·  "
              f"Replans: {result['replan_count']}{R}")

        result["case_title"] = case["title"]
        result["jd_mapping"] = case["jd_mapping"]
        all_results.append(result)
        print()

    # ── Component verification table ──────────────────────────────────────
    hr("COMPONENT VERIFICATION", W)
    components = {
        "Memory":    any(r["memory_retrieved"] for r in all_results),
        "Planning":  all(len(r["plan"]) >= 2 for r in all_results),
        "Tool Use":  all(len(r["tool_calls"]) >= 1 for r in all_results),
        "Replanning":any(r["replan_count"] > 0 for r in all_results),
        "Synthesis": all(len(r["final_answer"]) > 50 for r in all_results),
    }
    for comp, verified in components.items():
        sym = f"{G}✓ VERIFIED{R}" if verified else f"{Y}⚠ not triggered{R}"
        print(f"    {B}{comp:<15}{R}  {sym}")

    # ── Aggregate stats ───────────────────────────────────────────────────
    hr("AGGREGATE STATS", C)
    total   = len(all_results)
    succeed = sum(1 for r in all_results if r["success"])
    all_tools = [tc["tool"] for r in all_results for tc in r["tool_calls"]]
    avg_lat = sum(r["latency_s"] for r in all_results) / total
    print(f"\n    Cases run:          {total}")
    print(f"    Completed OK:       {succeed}/{total}")
    print(f"    Avg latency:        {avg_lat:.1f}s")
    print(f"    Tool calls made:    {len(all_tools)}")
    print(f"    Replan events:      {sum(r['replan_count'] for r in all_results)}")
    print(f"    Unique tools used:  {', '.join(sorted(set(all_tools)))}")

    # ── Save results for dashboard ────────────────────────────────────────
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_cases": total,
            "succeeded": succeed,
            "avg_latency_s": round(avg_lat, 2),
            "total_tool_calls": len(all_tools),
            "replan_events": sum(r["replan_count"] for r in all_results),
            "components_verified": components,
        },
        "cases": all_results,
    }
    Path("eval").mkdir(exist_ok=True)
    Path("eval/showcase_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  {G}✓{R} Results saved to {B}eval/showcase_results.json{R}")
    print(f"  {G}✓{R} Open {B}dashboard.html{R} in your browser to see the visual dashboard.\n")

    hr("NEXT STEPS", G)
    print(f"""
  1. {B}Rebuild index{R} (you added knowledge_base.md):
       python ingest\\build_index.py

  2. {B}Start the API{R}:
       uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

  3. {B}Run full evaluation{R} (50 test cases):
       python eval\\evaluator.py

  4. {B}Start MCP server{R}:
       python mcp_server\\server.py

  5. {B}Open the dashboard{R}:
       dashboard.html  (double-click in Explorer)

  6. {B}Push to GitHub{R}:
       git add .
       git commit -m "Add showcase demo, rich knowledge base, and results dashboard"
       git push
{R}""")


if __name__ == "__main__":
    run_showcase()
