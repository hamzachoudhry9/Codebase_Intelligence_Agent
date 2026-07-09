"""
ui/app.py - Codebase Intelligence Agent interface.
Run: streamlit run ui/app.py
"""

import json
import os
import time

import requests
import streamlit as st
# B7 FIX: load_dotenv() was missing - without it, .env is ignored and the UI
# sends the wrong API key (403 on every request) when vars aren't shell-exported.
from dotenv import load_dotenv
load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("AGENT_API_KEY", "dev-key-change-in-production")
HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(
    page_title="CIA - Codebase Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    box-sizing: border-box;
}

/* ── Base ── */
.stApp, .main, section[data-testid="stSidebar"] {
    background: #0C0F18 !important;
}
.stApp { color: #C9D1E0; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid #1A2035 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.25rem; }

/* ── Inputs ── */
.stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 6px !important;
    color: #C9D1E0 !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 14px !important;
}
.stTextArea textarea:focus {
    border-color: #3B7FEF !important;
    box-shadow: 0 0 0 2px rgba(59,127,239,0.12) !important;
}
.stTextArea label { color: #64748B !important; font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: #3B7FEF !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: background 0.15s, transform 0.1s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: #2563EB !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #131825 !important;
    border: 1px solid #1E2D45 !important;
    color: #94A3B8 !important;
    font-size: 12px !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #1A2540 !important;
    border-color: #3B7FEF !important;
    color: #C9D1E0 !important;
}

/* ── Dividers ── */
hr { border-color: #1A2035 !important; margin: 1rem 0 !important; }

/* ── Status ── */
.stStatus { background: #111827 !important; border: 1px solid #1E2D45 !important; border-radius: 8px !important; }

/* ── Captions / text ── */
.stCaption, [data-testid="stCaptionContainer"] { color: #475569 !important; font-size: 11px !important; }

/* ── Custom components ── */
.cia-wordmark {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1.5rem 0 1rem 0;
    margin-bottom: 0.25rem;
}
.cia-mark {
    width: 32px; height: 32px;
    background: #3B7FEF;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700; color: #fff;
    font-family: 'Exo 2', sans-serif;
    letter-spacing: -0.05em;
    flex-shrink: 0;
}
.cia-name {
    font-size: 13px; font-weight: 600;
    color: #E2E8F0; letter-spacing: 0.01em; line-height: 1.2;
}
.cia-sub {
    font-size: 10px; color: #475569;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1px;
}

/* Status badge */
.status-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 500;
    padding: 3px 9px; border-radius: 99px;
    font-family: 'JetBrains Mono', monospace;
}
.status-ok   { background: #0D2818; color: #4ADE80; border: 1px solid #14532D; }
.status-err  { background: #290D0D; color: #F87171; border: 1px solid #7F1D1D; }

/* Index stat row */
.idx-row {
    display: flex; gap: 8px; margin: 10px 0;
}
.idx-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1A2035;
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
}
.idx-val {
    font-size: 18px; font-weight: 600;
    color: #3B7FEF;
    font-family: 'JetBrains Mono', monospace;
    display: block;
}
.idx-lbl {
    font-size: 9px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em;
    display: block; margin-top: 2px;
}

/* Tool pill */
.tool-pill {
    display: inline-flex; align-items: center; gap: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500;
    padding: 2px 8px; border-radius: 4px;
    margin-right: 4px;
}
.tp-search_docs     { background:#0F2541; color:#60A5FA; border:1px solid #1E3A5F; }
.tp-web_search      { background:#27180A; color:#FB923C; border:1px solid #4A2E12; }
.tp-execute_code    { background:#1A0F2E; color:#A78BFA; border:1px solid #2D1B54; }
.tp-retrieve_memory { background:#0D2318; color:#34D399; border:1px solid #134E2B; }
.tp-none            { background:#1A1A2E; color:#64748B; border:1px solid #252540; }

/* Execution trace */
.trace-wrap { margin: 12px 0; }
.trace-row {
    display: flex; gap: 12px; align-items: flex-start;
    margin-bottom: 6px;
}
.trace-line {
    display: flex; flex-direction: column; align-items: center;
    flex-shrink: 0; margin-top: 3px;
}
.trace-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.trace-connector {
    width: 1px; flex: 1; min-height: 20px;
    background: #1A2035; margin-top: 3px;
}
.td-blue   { background: #3B7FEF; box-shadow: 0 0 6px rgba(59,127,239,.5); }
.td-orange { background: #F59E0B; box-shadow: 0 0 6px rgba(245,158,11,.4); }
.td-purple { background: #8B5CF6; box-shadow: 0 0 6px rgba(139,92,246,.4); }
.td-green  { background: #10B981; box-shadow: 0 0 6px rgba(16,185,129,.4); }
.td-gray   { background: #374151; }
.trace-body { flex: 1; min-width: 0; }
.trace-header {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; color: #94A3B8;
    font-family: 'JetBrains Mono', monospace;
    flex-wrap: wrap;
}
.trace-task {
    font-size: 11px; color: #475569;
    margin-top: 2px; line-height: 1.4;
    word-break: break-word;
}
.trace-result {
    background: #0A0E18;
    border: 1px solid #1A2035;
    border-left: 2px solid #1E2D45;
    border-radius: 4px;
    padding: 6px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #475569;
    margin-top: 5px;
    max-height: 80px; overflow-y: auto;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
}
.success-dot { color: #10B981; }
.fail-dot    { color: #EF4444; }
.trace-ts {
    font-size: 9px; color: #374151;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0; margin-left: auto;
}

/* Plan banner */
.plan-banner {
    background: #0E1729;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 0;
}
.plan-title {
    font-size: 10px; font-weight: 600; color: #3B7FEF;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}
.plan-step {
    display: flex; gap: 8px; align-items: flex-start;
    font-size: 12px; color: #94A3B8;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 5px; line-height: 1.4;
}
.plan-num {
    font-size: 9px; color: #374151; flex-shrink: 0;
    margin-top: 2px; min-width: 16px;
}

/* Answer */
.answer-wrap {
    background: #0E1729;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 20px 22px;
    margin-top: 14px;
    color: #C9D1E0;
    font-size: 14px; line-height: 1.75;
}
.answer-label {
    font-size: 10px; font-weight: 600; color: #3B7FEF;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 12px; display: flex; align-items: center; gap: 6px;
}

/* Stats */
.stat-row { display: flex; gap: 8px; margin-bottom: 10px; }
.stat-box {
    flex: 1; background: #111827;
    border: 1px solid #1A2035; border-radius: 6px;
    padding: 10px 12px; text-align: center;
}
.stat-n {
    font-size: 20px; font-weight: 600; color: #E2E8F0;
    font-family: 'JetBrains Mono', monospace;
    display: block;
}
.stat-l {
    font-size: 9px; color: #475569; text-transform: uppercase;
    letter-spacing: 0.08em; display: block; margin-top: 2px;
}

/* Replan alert */
.replan-bar {
    background: #1C130A; border: 1px solid #3A2204;
    border-radius: 6px; padding: 8px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #F59E0B; margin: 6px 0;
}

/* Session item */
.sess-item {
    background: #111827; border: 1px solid #1A2035;
    border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
}
.sess-q {
    font-size: 11px; color: #94A3B8; line-height: 1.4;
}
.sess-meta {
    font-size: 9px; color: #374151;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 3px;
}

/* Main header */
.main-header {
    padding: 1.75rem 0 1rem 0;
    border-bottom: 1px solid #1A2035;
    margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 14px;
}
.mh-mark {
    width: 40px; height: 40px; border-radius: 8px;
    background: #3B7FEF;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700; color: #fff;
    font-family: 'Exo 2', sans-serif; letter-spacing: -0.05em;
    flex-shrink: 0;
}
.mh-title {
    font-size: 20px; font-weight: 700; color: #E2E8F0;
    letter-spacing: -0.01em; margin: 0;
}
.mh-sub {
    font-size: 11px; color: #475569;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}
.mh-badges { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
.mh-badge {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: #475569; background: #111827;
    border: 1px solid #1A2035; border-radius: 4px;
    padding: 2px 7px;
}

/* Example chips */
.ex-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="cia-wordmark">
      <div class="cia-mark">◈</div>
      <div>
        <div class="cia-name">Codebase Intelligence</div>
        <div class="cia-sub">v2.1 · Ollama · LangGraph</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Health ──
    api_ok = False
    api_degraded = False
    try:
        h = requests.get(f"{API_BASE}/health", timeout=3).json()
        status = h.get("status")
        # B4 FIX: distinguish degraded (index empty, reachable) from fully ok
        # and from offline/unreachable. NOTE: these must be computed as two
        # independent checks against `status` - gating the "degraded" check
        # behind `api_ok` (status == "ok") made it impossible to ever be
        # True, since a degraded server never reports "ok".
        api_ok = status == "ok"
        api_degraded = status == "degraded"
        idx = h.get("index", {})
        doc_chunks = idx.get("project_docs_chunks", 0)
        mem_sessions = idx.get("session_memory_sessions", 0)
        warmed = h.get("ollama_warmed_up", False)
    except Exception:
        doc_chunks = 0
        mem_sessions = 0
        warmed = False

    badge_cls = "status-ok" if api_ok else "status-err"
    badge_txt = "● API connected" if api_ok else "● API offline"
    st.markdown(f'<span class="status-badge {badge_cls}">{badge_txt}</span>',
                unsafe_allow_html=True)

    if not api_ok and not api_degraded:
        st.caption("Start: `uvicorn api.main:app --port 8000`")

    if api_degraded:
        st.warning(
            "⚠️ Index is empty - run this first:\n\n"
            "`python ingest/build_index.py`\n\n"
            "Then restart the API. Queries will return empty answers until indexed.",
            icon=None,
        )

    # ── Index stats ──
    if api_ok:
        st.markdown(f"""
        <div class="idx-row">
          <div class="idx-card">
            <span class="idx-val">{doc_chunks}</span>
            <span class="idx-lbl">Doc chunks</span>
          </div>
          <div class="idx-card">
            <span class="idx-val">{mem_sessions}</span>
            <span class="idx-lbl">Sessions</span>
          </div>
          <div class="idx-card">
            <span class="idx-val">{'✓' if warmed else '…'}</span>
            <span class="idx-lbl">Warmed</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tools ──
    st.markdown('<div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;font-family:JetBrains Mono,monospace">Available tools</div>', unsafe_allow_html=True)
    tools = [
        ("search_docs",     "tp-search_docs",     "Semantic codebase retrieval"),
        ("web_search",      "tp-web_search",      "DuckDuckGo · 3-attempt retry"),
        ("execute_code",    "tp-execute_code",    "Sandboxed subprocess · 10s"),
        ("retrieve_memory", "tp-retrieve_memory", "ChromaDB session history"),
    ]
    for name, cls, desc in tools:
        st.markdown(
            f'<div style="margin-bottom:6px">'
            f'<span class="tool-pill {cls}">{name}</span>'
            f'<div style="font-size:10px;color:#374151;margin-top:2px;font-family:JetBrains Mono,monospace">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Recent sessions ──
    st.markdown('<div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;font-family:JetBrains Mono,monospace">Memory</div>', unsafe_allow_html=True)
    try:
        sess_resp = requests.get(f"{API_BASE}/sessions", headers=HEADERS, timeout=5).json()
        sessions  = sess_resp.get("sessions", [])[:5]
        total_s   = sess_resp.get("total", 0)
        if sessions:
            for s in sessions:
                q  = s["metadata"].get("query", "")[:65]
                ts = s["metadata"].get("timestamp", "")[:10]
                tools_used = json.loads(s["metadata"].get("tools_used", "[]"))
                pills = "".join(
                    f'<span class="tool-pill tp-{t}">{t}</span>'
                    for t in tools_used
                )
                st.markdown(
                    f'<div class="sess-item">'
                    f'<div class="sess-q">{q}…</div>'
                    f'<div class="sess-meta">{ts}&nbsp;&nbsp;{pills}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if total_s > 5:
                st.markdown(f'<div style="font-size:10px;color:#374151;text-align:center;font-family:JetBrains Mono,monospace">+{total_s-5} more</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:11px;color:#374151">No sessions yet.</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="font-size:11px;color:#374151">Could not load.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="mh-mark">◈</div>
  <div>
    <div class="mh-title">Codebase Intelligence Agent</div>
    <div class="mh-sub">LangGraph plan-execute-replan · ChromaDB · Ollama llama3.1:8b · FastAPI · MCP</div>
    <div class="mh-badges">
      <span class="mh-badge">100% local</span>
      <span class="mh-badge">zero data exposure</span>
      <span class="mh-badge">function-level retrieval</span>
      <span class="mh-badge">persistent memory</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Example chips ──
examples = [
    "What does execution_node do and how does it handle failures?",
    "How do I fix ChromaDB n_results exceeding index size?",
    "Write a Python function to validate JSON schema and verify it works.",
    "Debug: why does search_docs return empty results?",
    "Explain the plan-execute-replan loop in this codebase.",
]

ex_cols = st.columns(len(examples))
for col, ex in zip(ex_cols, examples):
    with col:
        if st.button(ex[:34] + "…", use_container_width=True,
                     key=f"ex_{ex[:20]}", type="secondary"):
            st.session_state["prefill"] = ex

# ── Query input ──
prefill = st.session_state.pop("prefill", "")
query   = st.text_area(
    "Query",
    value=prefill,
    height=80,
    placeholder="Ask anything about the codebase, errors, or code generation…",
    label_visibility="collapsed",
)

# ── Screenshot upload (BUG-10: multimodal input) ──────────────────────────
uploaded_img = st.file_uploader(
    "📎 Attach screenshot  (optional - stack trace, error dialog, architecture diagram)",
    type=["png", "jpg", "jpeg", "webp"],
    key="screenshot_upload",
    help="Upload a screenshot of an error or diagram. The agent extracts its text.",
)
image_b64 = None
if uploaded_img is not None:
    import base64 as _b64
    image_b64 = _b64.b64encode(uploaded_img.read()).decode()
    st.caption(f"📎 Attached: {uploaded_img.name} ({len(image_b64)//1024} KB)")

run_col, clear_col = st.columns([5, 1])
with run_col:
    submit = st.button("◈  Run Agent", type="primary", use_container_width=True)
with clear_col:
    if st.button("Clear", type="secondary", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Agent execution
# ─────────────────────────────────────────────────────────────────────────────
TOOL_DOT = {
    "search_docs":     "td-blue",
    "web_search":      "td-orange",
    "execute_code":    "td-purple",
    "retrieve_memory": "td-green",
    "none":            "td-gray",
}
TOOL_CLS = {
    "search_docs":     "tp-search_docs",
    "web_search":      "tp-web_search",
    "execute_code":    "tp-execute_code",
    "retrieve_memory": "tp-retrieve_memory",
    "none":            "tp-none",
}

if submit and query.strip():
    t0 = time.time()

    current_plan: list[str] = []
    tool_events:  list[dict] = []
    replan_count  = 0
    tool_count    = 0

    # ── Layout ──────────────────────────────────────────────────────────────
    stat_row_ph  = st.empty()
    status_ph    = st.empty()
    plan_ph      = st.empty()
    trace_ph     = st.empty()
    answer_ph    = st.empty()

    def _stat_html(steps, tools, replans, lat):
        return f"""<div class="stat-row">
          <div class="stat-box"><span class="stat-n">{steps}</span><span class="stat-l">Steps</span></div>
          <div class="stat-box"><span class="stat-n">{tools}</span><span class="stat-l">Tools</span></div>
          <div class="stat-box"><span class="stat-n">{replans}</span><span class="stat-l">Replans</span></div>
          <div class="stat-box"><span class="stat-n">{lat}</span><span class="stat-l">Elapsed</span></div>
        </div>"""

    def _plan_html(plan):
        steps_html = "".join(
            f'<div class="plan-step"><span class="plan-num">{i+1:02d}</span>{s}</div>'
            for i, s in enumerate(plan)
        )
        return f'<div class="plan-banner"><div class="plan-title">◈ Execution plan · {len(plan)} steps</div>{steps_html}</div>'

    def _trace_html(events):
        rows = []
        for i, ev in enumerate(events):
            is_last   = i == len(events) - 1
            tool      = ev.get("tool", "none")
            task      = ev.get("task", "")[:120]
            result    = ev.get("result", "").strip()[:300]
            success   = ev.get("success", True)
            ts        = ev.get("ts", "")
            dot_cls   = TOOL_DOT.get(tool, "td-gray")
            pill_cls  = TOOL_CLS.get(tool, "tp-none")
            ok_icon   = '<span class="success-dot">✓</span>' if success else '<span class="fail-dot">✗</span>'
            connector = '' if is_last else '<div class="trace-connector"></div>'
            result_block = (
                f'<div class="trace-result">{result}</div>'
                if result else ""
            )
            rows.append(f"""
            <div class="trace-row">
              <div class="trace-line">
                <div class="trace-dot {dot_cls}"></div>
                {connector}
              </div>
              <div class="trace-body">
                <div class="trace-header">
                  <span class="tool-pill {pill_cls}">{tool}</span>
                  {ok_icon}
                  <span class="trace-ts">{ts}</span>
                </div>
                <div class="trace-task">{task}</div>
                {result_block}
              </div>
            </div>""")
        return f'<div class="trace-wrap">{"".join(rows)}</div>'

    stat_row_ph.markdown(
        _stat_html(0, 0, 0, "0.0s"), unsafe_allow_html=True)
    status_ph.markdown(
        '<div style="font-size:11px;color:#3B7FEF;font-family:JetBrains Mono,monospace;margin:6px 0">◈ Connecting to agent…</div>',
        unsafe_allow_html=True,
    )

    try:
        with requests.post(
            f"{API_BASE}/query/stream",
            headers=HEADERS,
            json={"query": query, "image_base64": image_b64},
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue

                ev    = json.loads(line[6:])
                etype = ev.get("type")
                edata = ev.get("data")
                elapsed = f"{round(time.time() - t0, 1)}s"

                if etype == "plan":
                    current_plan = edata
                    plan_ph.markdown(_plan_html(current_plan), unsafe_allow_html=True)
                    status_ph.markdown(
                        f'<div style="font-size:11px;color:#3B7FEF;font-family:JetBrains Mono,monospace;margin:6px 0">◈ Plan ready · executing {len(current_plan)} steps</div>',
                        unsafe_allow_html=True)

                elif etype == "tool_call":
                    tool_count += 1
                    tool_name   = edata.get("tool", "none")
                    task_text   = edata.get("task", "")
                    tool_events.append({
                        "tool": tool_name, "task": task_text,
                        "result": "", "success": True,
                        "ts": elapsed,
                    })
                    trace_ph.markdown(_trace_html(tool_events), unsafe_allow_html=True)
                    status_ph.markdown(
                        f'<div style="font-size:11px;color:#3B7FEF;font-family:JetBrains Mono,monospace;margin:6px 0">◈ Running {tool_name}…</div>',
                        unsafe_allow_html=True)

                elif etype == "tool_result":
                    if tool_events:
                        tool_events[-1]["result"]  = edata.get("result", "")
                        tool_events[-1]["success"] = edata.get("success", True)
                    trace_ph.markdown(_trace_html(tool_events), unsafe_allow_html=True)

                elif etype == "replan":
                    replan_count = edata.get("count", 1)
                    trace_ph.markdown(
                        _trace_html(tool_events) +
                        f'<div class="replan-bar">↩ Replan triggered · attempt {replan_count}/2</div>',
                        unsafe_allow_html=True,
                    )

                elif etype == "answer":
                    status_ph.markdown(
                        '<div style="font-size:11px;color:#3B7FEF;font-family:JetBrains Mono,monospace;margin:6px 0">◈ Synthesising answer…</div>',
                        unsafe_allow_html=True)
                    # BUG-09 fix: use st.markdown - handles code fences/bold/tables
                    with answer_ph.container():
                        st.markdown("**◈ Answer**")
                        st.markdown(edata)

                elif etype == "done":
                    final_lat = edata.get("latency_s", round(time.time() - t0, 1))
                    status_ph.markdown(
                        f'<div style="font-size:11px;color:#10B981;font-family:JetBrains Mono,monospace;margin:6px 0">✓ Completed in {final_lat}s</div>',
                        unsafe_allow_html=True)
                    stat_row_ph.markdown(
                        _stat_html(len(current_plan), tool_count, replan_count, f"{final_lat}s"),
                        unsafe_allow_html=True,
                    )

                elif etype == "error":
                    st.error(f"Agent error: {edata.get('message', 'Unknown')}")

                if etype not in ("answer", "done", "error"):
                    stat_row_ph.markdown(
                        _stat_html(len(current_plan), tool_count, replan_count, elapsed),
                        unsafe_allow_html=True,
                    )

    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API server.\n\n`uvicorn api.main:app --host 0.0.0.0 --port 8000`")
    except Exception as exc:
        st.error(f"Error: {exc}")

elif submit:
    st.warning("Enter a query first.")
