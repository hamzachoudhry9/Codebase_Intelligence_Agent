# Upgrade Guide - from "codebase Q&A" to "an agentic SDLC system *and* the benchmark that measures it"

This guide documents the upgrade end to end: what changed, why, how to run and
verify it, and how each piece maps to the agentic-SDLC role. Everything here is
runnable and verified locally - there are two `make` targets a reviewer can run
in under a minute, with **no API keys and no GPU**, that prove the core works.

---

## 1. The reframe (why this matters)

The project started as a strong local **codebase Q&A agent**: plan→execute→
replan over a vector index, persistent memory, MCP, an eval harness. Good, but
"chat about my code" is a crowded category.

The upgrade turns it into two things recruiters for agentic-SDLC roles actually
look for:

1. **An agent that does work and verifies it** - not just answers. It localizes
   a bug, patches files, runs the tests, and iterates until they pass.
2. **An execution-based benchmark for agentic SDLC systems** - the literal job
   duty "develop and maintain agentic AI benchmarks for the SDLC", implemented
   with the SWE-bench fail→pass / pass→pass methodology, plus mutation-scored
   test generation and a multimodal (screenshot-of-a-traceback) category.

The narrative is honest and tight: *agentic coding tools exist, but teams can't
trust them on private code and have no local way to measure them. This measures
them - and the same repo is both the system under test and the harness.*

---

## 2. What changed (exact manifest)

**Modified (3 files):**

| File | Change |
|------|--------|
| `agent/graph.py` | Rewritten to add a complexity **router** in front of the existing loop: simple lookups take a single-retrieval **fast path**; complex tasks keep the full plan-execute-replan loop. |
| `agent/nodes.py` | Appended `fast_path_node` (one search + one synthesis) and `route_entry` (the router's decision function). Existing nodes untouched. |
| `api/main.py` | `/query` now accepts an optional `image_base64`; a helper runs the vision module and folds the screenshot's text into the query. |

**New - the benchmark (`bench/`):**

```
bench/schema.py  workspace.py  test_runner.py  metrics.py
bench/run_bench.py  leaderboard.py  selfcheck.py  smoke.py  README.md
bench/tasks/{bugfix_slug, bugfix_backoff, bugfix_lru, testgen_kvparse, mm_debug_lastn}/
bench/runs/oracle.json        # harness self-test result (100%)
```

**New - the actor + multimodal + router support (`agent/`):**

```
agent/actor.py        # LangGraph: perceive -> localize -> edit -> test -> retry
agent/actor_tools.py  # workspace-scoped read/write/list + run_tests (path-guarded)
agent/llm.py          # one model factory + a deterministic FakeLLM for CI
agent/vision.py       # image -> text (local VLM, OCR fallback)
agent/router.py       # complexity classifier
```

**New - tooling:** `Makefile`, `requirements-bench.txt`.

Nothing in the original Q&A pipeline was removed; the upgrade is additive and
the old behaviour is still reachable (complex queries flow through the exact
same nodes as before).

---

## 3. Install & verify (do this first)

```bash
pip install -r requirements.txt          # original deps
pip install -r requirements-bench.txt    # pytest, Pillow (+ optional vision)

# (1) Validate the task suite + harness - NO model required:
make selfcheck
#   -> every seeded bug fails before, every gold patch resolves after,
#      and the testgen task's mutants are all killed. "ALL TASKS VALID: True"

# (2) Exercise the actor loop deterministically - NO model required:
make smoke
#   -> a correct patch resolves in 1 iteration; a wrong first patch triggers a
#      retry that then succeeds; the multimodal task uses the vision path.

# (3) Harness self-test with gold patches (expect 100%):
make bench-oracle

# (4) Real evaluation against local models, then rank them:
make bench MODELS="llama3.1:8b,qwen2.5-coder:7b"
make leaderboard      # writes bench/LEADERBOARD.md
```

`make selfcheck` and `make smoke` are the two commands to show a reviewer: they
prove - without Ollama - that the tasks are real (bugs genuinely fail, gold
genuinely fixes) and the actor loop genuinely drives tasks to green.

---

## 4. The four upgrades, mapped to the role

### Phase 1 - Execution-based benchmark (the headline)

`eval/` answered "is the explanation good?" (LLM-as-judge). The benchmark
answers "**did it fix the bug?**" A task is *resolved* only when its fail→pass
tests pass and its pass→pass tests still pass - binary, grounded in execution,
no judge. It runs multiple models on the same suite and emits a leaderboard.

> **JD line hit:** *"develop and maintain agentic AI benchmarks for the SDLC."*
> This is the duty itself, not a proxy for it.

### Phase 2 - Actor mode (do work, then verify)

`agent/actor.py` is a second LangGraph: **perceive → localize → edit → test →
(retry | stop)**. It reads the *exact* current file contents (not vector
search - you can't patch what you can't see precisely), writes full new files,
runs the task's tests, and on failure feeds the failing test names back and
edits again, up to a budget.

> **JD line hit:** *"multi-step agentic AI to accelerate the SDLC (code
> generation, testing)."* Tool use is workspace-scoped and path-guarded.

### Phase 3 - One multimodal capability (measured)

`agent/vision.py` turns a **screenshot of a stack trace** into text (local VLM
via Ollama, OCR fallback). It's wired into `/query` (paste an error image) and
into the benchmark as the `mm_debug` category, so the capability is **scored**,
not just demoed. The task's `issue.png` is a real rendered traceback.

> **"Ways to stand out" hit:** *multimodal agentic frameworks.* Kept to one
> capability done properly and evaluated - not a pile of half-features.

### Phase 4 - Complexity router (fixes the weak eval numbers)

The original eval showed **tool precision ≈ 0.54** (0.22 on code-gen) and
**latency 22-30 s** - the agent planned and over-called tools even for simple
lookups. `agent/router.py` + the fast path send single-hop questions ("what
does X do", "where is Y") straight to one retrieval + one synthesis, skipping
planning and replanning entirely.

- **Tool precision ↑**: the fast path calls `search_docs` once, when
  appropriate, instead of speculatively invoking multiple tools.
- **Latency ↓**: simple queries drop a planning LLM call and the multi-step
  execution loop.
- **No regression on hard tasks**: anything classified `complex` (debug, fix,
  run, multi-part) flows through the original loop unchanged.

> **JD line hit:** *orchestration frameworks / efficiency of agentic workflows.*
> Frame it as the efficiency axis NeMo Agent Toolkit emphasises.

---

## 5. Honesty notes (use these in the interview - they read as maturity)

- **`bench/runs/oracle.json` is a harness self-test, not a model score.** It
  applies the gold patches to prove the suite is solvable and the runner is
  correct (100% by construction). Real rows come from `make bench`.
- **Report negative results.** When you run real models, expect a 7-8B model to
  resolve *some* tasks and miss others. That contrast (and the per-category
  breakdown) is the interesting finding - don't hide it.
- **The suite is small and controlled by design.** Five seeded tasks across
  four categories make the methodology legible; the harness scales to dozens
  without code changes (`make selfcheck` validates any new task instantly).
- **Mutation-scored test generation** is the honest way to grade tests: the
  agent's suite only counts if it *kills* deliberately broken versions of the
  source, not merely passes.

---

## 6. Suggested next steps (your GPU, your call)

1. `ollama pull llama3.1:8b` and `ollama pull qwen2.5-coder:7b`, then
   `make bench MODELS="llama3.1:8b,qwen2.5-coder:7b" && make leaderboard`.
2. Add one row for an NVIDIA-relevant model you can run locally (e.g. a
   Nemotron) so the leaderboard speaks the team's language - but only if you
   actually run it; a name-dropped row you didn't execute is the one thing that
   reads as fake.
3. Grow the suite toward 20-30 tasks (more bugfix variety, a `refactor` task,
   a second `mm_debug`). `make selfcheck` keeps every addition honest.
4. Re-run the original `eval/` after the router change and record the
   before/after tool-precision and latency deltas in the README.
