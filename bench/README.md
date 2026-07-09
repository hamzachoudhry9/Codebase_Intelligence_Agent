# SDLC Agent Benchmark (`bench/`)

An **execution-based** benchmark for agentic coding systems, applied to a
controlled, local, multi-category task suite. Where `eval/` scores Q&A answer
quality with an LLM-as-judge, this measures whether the agent can **actually
change code and make the tests pass** - the SWE-bench methodology, runnable on
a laptop with no API keys.

## Why this exists

The agent shipped with a Q&A evaluator. That answers "is the explanation
good?" but not "can it *fix* the bug?". Recruiters for agentic-SDLC roles care
about the second question, and it's the one almost no portfolio project
measures. This benchmark closes that gap and doubles as the thing the agent is
evaluated *against*.

## How a task is scored

Each task is a self-contained mini-repo with tests. A solution is **resolved**
only if:

- every **fail→pass** test (failing on the buggy code) now passes, **and**
- every **pass→pass** test (a guard against regressions) still passes.

This is binary and execution-grounded - no judge, no keywords, no partial
credit for a plausible-sounding diff that doesn't run.

## Categories (mapped to SDLC slices in the JD)

| Category   | What the agent must do                                  | Scoring |
|------------|----------------------------------------------------------|---------|
| `bugfix`   | localize and patch a defect                              | fail→pass / pass→pass |
| `testgen`  | write tests that pin a contract                          | tests pass on original **and** kill every seeded mutant (mutation score = 1.0) |
| `mm_debug` | same as bugfix, but the bug report is an **image** of a traceback | fail→pass / pass→pass |
| `refactor` | change structure while the suite stays green             | pass→pass |

The `testgen` task uses **mutation testing**: the suite the agent writes is run
against deliberately broken copies of the source (e.g. splitting on every `=`
instead of the first, or dropping a `.strip()`). A suite only scores if it
*catches* those mutants - which is how you measure test quality objectively.

## Layout

```
bench/
  schema.py        Task definition + loader
  workspace.py     Disposable per-run copy of a task (originals never mutated)
  test_runner.py   Runs selected pytest tests, reports per-test pass/fail
  metrics.py       Resolve rate, tool precision/recall, latency, tokens
  run_bench.py     Run all tasks across one or more models -> runs/<model>.json
  leaderboard.py   Combine runs -> LEADERBOARD.md + leaderboard.json
  selfcheck.py     Validate the suite + harness (no model needed)
  smoke.py         Exercise the actor loop with a deterministic FakeLLM
  tasks/<id>/      task.json, src/, tests/, gold/ (+ issue.png for mm_debug)
  runs/            per-model results (oracle.json is the harness self-test)
```

## Run it

```bash
pip install -r requirements-bench.txt

# 1) Prove the suite + harness are correct (no Ollama needed):
make selfcheck        # every bug fails before, every gold patch resolves after
make smoke            # the actor loop: solve, retry-then-fix, multimodal

# 2) Harness self-test with gold patches (expect 100%):
make bench-oracle

# 3) Real run against local models, then rank them:
make bench MODELS="llama3.1:8b,qwen2.5-coder:7b"
make leaderboard
```

## Extending the suite

Add a folder under `tasks/` with a `task.json` (see `schema.py` for fields),
a buggy `src/`, the verifying `tests/`, and a `gold/` reference solution.
`make selfcheck` will tell you immediately whether the task is well-formed
(the bug must fail before and the gold must resolve after). The harness scales
to dozens of tasks without code changes.
