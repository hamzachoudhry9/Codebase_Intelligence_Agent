"""bench/smoke.py - Exercise the actor loop deterministically, no Ollama needed.

Uses the FakeLLM to feed the actor canned (correct / incorrect) patches so the
perceive -> localize -> edit -> test -> retry loop can be verified anywhere,
including CI. Demonstrates three behaviours:

    1. a correct patch resolves a bugfix task in one iteration
    2. a wrong first patch triggers a retry that then succeeds
    3. a multimodal task routes its issue image through the (faked) vision fn

Run:  python -m bench.smoke   (or: make smoke)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from agent.actor import run_actor
from agent.llm import FakeLLM

from .schema import load_task
from .workspace import workspace

TASKS = Path(__file__).parent / "tasks"


def _td(t) -> dict:
    return {"id": t.id, "category": t.category, "prompt": t.prompt,
            "fail_to_pass": t.fail_to_pass, "pass_to_pass": t.pass_to_pass,
            "test_files": t.test_files, "image": t.image,
            "expected_tools": t.expected_tools}


def main() -> int:
    ok = True

    # 1) correct patch -> resolved in one iteration
    t = load_task(TASKS / "bugfix_slug")
    gold = (t.gold_dir / "textutil.py").read_text()
    fake = FakeLLM(['["src/textutil.py"]', json.dumps({"src/textutil.py": gold})])
    with workspace(t) as ws:
        f = run_actor(_td(t), str(ws), llm=fake, vision=lambda p: "", max_iters=3)
    ok &= f["resolved"] and f["iters"] == 1
    print(f"bugfix_slug    resolved={f['resolved']} iters={f['iters']} tools={f['tools_used']}")

    # 2) wrong patch then correct -> retry, resolved on iter 2
    t2 = load_task(TASKS / "bugfix_backoff")
    bad = (t2.src_dir / "retry.py").read_text()
    good = (t2.gold_dir / "retry.py").read_text()
    fake2 = FakeLLM(['["src/retry.py"]',
                     json.dumps({"src/retry.py": bad}),
                     json.dumps({"src/retry.py": good})])
    with workspace(t2) as ws:
        f2 = run_actor(_td(t2), str(ws), llm=fake2, vision=lambda p: "", max_iters=3)
    ok &= f2["resolved"] and f2["iters"] == 2
    print(f"bugfix_backoff resolved={f2['resolved']} iters={f2['iters']} (retry-then-fix)")

    # 3) multimodal -> vision fn supplies the traceback text
    t3 = load_task(TASKS / "mm_debug_lastn")
    goldseq = (t3.gold_dir / "seq.py").read_text()
    fake3 = FakeLLM(['["src/seq.py"]', json.dumps({"src/seq.py": goldseq})])
    with workspace(t3) as ws:
        shutil.copy2(t3.image_path, Path(ws) / "issue_image.png")
        f3 = run_actor(_td(t3), str(ws), llm=fake3,
                       vision=lambda p: "IndexError at src/seq.py:5 in last_n", max_iters=3)
    ok &= f3["resolved"] and "read_image" in f3["tools_used"]
    print(f"mm_debug_lastn resolved={f3['resolved']} read_image_used={'read_image' in f3['tools_used']}")

    print(f"\nSMOKE OK: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
