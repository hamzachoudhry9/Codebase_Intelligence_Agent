"""bench/selfcheck.py - Validate the task suite and harness, no Ollama needed.

For every task this proves the benchmark is well-formed and the runner is
correct, independent of any model:

    bugfix / mm_debug : the seeded bug makes fail->pass tests fail BEFORE,
                        pass->pass tests pass BEFORE, and the gold patch
                        RESOLVES the task.
    testgen           : the reference suite passes on the original source and
                        kills every seeded mutant.

Run:  python -m bench.selfcheck   (or: make selfcheck)
Exits non-zero if anything is broken - safe to use in CI.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .schema import discover_tasks
from .test_runner import resolved, run_tests
from .workspace import apply_gold, workspace


def check_bugfix(task) -> bool:
    with workspace(task) as ws:
        if task.image:
            shutil.copy2(task.image_path, Path(ws) / "issue_image.png")
        before = run_tests(ws, task.test_files)
        f2p_fail = all(not before.get(x, True) for x in task.fail_to_pass)
        p2p_ok = all(before.get(x, False) for x in task.pass_to_pass)
    with workspace(task) as ws:
        apply_gold(task, ws)
        after = run_tests(ws, task.test_files)
        gold_ok = resolved(task.fail_to_pass, task.pass_to_pass, after)
    ok = f2p_fail and p2p_ok and gold_ok
    print(f"[{task.id:18}] {task.category:8} bug_fails_before={f2p_fail!s:5} "
          f"others_pass={p2p_ok!s:5} gold_resolves={gold_ok!s:5} => {'OK' if ok else 'BROKEN'}")
    return ok


def check_testgen(task) -> bool:
    gen = task.test_files[0]
    target = task.mutated_file
    with workspace(task) as ws:
        shutil.copy2(task.gold_dir / "reference_tests.py", ws / gen)
        base = run_tests(ws, [gen])
        base_ok = bool(base) and all(base.values())
        killed = 0
        for m in task.mutants:
            shutil.copy2(task.root / m, ws / target)
            r = run_tests(ws, [gen])
            if not (r and all(r.values())):
                killed += 1
            shutil.copy2(task.src_dir / Path(target).name, ws / target)
    ok = base_ok and killed == len(task.mutants)
    print(f"[{task.id:18}] testgen  ref_passes={base_ok!s:5} "
          f"mutants_killed={killed}/{len(task.mutants)} => {'OK' if ok else 'BROKEN'}")
    return ok


def main() -> int:
    tasks = discover_tasks()
    print(f"Validating {len(tasks)} tasks\n")
    ok = True
    for t in tasks:
        ok &= check_testgen(t) if t.category == "testgen" else check_bugfix(t)
    print(f"\nALL TASKS VALID: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
