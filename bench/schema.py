"""bench/schema.py - Task schema + loader for the execution-based benchmark.

A task is a self-contained mini-codebase with a seeded defect (or a
test-generation / refactoring objective) plus the tests that verify a
correct solution. Tasks live on disk under bench/tasks/<task_id>/:

    bench/tasks/<task_id>/
        task.json          # this schema, serialized
        src/               # the (buggy) source the agent edits
        tests/             # pytest tests; the verifier runs a subset
        gold/              # OPTIONAL full-file reference solution(s).
                           # Used only by `make selfcheck` to prove the
                           # task is solvable and the harness is correct.
                           # NEVER shown to the agent.
        issue.png          # OPTIONAL rendered error image (multimodal tasks)

Categories map to the SDLC slices named in the JD:
    bugfix       - localize + patch a failing behaviour
    testgen      - write tests that kill seeded mutants (mutation-based)
    refactor     - change structure while the suite stays green
    mm_debug     - same as bugfix, but the issue is delivered as an image
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = {"bugfix", "testgen", "refactor", "mm_debug"}


@dataclass
class Task:
    id: str
    category: str
    prompt: str                      # natural-language issue shown to the agent
    fail_to_pass: list[str]          # test names that must flip failing -> passing
    pass_to_pass: list[str]          # test names that must remain passing
    test_files: list[str]            # pytest files the verifier runs
    expected_tools: list[str] = field(default_factory=list)
    image: str | None = None         # filename of issue image (mm_debug only)
    mutants: list[str] = field(default_factory=list)   # testgen only: gold/<m>.py
    mutated_file: str | None = None                    # testgen only: src file mutants replace
    root: Path = field(default=Path("."))              # absolute task dir

    @property
    def src_dir(self) -> Path:
        return self.root / "src"

    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def gold_dir(self) -> Path:
        return self.root / "gold"

    @property
    def image_path(self) -> Path | None:
        return (self.root / self.image) if self.image else None

    def validate(self) -> None:
        assert self.category in CATEGORIES, f"{self.id}: bad category {self.category!r}"
        assert self.src_dir.is_dir(), f"{self.id}: missing src/"
        assert self.tests_dir.is_dir(), f"{self.id}: missing tests/"
        if self.category == "mm_debug":
            assert self.image_path and self.image_path.exists(), \
                f"{self.id}: mm_debug task needs an issue image"
        if self.category == "testgen":
            assert self.mutants, f"{self.id}: testgen task needs >=1 mutant"


def load_task(task_dir: str | Path) -> Task:
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "task.json").read_text())
    meta["root"] = task_dir
    # Drop unknown keys so the schema stays forward-compatible.
    known = Task.__dataclass_fields__.keys()
    meta = {k: v for k, v in meta.items() if k in known}
    task = Task(**meta)
    task.validate()
    return task


def discover_tasks(tasks_root: str | Path = None) -> list[Task]:
    tasks_root = Path(tasks_root or Path(__file__).parent / "tasks")
    out = []
    for d in sorted(tasks_root.iterdir()):
        if d.is_dir() and (d / "task.json").exists():
            out.append(load_task(d))
    return out
