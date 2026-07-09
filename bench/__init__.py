"""SWE-style execution-based benchmark for the Codebase Intelligence Agent.

Unlike the LLM-as-judge harness in eval/, this measures *execution-based*
correctness: a task ships with tests, and a solution only counts if
previously-failing tests now pass (fail->pass) and previously-passing
tests still pass (pass->pass). This is the SWE-bench methodology applied
to a controlled, local, multi-category SDLC task suite.
"""
