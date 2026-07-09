"""
state.py - The AgentState TypedDict that flows through every LangGraph node.

Every node receives the current state dict and returns a partial dict with
only the keys it modifies. LangGraph merges updates automatically.

The `messages` field uses operator.add as its reducer so each node's
returned messages are appended rather than overwriting the list.
"""

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    query: str                   # Original user query - never modified after init
    past_context: str            # Serialized summaries from session memory retrieval
    plan: list[str]              # Ordered sub-task strings produced by planning_node
    current_step_index: int      # Index of the next plan step to execute
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Full message history
    tool_outputs: list[dict]     # Log: [{step, task, tool, args, result, success}]
    final_answer: str            # Synthesized response assembled by synthesis_node
    replan_count: int            # Guard: max 2 replans per query to prevent loops
    done: bool                   # Sentinel: True once synthesis_node completes
