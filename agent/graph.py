r"""
graph.py - Constructs and compiles the LangGraph StateGraph.

UPGRADE: an entry router now sends simple lookups down a single-retrieval
fast path and reserves the full plan-execute-replan loop for multi-step work.
This raises tool precision and cuts latency on the common case.

Graph flow:
                          START
                            |
                    [route_entry]
                    /            \
              "fast"              "full"
                |                   |
           fast_path        memory_retrieval
                |                   |
                |               planning
                |                   |
                |               execution <───────────────┐
                |                   |                      |
                |             [should_replan]              |
                |              /     |      \              |
                |        execute  replan  synthesize       |
                |           |        └──────────────────── ┘
                |           |
                |           └──> synthesis
                |                   |
                └──────────────> save_memory
                                    |
                                   END
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    execution_node,
    fast_path_node,
    memory_retrieval_node,
    planning_node,
    replan_node,
    route_entry,
    save_memory_node,
    should_replan,
    synthesis_node,
)
from .state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("fast_path", fast_path_node)
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("planning", planning_node)
    graph.add_node("execution", execution_node)
    graph.add_node("replan", replan_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("save_memory", save_memory_node)

    # Entry router: fast path vs full loop
    graph.add_conditional_edges(
        START,
        route_entry,
        {"fast": "fast_path", "full": "memory_retrieval"},
    )

    # Fast path -> save -> end
    graph.add_edge("fast_path", "save_memory")

    # Full loop
    graph.add_edge("memory_retrieval", "planning")
    graph.add_edge("planning", "execution")
    graph.add_conditional_edges(
        "execution",
        should_replan,
        {"execute": "execution", "replan": "replan", "synthesize": "synthesis"},
    )
    graph.add_edge("replan", "execution")
    graph.add_edge("synthesis", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# Module-level singleton - import this in api/main.py and mcp_server/server.py
agent_graph = build_graph()
