"""
test_graph.py - Smoke test for the full LangGraph agent.

FIX (Bug 6): initial_state now includes ALL keys required by AgentState:
             query, past_context, plan, current_step_index, messages,
             tool_outputs, final_answer, replan_count, done.

             The original test only had 3 of these 9 keys. memory_retrieval_node
             immediately crashed with KeyError because it accessed state["query"]
             which technically existed, but should_replan accessed state["plan"]
             which did not - causing an unhandled KeyError mid-graph.

Run from the project root (with venv activated):
    python test_graph.py
"""

from dotenv import load_dotenv

load_dotenv()

from agent.graph import agent_graph


def run_agent_test(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    # FIX (Bug 6): All 9 AgentState keys must be present.
    # Missing any key causes a KeyError inside a node function.
    initial_state = {
        "query": query,
        "past_context": "",           # populated by memory_retrieval_node
        "plan": [],                   # populated by planning_node
        "current_step_index": 0,      # incremented by execution_node
        "messages": [],               # accumulated across nodes
        "tool_outputs": [],           # log of every tool call + result
        "final_answer": "",           # populated by synthesis_node
        "replan_count": 0,            # guard: max 2 replans
        "done": False,                # sentinel: True after synthesis
    }

    # Use .invoke() for a blocking call that returns the final state.
    # Use .stream() only if you want to watch each node fire in real time.
    result = agent_graph.invoke(initial_state)

    print("\n--- PLAN ---")
    for i, step in enumerate(result.get("plan", []), 1):
        print(f"  {i}. {step}")

    print("\n--- TOOLS USED ---")
    tools_used = {o["tool"] for o in result.get("tool_outputs", []) if o["tool"] != "none"}
    print(f"  {tools_used}")

    print("\n--- FINAL ANSWER (first 600 chars) ---")
    print(result.get("final_answer", "")[:600])
    print()


if __name__ == "__main__":
    # Test 1: should hit search_docs
    run_agent_test("What is the folder structure of this project according to the docs?")

    # Test 2: should hit execute_code
    run_agent_test("Write a Python function to add two numbers and verify it works.")
