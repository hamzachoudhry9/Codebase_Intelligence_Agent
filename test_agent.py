from dotenv import load_dotenv
from agent.graph import agent_graph as graph
load_dotenv()

def run_agent_test(query: str):
    print(f"\n--- USER QUERY: {query} ---")

    initial_state = {
        "query": query,
        "past_context": "",
        "plan": [],
        "current_step_index": 0,
        "messages": [],
        "tool_outputs": [],
        "final_answer": "",
        "replan_count": 0,
        "done": False,
    }

    for event in graph.stream(initial_state):
        for node_name, output in event.items():
            print(f"\n[Node: {node_name}]")

            # Show the plan when planning node fires
            if node_name == "planning" and "plan" in output:
                for i, step in enumerate(output["plan"], 1):
                    print(f"  Step {i}: {step}")

            # Show tool result when execution node fires
            elif node_name == "execution" and "tool_outputs" in output:
                outputs = output.get("tool_outputs", [])
                if outputs:
                    last = outputs[-1]
                    print(f"  Tool: {last.get('tool')}")
                    print(f"  Result: {str(last.get('result', ''))[:200]}")

            # Show final answer when synthesis fires
            elif node_name == "synthesis" and "final_answer" in output:
                print(f"  Answer: {output['final_answer'][:400]}")

if __name__ == "__main__":
    run_agent_test("What is the folder structure of this project according to the docs?")