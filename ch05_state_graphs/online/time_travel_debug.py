"""
Time-Travel Debugging (Online Extra)
=======================================
Demonstrates LangGraph's checkpoint-based state history for debugging.
Replay, inspect, and patch state at any historical point.

Usage:
    python time_travel_debug.py

Requires: pip install langgraph
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class DebugState(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int
    data: str


def step_one(state: DebugState) -> dict:
    print(f"  Step 1: counter={state['counter']}")
    return {"counter": state["counter"] + 1, "data": "fetched"}


def step_two(state: DebugState) -> dict:
    print(f"  Step 2: counter={state['counter']}, data={state['data']}")
    return {"counter": state["counter"] + 1, "data": "processed"}


def step_three(state: DebugState) -> dict:
    """This step simulates a bug — it corrupts the data."""
    print(f"  Step 3: counter={state['counter']}, data={state['data']}")
    return {"counter": state["counter"] + 1, "data": "CORRUPTED_VALUE"}


# ── Build Graph ──────────────────────────────────────────────────────

graph = StateGraph(DebugState)
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)
graph.add_node("step_three", step_three)

graph.add_edge(START, "step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", "step_three")
graph.add_edge("step_three", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "debug-session-001"}}
    initial = {"messages": [], "counter": 0, "data": ""}

    # Run the graph (step_three will "corrupt" the data)
    print("🏃 Running graph (step 3 has a bug)...")
    result = app.invoke(initial, config=config)
    print(f"\n❌ Final state: counter={result['counter']}, data='{result['data']}'\n")

    # Time-travel: inspect all historical states
    print("⏪ State History (newest first):")
    history = list(app.get_state_history(config))
    for i, snapshot in enumerate(history):
        state = snapshot.values
        print(f"  [{i}] counter={state.get('counter', '?')}, data='{state.get('data', '?')}'")

    # Fix: rewind to the state BEFORE step_three corrupted the data
    print(f"\n🔧 Rewinding to snapshot [{len(history)-2}] (before corruption)...")
    target = history[-2]  # State after step_two, before step_three

    # Patch the state with corrected data
    app.update_state(
        config=target.config,
        values={"data": "processed_correctly"},
    )

    # Resume from the patched checkpoint
    print("▶️ Resuming from patched state...")
    fixed_result = app.invoke(None, config=target.config)
    print(f"\n✅ Fixed state: counter={fixed_result['counter']}, data='{fixed_result['data']}'")