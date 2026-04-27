"""
Approval Gateway Pattern (Online Extra)
==========================================
Demonstrates LangGraph's interrupt_before for human-in-the-loop approval.
The graph pauses before a critical node and waits for human input.

Usage:
    python approval_gateway.py

Requires: pip install langgraph langchain-openai
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    proposal: str
    approved: bool


def generate_proposal(state: ApprovalState) -> dict:
    """Generate a proposal that needs human approval."""
    proposal = "Deploy v2.3.1 to production with new auth module and rate limiting."
    print(f"📝 Proposal: {proposal}")
    return {"proposal": proposal}


def human_approval_node(state: ApprovalState) -> dict:
    """This node is interrupted — execution pauses here for human input."""
    print(f"\n👤 Reviewing proposal: {state['proposal']}")
    print("[Auto-approving for demo]")
    return {"approved": True}


def execute_deployment(state: ApprovalState) -> dict:
    """Execute the approved proposal."""
    if state["approved"]:
        print("🚀 Deployment executed successfully!")
    else:
        print("⛔ Deployment cancelled by reviewer.")
    return {"messages": [{"role": "system", "content": f"Deployment {'executed' if state['approved'] else 'cancelled'}."}]}


# ── Build Graph ──────────────────────────────────────────────────────

graph = StateGraph(ApprovalState)
graph.add_node("generate_proposal", generate_proposal)
graph.add_node("human_approval", human_approval_node)
graph.add_node("execute_deployment", execute_deployment)

graph.add_edge(START, "generate_proposal")
graph.add_edge("generate_proposal", "human_approval")
graph.add_edge("human_approval", "execute_deployment")
graph.add_edge("execute_deployment", END)

checkpointer = MemorySaver()

# interrupt_before pauses the graph BEFORE this node runs,
# allowing a human to review and modify state before continuing.
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"],
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "deploy-001"}}
    initial = {"messages": [], "proposal": "", "approved": False}

    # First invocation: runs until interrupt_before pauses at human_approval
    print("Phase 1: Generating proposal...")
    result = app.invoke(initial, config=config)
    print(f"\n⏸️ Graph paused before human_approval node.")
    print(f"   State: proposal='{result.get('proposal', '')}'")

    # Resume: human approves and graph continues
    print("\nPhase 2: Resuming after approval...")
    result = app.invoke(None, config=config)
    print("\n✅ Pipeline complete.")