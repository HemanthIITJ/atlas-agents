"""
Parallel Fan-Out Pattern (Online Extra)
=========================================
Demonstrates LangGraph's parallel execution: multiple nodes run
simultaneously and their outputs merge via a reducer (operator.add).

Usage:
    python parallel_fanout.py

Requires: pip install langgraph
"""

import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


class FanOutState(TypedDict):
    query: str
    # operator.add merges lists from parallel branches into one state variable
    search_results: Annotated[list[str], operator.add]


def search_reddit(state: FanOutState) -> dict:
    """Search Reddit for relevant discussions."""
    print("  🔍 Searching Reddit...")
    return {"search_results": [
        "[Reddit] r/MachineLearning: 'LangGraph vs CrewAI comparison thread'",
        "[Reddit] r/LocalLLaMA: 'Best framework for multi-agent agents'",
    ]}


def search_arxiv(state: FanOutState) -> dict:
    """Search arXiv for academic papers."""
    print("  📚 Searching arXiv...")
    return {"search_results": [
        "[arXiv] 'A Survey of LLM-based Multi-Agent Systems' (2025)",
        "[arXiv] 'ReAct: Synergizing Reasoning and Acting' (Yao et al.)",
    ]}


def search_github(state: FanOutState) -> dict:
    """Search GitHub for repositories."""
    print("  🐙 Searching GitHub...")
    return {"search_results": [
        "[GitHub] langchain-ai/langgraph — 15.2k stars",
        "[GitHub] crewAIInc/crewAI — 22.8k stars",
    ]}


def summarize(state: FanOutState) -> dict:
    """Summarize all gathered results."""
    print(f"\n📊 Summarizing {len(state['search_results'])} results...")
    return {"search_results": state["search_results"]}


# ── Build Graph ──────────────────────────────────────────────────────

graph = StateGraph(FanOutState)

graph.add_node("search_reddit", search_reddit)
graph.add_node("search_arxiv", search_arxiv)
graph.add_node("search_github", search_github)
graph.add_node("summarize", summarize)

# Fan-out: three edges from START execute their nodes in parallel
graph.add_edge(START, "search_reddit")
graph.add_edge(START, "search_arxiv")
graph.add_edge(START, "search_github")

# Fan-in: all three merge into summarize
graph.add_edge("search_reddit", "summarize")
graph.add_edge("search_arxiv", "summarize")
graph.add_edge("search_github", "summarize")

graph.add_edge("summarize", END)

app = graph.compile()


if __name__ == "__main__":
    print("⚡ Parallel Fan-Out Demo\n")
    print("Searching 3 sources simultaneously...")

    result = app.invoke({"query": "AI agent frameworks", "search_results": []})

    print("\n📋 All results (merged via operator.add):")
    for r in result["search_results"]:
        print(f"  • {r}")
    print(f"\nTotal: {len(result['search_results'])} results from 3 parallel sources")