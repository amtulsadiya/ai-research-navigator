"""
LangGraph state machine — the M3 agent graph.

Graph topology:
  START → router_node → [conditional edge based on route] →
    concept_explanation_node  → END
    paper_deep_dive_node      → END
    compare_approaches_node   → END
    recent_developments_node  → END
    find_papers_node          → END
    fallback_node             → END

Why a graph instead of if/else?
- Visualisable: LangGraph renders the graph as a diagram
- Extensible: add new nodes/edges without touching existing code
- Observable: each node transition is logged and traceable
- Serialisable: state at each step can be saved for debugging
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from research_navigator.agents.nodes import (
    compare_approaches_node,
    concept_explanation_node,
    fallback_node,
    find_papers_node,
    paper_deep_dive_node,
    recent_developments_node,
    router_node,
)
from research_navigator.agents.state import AgentState


def route_query(state: AgentState) -> str:
    """
    Conditional edge function — reads route from state and
    returns the name of the next node to execute.

    This is LangGraph's routing mechanism:
    add_conditional_edges(source_node, routing_function, mapping)
    """
    route = state.get("route", "concept_explanation")
    return route


def build_graph() -> StateGraph:
    """
    Build and return the compiled LangGraph state machine.

    Building the graph:
    1. Create StateGraph with our AgentState schema
    2. Add all nodes (functions that transform state)
    3. Add edges (connections between nodes)
    4. Add conditional edges (router → one of 6 destinations)
    5. Compile and return
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("concept_explanation", concept_explanation_node)
    graph.add_node("paper_deep_dive", paper_deep_dive_node)
    graph.add_node("compare_approaches", compare_approaches_node)
    graph.add_node("recent_developments", recent_developments_node)
    graph.add_node("find_papers", find_papers_node)
    graph.add_node("fallback", fallback_node)

    # Entry point: START → router
    graph.add_edge(START, "router")

    # Conditional routing: router → one of 6 nodes based on state["route"]
    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "concept_explanation": "concept_explanation",
            "paper_deep_dive": "paper_deep_dive",
            "compare_approaches": "compare_approaches",
            "recent_developments": "recent_developments",
            "find_papers": "find_papers",
            "out_of_scope": "fallback",
        },
    )

    # All handler nodes go to END
    graph.add_edge("concept_explanation", END)
    graph.add_edge("paper_deep_dive", END)
    graph.add_edge("compare_approaches", END)
    graph.add_edge("recent_developments", END)
    graph.add_edge("find_papers", END)
    graph.add_edge("fallback", END)

    return graph


def get_compiled_graph() -> Any:
    """Return the compiled graph ready for invocation."""
    graph = build_graph()
    return graph.compile()


def run_query(user_query: str) -> AgentState:
    app = get_compiled_graph()
    initial_state: AgentState = {"query": user_query}
    final_state: AgentState = app.invoke(initial_state)  # type: ignore[no-any-expr]
    return final_state


def visualize_graph() -> None:
    """
    Print the graph structure using LangGraph's built-in renderer.
    Saved to docs/adr/graph_visualization.txt
    """
    from pathlib import Path

    graph = build_graph()
    compiled = graph.compile()

    # Get mermaid diagram
    try:
        diagram = compiled.get_graph().draw_mermaid()
        output_path = Path("docs/adr/graph_visualization.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"# M3 Agent Graph\n\n```mermaid\n{diagram}\n```\n")
        print(f"Graph saved to {output_path}")
        print(diagram)
    except Exception as e:
        print(f"Visualization failed: {e}")
        # Fallback: print node list
        print("Nodes:", list(graph.nodes.keys()))
