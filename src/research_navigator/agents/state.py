"""
LangGraph state — explicit and serialisable.

Why explicit state?
LangGraph passes state through every node. Each node reads what it needs
and writes back what it produces. Making state explicit means:
- Every field is typed and documented
- Nodes can't accidentally share data via side effects
- State can be serialised to JSON for debugging and persistence

Why TypedDict instead of dataclass?
LangGraph requires TypedDict for state — it uses dict semantics internally
for merging partial state updates from nodes.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    The state object that flows through every node in the graph.

    total=False means all fields are optional — nodes only write
    the fields they produce, leaving others unchanged.

    Flow:
    Router reads: query
    Router writes: route, router_reasoning

    Each handler reads: query, route
    Each handler writes: answer, citations, was_refused, error

    Final output reads: answer, citations, was_refused
    """

    # Input
    query: str  # The user's original question

    # Router output
    route: str  # One of the 6 route names
    router_reasoning: str  # Why the router chose this route

    # Retrieval context
    top_score: float  # Cosine similarity of best chunk
    retrieved_chunk_count: int  # How many chunks were retrieved

    # Generation output
    answer: str  # The generated answer text
    citations: list[dict[str, object]]  # Serialised citation objects
    was_refused: bool  # True if answer was refused

    # Metadata
    error: str | None  # Error message if something failed
    doc_id_hint: str | None  # For PaperDeepDive: specific doc_id
    comparison_terms: list[str] | None  # For CompareApproaches: two terms


# Valid route names — used by router and conditional edges
ROUTES = {
    "concept_explanation",
    "paper_deep_dive",
    "compare_approaches",
    "recent_developments",
    "find_papers",
    "out_of_scope",
}
