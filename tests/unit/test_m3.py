"""
Unit tests for M3 — router classification and state validation.

We test:
1. State schema validation
2. Route names are valid
3. Router output format
4. Fallback message content
5. Graph structure (nodes and edges exist)

We do NOT test live Gemini calls here — those are in acceptance tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from research_navigator.agents.graph import build_graph, route_query
from research_navigator.agents.nodes import (
    _format_result,
    fallback_node,
)
from research_navigator.agents.state import ROUTES, AgentState

# ── State tests ────────────────────────────────────────────────────────────────


def test_agent_state_is_typed_dict() -> None:
    """AgentState should be a TypedDict."""
    state: AgentState = {"query": "What is attention?"}
    assert state["query"] == "What is attention?"


def test_routes_set_contains_all_routes() -> None:
    """All 6 routes must be defined."""
    assert "concept_explanation" in ROUTES
    assert "paper_deep_dive" in ROUTES
    assert "compare_approaches" in ROUTES
    assert "recent_developments" in ROUTES
    assert "find_papers" in ROUTES
    assert "out_of_scope" in ROUTES
    assert len(ROUTES) == 6


# ── Route query tests ──────────────────────────────────────────────────────────


def test_route_query_returns_correct_route() -> None:
    """route_query should return the route from state."""
    state: AgentState = {
        "query": "test",
        "route": "concept_explanation",
    }
    assert route_query(state) == "concept_explanation"


def test_route_query_defaults_to_concept_explanation() -> None:
    """route_query should default if route not set."""
    state: AgentState = {"query": "test"}
    result = route_query(state)
    assert result == "concept_explanation"


def test_route_query_all_routes() -> None:
    """Every valid route should be returnable."""
    for route in ROUTES:
        state: AgentState = {"query": "test", "route": route}
        assert route_query(state) == route


# ── Fallback node tests ────────────────────────────────────────────────────────


def test_fallback_node_returns_refused() -> None:
    """Fallback node must set was_refused=True."""
    state: AgentState = {"query": "What is pizza?"}
    result = fallback_node(state)
    assert result["was_refused"] is True


def test_fallback_node_has_helpful_message() -> None:
    """Fallback message should suggest valid query types."""
    state: AgentState = {"query": "What is pizza?"}
    result = fallback_node(state)
    assert "concept" in result["answer"].lower() or "AI/ML" in result["answer"]


def test_fallback_node_empty_citations() -> None:
    """Fallback node should return empty citations."""
    state: AgentState = {"query": "cooking recipe"}
    result = fallback_node(state)
    assert result["citations"] == []


# ── Format result tests ────────────────────────────────────────────────────────


def test_format_result_refused() -> None:
    """_format_result should handle refused answers."""
    mock_result = MagicMock()
    mock_result.answer = "I don't have enough material."
    mock_result.citations = []
    mock_result.was_refused = True
    mock_result.top_score = 0.3

    result = _format_result(mock_result)
    assert result["was_refused"] is True
    assert result["answer"] == "I don't have enough material."
    assert result["citations"] == []


def test_format_result_with_citations() -> None:
    """_format_result should serialise citations correctly."""
    mock_citation = MagicMock()
    mock_citation.number = 1
    mock_citation.title = "Attention Is All You Need"
    mock_citation.authors_formatted = "Vaswani et al."
    mock_citation.year = 2017
    mock_citation.source_label = "arXiv:1706.03762"
    mock_citation.section_title = "Abstract"
    mock_citation.source_url = "https://arxiv.org/abs/1706.03762"

    mock_result = MagicMock()
    mock_result.answer = "The transformer uses attention [1]."
    mock_result.citations = [mock_citation]
    mock_result.was_refused = False
    mock_result.top_score = 0.85

    result = _format_result(mock_result)
    assert len(result["citations"]) == 1
    assert result["citations"][0]["title"] == "Attention Is All You Need"
    assert result["citations"][0]["number"] == 1


# ── Graph structure tests ──────────────────────────────────────────────────────


def test_graph_has_all_nodes() -> None:
    """Graph must contain all required nodes."""
    graph = build_graph()
    node_names = set(graph.nodes.keys())

    required_nodes = {
        "router",
        "concept_explanation",
        "paper_deep_dive",
        "compare_approaches",
        "recent_developments",
        "find_papers",
        "fallback",
    }
    for node in required_nodes:
        assert node in node_names, f"Missing node: {node}"


def test_graph_compiles() -> None:
    """Graph should compile without errors."""
    from research_navigator.agents.graph import get_compiled_graph

    app = get_compiled_graph()
    assert app is not None


# ── Router node tests (mocked) ─────────────────────────────────────────────────


def test_router_node_concept_explanation() -> None:
    """Router should classify concept queries correctly."""
    with patch("research_navigator.agents.nodes.ChatGoogleGenerativeAI") as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.invoke.return_value.content = (
            '{"route": "concept_explanation", "reasoning": "user wants explanation", '
            '"doc_id_hint": null, "comparison_terms": null}'
        )
        from research_navigator.agents.nodes import router_node

        state: AgentState = {"query": "What is attention?"}
        result = router_node(state)

    assert result["route"] == "concept_explanation"


def test_router_node_out_of_scope() -> None:
    """Router should classify non-AI queries as out_of_scope."""
    with patch("research_navigator.agents.nodes.ChatGoogleGenerativeAI") as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.invoke.return_value.content = (
            '{"route": "out_of_scope", "reasoning": "not about AI/ML", '
            '"doc_id_hint": null, "comparison_terms": null}'
        )
        from research_navigator.agents.nodes import router_node

        state: AgentState = {"query": "What is the best pizza recipe?"}
        result = router_node(state)

    assert result["route"] == "out_of_scope"


def test_router_node_paper_deep_dive() -> None:
    """Router should classify paper-specific queries correctly."""
    with patch("research_navigator.agents.nodes.ChatGoogleGenerativeAI") as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.invoke.return_value.content = (
            '{"route": "paper_deep_dive", "reasoning": "specific paper mentioned", '
            '"doc_id_hint": "arxiv-1706.03762", "comparison_terms": null}'
        )
        from research_navigator.agents.nodes import router_node

        state: AgentState = {"query": "Tell me about Attention Is All You Need"}
        result = router_node(state)

    assert result["route"] == "paper_deep_dive"
    assert result["doc_id_hint"] == "arxiv-1706.03762"


def test_router_node_compare() -> None:
    """Router should extract comparison terms."""
    with patch("research_navigator.agents.nodes.ChatGoogleGenerativeAI") as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.invoke.return_value.content = (
            '{"route": "compare_approaches", "reasoning": "comparing two methods", '
            '"doc_id_hint": null, "comparison_terms": ["RLHF", "DPO"]}'
        )
        from research_navigator.agents.nodes import router_node

        state: AgentState = {"query": "Compare RLHF and DPO"}
        result = router_node(state)

    assert result["route"] == "compare_approaches"
    assert result["comparison_terms"] == ["RLHF", "DPO"]


def test_router_node_fallback_on_error() -> None:
    """Router should default to concept_explanation on API error."""
    with patch("research_navigator.agents.nodes.ChatGoogleGenerativeAI") as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.invoke.side_effect = Exception("API error")
        from research_navigator.agents.nodes import router_node

        state: AgentState = {"query": "What is attention?"}
        result = router_node(state)

    assert result["route"] == "concept_explanation"
