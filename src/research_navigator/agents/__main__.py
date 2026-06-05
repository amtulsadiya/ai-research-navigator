"""
CLI for the M3 agent.

Commands:
  python -m research_navigator.agents query "your question here"
  python -m research_navigator.agents visualize
  python -m research_navigator.agents test
"""

from __future__ import annotations

import argparse
import sys

from research_navigator.config import get_settings
from research_navigator.logger import get_logger, setup_logging

logger = get_logger(__name__)


def cmd_query(args: argparse.Namespace) -> int:
    """Run a single query through the agent."""
    from research_navigator.agents.graph import run_query

    query = args.query
    print(f"\nQuery: {query}")
    print("-" * 60)

    state = run_query(query)

    print(f"Route: {state.get('route', 'unknown')}")
    print(f"Reasoning: {state.get('router_reasoning', '')}")
    print()
    print(state.get("answer", "No answer generated"))
    print()

    citations = state.get("citations", [])
    if citations:
        print("---")
        print("Sources:")
        for c in citations:
            print(
                f"[{c.get('number', '?')}] {c.get('title', '')} ({c.get('year', '')})"
            )
            print(f"    {c.get('source_url', '')}")

    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize the agent graph."""
    from research_navigator.agents.graph import visualize_graph

    visualize_graph()
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run a quick test across all 6 routes."""
    from research_navigator.agents.graph import run_query

    test_queries = [
        ("concept_explanation", "What is the attention mechanism?"),
        ("paper_deep_dive", "Tell me about the Attention Is All You Need paper"),
        ("compare_approaches", "Compare RLHF and DPO for alignment"),
        ("recent_developments", "What are the latest developments in LLM reasoning?"),
        ("find_papers", "Recommend foundational papers on transformers"),
        ("out_of_scope", "What is the best pizza recipe?"),
    ]

    print("\nM3 Agent Route Test")
    print("=" * 60)

    passed = 0
    for expected_route, query in test_queries:
        print(f"\nQuery: {query[:60]}")
        state = run_query(query)
        actual_route = state.get("route", "unknown")
        correct = actual_route == expected_route
        status = "✅" if correct else "❌"
        print(f"{status} Expected: {expected_route} | Got: {actual_route}")
        if correct:
            passed += 1

    print(f"\nResult: {passed}/{len(test_queries)} routes correct")
    return 0 if passed == len(test_queries) else 1


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    parser = argparse.ArgumentParser(
        prog="python -m research_navigator.agents",
        description="AI Research Navigator — agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_p = subparsers.add_parser("query", help="Run a query")
    query_p.add_argument("query", type=str, help="The question to ask")
    query_p.set_defaults(func=cmd_query)

    viz_p = subparsers.add_parser("visualize", help="Visualize the agent graph")
    viz_p.set_defaults(func=cmd_visualize)

    test_p = subparsers.add_parser("test", help="Test all 6 routes")
    test_p.set_defaults(func=cmd_test)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
