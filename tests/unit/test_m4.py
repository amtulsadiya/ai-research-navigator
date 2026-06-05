"""
Unit tests for M4 evaluation metrics.
Tests pure metric functions — no API calls needed.
"""

from __future__ import annotations

from research_navigator.eval.harness import (
    EvalConfig,
    QuestionResult,
    check_refusal_correctness,
    compute_summary,
    generate_markdown_report,
    precision_at_k,
    recall_at_k,
)

# ── Precision and recall tests ─────────────────────────────────────────────────


def test_precision_at_k_perfect() -> None:
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc1", "doc2", "doc3"]
    assert precision_at_k(retrieved, expected, k=3) == 1.0


def test_precision_at_k_zero() -> None:
    retrieved = ["doc4", "doc5", "doc6"]
    expected = ["doc1", "doc2", "doc3"]
    assert precision_at_k(retrieved, expected, k=3) == 0.0


def test_precision_at_k_partial() -> None:
    retrieved = ["doc1", "doc4", "doc5"]
    expected = ["doc1", "doc2"]
    result = precision_at_k(retrieved, expected, k=3)
    assert abs(result - 1 / 3) < 0.001


def test_precision_at_k_empty() -> None:
    assert precision_at_k([], ["doc1"], k=3) == 0.0
    assert precision_at_k(["doc1"], [], k=3) == 0.0


def test_recall_at_k_perfect() -> None:
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc1", "doc2"]
    assert recall_at_k(retrieved, expected, k=3) == 1.0


def test_recall_at_k_partial() -> None:
    retrieved = ["doc1", "doc4", "doc5"]
    expected = ["doc1", "doc2"]
    assert recall_at_k(retrieved, expected, k=3) == 0.5


def test_recall_at_k_zero() -> None:
    retrieved = ["doc4", "doc5"]
    expected = ["doc1", "doc2"]
    assert recall_at_k(retrieved, expected, k=3) == 0.0


# ── Refusal correctness tests ──────────────────────────────────────────────────


def test_refusal_correct_when_should_refuse() -> None:
    result = QuestionResult(
        question_id="q001",
        route="out_of_scope",
        query="pizza recipe",
        expected_doc_ids=[],
        should_refuse=True,
        was_refused=True,
    )
    assert check_refusal_correctness(result) is True


def test_refusal_incorrect_when_should_refuse_but_answered() -> None:
    result = QuestionResult(
        question_id="q001",
        route="out_of_scope",
        query="pizza recipe",
        expected_doc_ids=[],
        should_refuse=True,
        was_refused=False,
    )
    assert check_refusal_correctness(result) is False


def test_refusal_correct_when_should_answer() -> None:
    result = QuestionResult(
        question_id="q002",
        route="concept_explanation",
        query="what is attention?",
        expected_doc_ids=["arxiv-1706.03762"],
        should_refuse=False,
        was_refused=False,
    )
    assert check_refusal_correctness(result) is True


def test_refusal_incorrect_when_should_answer_but_refused() -> None:
    result = QuestionResult(
        question_id="q002",
        route="concept_explanation",
        query="what is attention?",
        expected_doc_ids=["arxiv-1706.03762"],
        should_refuse=False,
        was_refused=True,
    )
    assert check_refusal_correctness(result) is False


# ── Summary computation tests ──────────────────────────────────────────────────


def make_result(
    should_refuse: bool = False,
    was_refused: bool = False,
    route: str = "concept_explanation",
    actual_route: str = "concept_explanation",
    precision: float = 0.5,
    recall: float = 0.5,
    latency: float = 1000.0,
    error: str | None = None,
) -> QuestionResult:
    r = QuestionResult(
        question_id="test",
        route=route,
        query="test query",
        expected_doc_ids=["doc1"],
        should_refuse=should_refuse,
    )
    r.was_refused = was_refused
    r.actual_route = actual_route
    r.route_correct = route == actual_route
    r.refusal_correct = was_refused == should_refuse
    r.precision_at_k = precision
    r.recall_at_k = recall
    r.latency_ms = latency
    r.error = error
    return r


def test_compute_summary_perfect() -> None:
    config = EvalConfig(name="test", use_metadata_filters=True)
    results = [make_result() for _ in range(10)]
    summary = compute_summary(results, config)

    assert summary["route_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["errors"] == 0


def test_compute_summary_with_errors() -> None:
    config = EvalConfig(name="test", use_metadata_filters=True)
    results = [make_result() for _ in range(8)] + [
        make_result(error="API error") for _ in range(2)
    ]
    summary = compute_summary(results, config)
    assert summary["errors"] == 2


def test_compute_summary_latency() -> None:
    config = EvalConfig(name="test", use_metadata_filters=True)
    results = [make_result(latency=float(i * 100)) for i in range(1, 11)]
    summary = compute_summary(results, config)
    assert summary["latency_ms"]["p50"] > 0
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]


def test_generate_markdown_report() -> None:
    config_a = EvalConfig(
        name="config_a", use_metadata_filters=True, description="With filters"
    )
    config_b = EvalConfig(
        name="config_b", use_metadata_filters=False, description="Without filters"
    )

    results = [make_result() for _ in range(5)]
    summary_a = compute_summary(results, config_a)
    summary_b = compute_summary(results, config_b)

    report = generate_markdown_report([summary_a, summary_b], {})

    assert "# AI Research Navigator" in report
    assert "config_a" in report
    assert "config_b" in report
    assert "Precision" in report
    assert "Known Limitations" in report
