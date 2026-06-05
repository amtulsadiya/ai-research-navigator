"""
M2 Acceptance Test — 20 questions

Tests the acceptance criteria:
- Every answer either carries valid citations OR refuses
- No claim appears without attribution
- No citation is fabricated (every [N] traces to a real retrieved chunk)
- Out-of-domain queries are refused

Run: uv run python tests/acceptance/test_m2_acceptance.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from research_navigator.retrieve.pipeline import query

# ── 20 test questions ──────────────────────────────────────────────────────────

# 15 valid AI/ML questions — should be ANSWERED with citations
VALID_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "How does RLHF work for fine-tuning language models?",
    "What is retrieval augmented generation?",
    "Explain the difference between GPT and BERT architectures",
    "What are the key contributions of the Llama 2 paper?",
    "How does chain of thought prompting improve reasoning?",
    "What is the transformer encoder-decoder architecture?",
    "How does reinforcement learning work in the context of LLMs?",
    "What are the challenges of long context in language models?",
    "Explain instruction tuning for language models",
    "What is the role of tokenization in NLP models?",
    "How do mixture of experts models work?",
    "What is constitutional AI and how does it relate to alignment?",
    "What are embeddings and how are they used in NLP?",
    "How does the self-attention mechanism scale with sequence length?",
]

# 5 out-of-domain questions — should be REFUSED
INVALID_QUESTIONS = [
    "What is the best pizza recipe?",
    "How do I fix my car engine?",
    "What is the weather in Mumbai today?",
    "How to cook biryani at home?",
    "Who won the cricket world cup?",
]


@dataclass
class TestResult:
    question: str
    expected: str  # "answered" or "refused"
    actual: str  # "answered" or "refused"
    passed: bool
    top_score: float
    citation_count: int
    has_citation_markers: bool
    citations_traceable: bool
    answer_preview: str


def check_citation_markers(answer: str) -> bool:
    """Check if answer contains at least one [N] citation marker."""
    import re

    return bool(re.search(r"\[\d+\]", answer))


def check_citations_traceable(answer: object) -> bool:
    """
    Verify every [N] in the answer maps to a real citation.
    Every number used as [N] must exist in the citations list.
    """
    import re

    if answer.was_refused:  # type: ignore[union-attr]
        return True  # Refused answers don't need citations

    answer_text = answer.answer  # type: ignore[union-attr]
    citation_numbers = set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text))
    available_numbers = set(c.number for c in answer.citations)  # type: ignore[union-attr]

    # Every [N] in the answer must have a corresponding citation
    fabricated = citation_numbers - available_numbers
    return len(fabricated) == 0


def run_acceptance_tests() -> None:
    print("=" * 60)
    print("M2 ACCEPTANCE TEST — 20 Questions")
    print("=" * 60)
    print()

    results: list[TestResult] = []

    # Test valid questions
    print("--- VALID AI/ML QUESTIONS (should be ANSWERED) ---")
    for i, q_text in enumerate(VALID_QUESTIONS, 1):
        print(f"[{i:2d}/20] Testing: {q_text[:55]}...")
        try:
            result = query(q_text)
            actual = "refused" if result.was_refused else "answered"
            expected = "answered"
            passed = actual == expected

            has_markers = (
                check_citation_markers(result.answer)
                if not result.was_refused
                else True
            )
            traceable = check_citations_traceable(result)

            # If answered, must have citation markers
            if not result.was_refused and not has_markers:
                passed = False

            results.append(
                TestResult(
                    question=q_text,
                    expected=expected,
                    actual=actual,
                    passed=passed and traceable,
                    top_score=result.top_score,
                    citation_count=len(result.citations),
                    has_citation_markers=has_markers,
                    citations_traceable=traceable,
                    answer_preview=result.answer[:100].replace("\n", " "),
                )
            )

            status = "✅ PASS" if passed and traceable else "❌ FAIL"
            print(
                f"        {status} | score={result.top_score:.3f} | citations={len(result.citations)} | markers={'yes' if has_markers else 'NO'}"
            )

        except Exception as e:
            print(f"        ❌ ERROR: {e}")
            results.append(
                TestResult(
                    question=q_text,
                    expected="answered",
                    actual="error",
                    passed=False,
                    top_score=0.0,
                    citation_count=0,
                    has_citation_markers=False,
                    citations_traceable=False,
                    answer_preview=str(e)[:100],
                )
            )

    print()
    print("--- OUT-OF-DOMAIN QUESTIONS (should be REFUSED) ---")
    for i, q_text in enumerate(INVALID_QUESTIONS, 1):
        print(f"[{i+15:2d}/20] Testing: {q_text[:55]}...")
        try:
            result = query(q_text)
            actual = "refused" if result.was_refused else "answered"
            expected = "refused"
            passed = actual == expected

            results.append(
                TestResult(
                    question=q_text,
                    expected=expected,
                    actual=actual,
                    passed=passed,
                    top_score=result.top_score,
                    citation_count=len(result.citations),
                    has_citation_markers=False,
                    citations_traceable=True,
                    answer_preview=result.answer[:100].replace("\n", " "),
                )
            )

            status = "✅ PASS" if passed else "❌ FAIL"
            print(
                f"        {status} | score={result.top_score:.3f} | {'correctly refused' if passed else 'WRONGLY ANSWERED'}"
            )

        except Exception as e:
            print(f"        ❌ ERROR: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    valid_results = results[:15]
    invalid_results = results[15:]

    valid_passed = sum(1 for r in valid_results if r.passed)
    invalid_passed = sum(1 for r in invalid_results if r.passed)

    print(f"Total:          {passed}/{total} passed ({100*passed//total}%)")
    print(f"Valid queries:  {valid_passed}/15 answered with citations")
    print(f"Refusal:        {invalid_passed}/5 out-of-domain correctly refused")
    print()

    # Citation integrity
    answered = [r for r in valid_results if r.actual == "answered"]
    with_markers = sum(1 for r in answered if r.has_citation_markers)
    traceable = sum(1 for r in answered if r.citations_traceable)

    print(
        f"Citation markers present:    {with_markers}/{len(answered)} answered queries"
    )
    print(f"All citations traceable:     {traceable}/{len(answered)} answered queries")
    print()

    # Failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("FAILURES:")
        for r in failures:
            print(f"  ❌ [{r.expected}→{r.actual}] {r.question[:60]}")
            print(
                f"     score={r.top_score:.3f} markers={r.has_citation_markers} traceable={r.citations_traceable}"
            )
    else:
        print("🎉 ALL TESTS PASSED")

    print()

    # Save report
    report = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{100*passed//total}%",
        "valid_queries": {"passed": valid_passed, "total": 15},
        "refusal": {"passed": invalid_passed, "total": 5},
        "citation_integrity": {
            "with_markers": with_markers,
            "traceable": traceable,
            "total_answered": len(answered),
        },
        "results": [
            {
                "question": r.question,
                "expected": r.expected,
                "actual": r.actual,
                "passed": r.passed,
                "top_score": r.top_score,
                "citation_count": r.citation_count,
            }
            for r in results
        ],
    }

    report_path = Path("eval/m2_acceptance_report.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {report_path}")

    # Exit with error if any failures
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_acceptance_tests()
