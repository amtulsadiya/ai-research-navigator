"""
eval/__main__.py — CLI entry for evaluation harness.
Run: uv run python -m research_navigator.eval
  or: make eval
"""

from __future__ import annotations


def main() -> None:
    from research_navigator.eval.harness import main as run_eval

    run_eval()


if __name__ == "__main__":
    main()
