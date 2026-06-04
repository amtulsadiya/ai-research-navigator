"""
CLI entrypoint for the ingestion pipeline.

Commands:
  python -m research_navigator.ingest ingest   -- ingest full corpus
  python -m research_navigator.ingest validate -- validate manifest + file existence
  python -m research_navigator.ingest stats    -- show Qdrant collection stats
  python -m research_navigator.ingest reindex  -- re-ingest specific doc_ids

No business logic lives here — this file only parses args and calls pipeline.py.
This is the standard pattern: CLI = thin wrapper around library code.
Why? Because library code is testable; CLI code is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_navigator.config import get_settings
from research_navigator.logger import get_logger, setup_logging

logger = get_logger(__name__)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest the full corpus into Qdrant."""
    from research_navigator.ingest.pipeline import run_ingestion

    manifest_path = Path(args.manifest) if args.manifest else None

    results = run_ingestion(
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )

    print(json.dumps(results, indent=2))
    failed = results.get("failed", 0)
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate that all files in the manifest exist on disk."""
    from research_navigator.ingest.pipeline import load_manifest

    manifest_path = (
        Path(args.manifest) if args.manifest else Path("corpus/manifest.json")
    )
    metas = load_manifest(manifest_path)

    missing = []
    for meta in metas:
        file_path = Path("corpus") / meta.local_path
        if not file_path.exists():
            missing.append({"doc_id": meta.doc_id, "path": str(file_path)})

    if missing:
        print(f"MISSING FILES ({len(missing)}):")
        for m in missing:
            print(f"  {m['doc_id']}: {m['path']}")
        return 1

    print(f"All {len(metas)} files present. Corpus is complete.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show Qdrant collection statistics."""
    from research_navigator.ingest.pipeline import run_stats

    stats = run_stats()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    """Re-ingest specific documents by doc_id."""
    from research_navigator.ingest.pipeline import run_ingestion

    if not args.doc_ids:
        print("Error: provide --doc-ids DOC_ID1 DOC_ID2 ...")
        return 1

    results = run_ingestion(
        doc_ids=args.doc_ids,
        dry_run=args.dry_run,
    )
    print(json.dumps(results, indent=2))
    return 0


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    parser = argparse.ArgumentParser(
        prog="python -m research_navigator.ingest",
        description="AI Research Navigator — ingestion pipeline",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json (default: corpus/manifest.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but skip embedding and Qdrant writes",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest command
    # ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest full corpus into Qdrant")
    ingest_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but skip embedding and Qdrant writes",
    )
    ingest_p.set_defaults(func=cmd_ingest)

    # validate command
    validate_p = subparsers.add_parser(
        "validate", help="Validate all corpus files exist"
    )
    validate_p.set_defaults(func=cmd_validate)

    # stats command
    stats_p = subparsers.add_parser("stats", help="Show Qdrant collection statistics")
    stats_p.set_defaults(func=cmd_stats)

    # reindex command
    # reindex command
    reindex_p = subparsers.add_parser("reindex", help="Re-ingest specific documents")
    reindex_p.add_argument("--doc-ids", nargs="+", help="Document IDs to re-ingest")
    reindex_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but skip embedding and Qdrant writes",
    )
    reindex_p.set_defaults(func=cmd_reindex)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
