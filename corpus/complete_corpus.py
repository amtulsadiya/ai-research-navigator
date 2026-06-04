#!/usr/bin/env python3
"""
Complete the AI Research Navigator corpus by fetching the documents that could
not be downloaded in the sealed environment: arXiv papers (30) and lab blog
posts (3). HF Learn chapters and Lil'Log posts are already present.

Run from the corpus root (where this script and manifest.json live):

    python3 complete_corpus.py

Dependencies: requests, trafilatura, html2text
    pip install requests trafilatura html2text
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Please install dependencies: pip install requests trafilatura html2text")

try:
    import trafilatura
except ImportError:
    trafilatura = None
try:
    import html2text
except ImportError:
    html2text = None

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"

USER_AGENT = (
    "AI-Research-Navigator-Corpus-Builder/1.0 "
    "(intern assignment; one-time fetch; contact: koundinya@example.com)"
)
HEADERS = {"User-Agent": USER_AGENT}

# Polite delay between requests to each host (arxiv asks for >=3s)
ARXIV_DELAY = 3.5
BLOG_DELAY = 1.0


def fetch_arxiv_pdf(arxiv_id: str, dest: Path) -> tuple[bool, str]:
    """Fetch an arXiv PDF, verifying it's actually a PDF before saving."""
    if dest.exists() and dest.stat().st_size > 10_000:
        return True, "already present"
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        return False, f"request failed: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    if not r.content.startswith(b"%PDF"):
        return False, "response was not a PDF (arXiv may be rate-limiting; retry later)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True, f"{len(r.content)} bytes"


def fetch_blog_post(
    url: str, dest: Path, doc_id: str, title: str, source_label: str
) -> tuple[bool, str]:
    """Fetch a blog post and convert to markdown."""
    if dest.exists() and dest.stat().st_size > 1_000:
        return True, "already present"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        return False, f"request failed: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"

    html = r.text
    md_body = None

    if trafilatura is not None:
        md_body = trafilatura.extract(
            html, output_format="markdown", include_links=True
        )

    if not md_body and html2text is not None:
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = False
        md_body = h.handle(html)

    if not md_body:
        return False, "no extractor available (install trafilatura or html2text)"

    header = (
        f"# {title}\n\n" f"Source: {url}\n" f"Publisher: {source_label}\n\n" f"---\n\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + md_body, encoding="utf-8")
    return True, f"{dest.stat().st_size} bytes"


def lab_blog_post_entries() -> Iterable[dict]:
    """The 3 lab blog posts we need to fetch externally."""
    return [
        {
            "doc_id": "anthropic-mapping-mind-2024-05",
            "title": "Mapping the Mind of a Large Language Model",
            "url": "https://www.anthropic.com/research/mapping-mind-language-model",
            "publisher": "Anthropic",
            "local_path": "documents/lab-blogs/anthropic-mapping-mind-2024-05.md",
        },
        {
            "doc_id": "openai-weak-to-strong-2023-12",
            "title": "Weak-to-Strong Generalization",
            "url": "https://openai.com/index/weak-to-strong-generalization/",
            "publisher": "OpenAI",
            "local_path": "documents/lab-blogs/openai-weak-to-strong-2023-12.md",
        },
        {
            "doc_id": "deepmind-alphaproof-2024-07",
            "title": "AI Achieves Silver-Medal Standard Solving IMO Problems",
            "url": "https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/",
            "publisher": "Google DeepMind",
            "local_path": "documents/lab-blogs/deepmind-alphaproof-2024-07.md",
        },
    ]


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"manifest.json not found at {MANIFEST_PATH}", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    arxiv_docs = [
        d for d in manifest["documents"] if d["content_type"] == "arxiv_paper"
    ]

    print(f"\n=== arXiv papers ({len(arxiv_docs)}) ===")
    arxiv_failures: list[tuple[str, str]] = []
    for i, doc in enumerate(arxiv_docs, 1):
        arxiv_id = doc["doc_id"].removeprefix("arxiv-")
        dest = ROOT / doc["local_path"]
        ok, msg = fetch_arxiv_pdf(arxiv_id, dest)
        status = "OK " if ok else "FAIL"
        print(f"  [{i:2d}/{len(arxiv_docs)}] {status} {doc['doc_id']}: {msg}")
        if not ok:
            arxiv_failures.append((doc["doc_id"], msg))
        time.sleep(ARXIV_DELAY)

    blog_docs = lab_blog_post_entries()
    print(f"\n=== Lab blog posts ({len(blog_docs)}) ===")
    blog_failures: list[tuple[str, str]] = []
    for i, doc in enumerate(blog_docs, 1):
        dest = ROOT / doc["local_path"]
        ok, msg = fetch_blog_post(
            url=doc["url"],
            dest=dest,
            doc_id=doc["doc_id"],
            title=doc["title"],
            source_label=doc["publisher"],
        )
        status = "OK " if ok else "FAIL"
        print(f"  [{i}/{len(blog_docs)}] {status} {doc['doc_id']}: {msg}")
        if not ok:
            blog_failures.append((doc["doc_id"], msg))
        time.sleep(BLOG_DELAY)

    print("\n=== Summary ===")
    print(f"arXiv: {len(arxiv_docs) - len(arxiv_failures)}/{len(arxiv_docs)} succeeded")
    print(f"Lab blogs: {len(list(blog_docs)) - len(blog_failures)}/3 succeeded")
    if arxiv_failures or blog_failures:
        print("\nFailures (retry these):")
        for did, msg in arxiv_failures + blog_failures:
            print(f"  - {did}: {msg}")
        return 2
    print("\nCorpus complete. You can now zip it and hand it to the interns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
