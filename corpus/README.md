# AI Research Navigator — Corpus Package

This is the sealed corpus for the **AI Research Navigator** intern assignment. It contains 50 documents (or rather, *will* contain 50 once `complete_corpus.py` is run — see §3) plus a manifest.

## 1. Layout

```
corpus/
├── README.md                ← this file
├── manifest.json            ← metadata for all 50 documents
├── complete_corpus.py       ← script to fetch arXiv + lab blog content
└── documents/
    ├── arxiv/               ← 30 PDFs (populated by complete_corpus.py)
    ├── hf-learn/            ← 12 markdown chapters (already present)
    ├── lillog/              ← 5 markdown posts (already present)
    └── lab-blogs/           ← 3 markdown posts (populated by complete_corpus.py)
```

## 2. What is already in the zip

| Source | Count | Format | Status |
|---|---|---|---|
| Hugging Face Learn (NLP, Agents, Deep RL) | 12 | `.md` | ✅ Included |
| Lil'Log (Lilian Weng) | 5 | `.md` | ✅ Included |
| arXiv papers | 30 | `.pdf` | ⏳ Fetched by `complete_corpus.py` |
| Lab blog posts (Anthropic, OpenAI, DeepMind) | 3 | `.md` | ⏳ Fetched by `complete_corpus.py` |
| **Total** | **50** | — | — |

The HF Learn and Lil'Log content was assembled from each project's public GitHub source (`huggingface/course`, `huggingface/agents-course`, `huggingface/deep-rl-class`, `lilianweng/lilianweng.github.io`). Each chapter or unit was concatenated from its constituent pages/sections into a single Markdown file. Lil'Log posts were extracted from the published HTML and converted to Markdown; some footer cruft (share buttons, navigation links) remains and should be handled by the ingestion parser.

## 3. Completing the corpus

The arXiv and lab-blog domains were not reachable in the environment that built this package. Run the following on a machine with normal internet access:

```bash
# 1. Unzip
unzip ai-research-navigator-corpus.zip
cd ai-research-navigator-corpus

# 2. Install dependencies
pip install requests trafilatura html2text

# 3. Run the completion script
python3 complete_corpus.py
```

The script:
- Fetches all 30 arXiv PDFs from `https://arxiv.org/pdf/<id>.pdf`.
- Fetches all 3 lab blog posts and converts them to Markdown.
- Honours arXiv's request to pace bulk downloads (≥ 3 s between requests).
- Identifies itself with a clear User-Agent.
- Is idempotent — re-running it skips files already present.
- Exits non-zero if any document fails to fetch.

End-to-end runtime: roughly 3 minutes (30 arXiv papers × 3.5 s delay ≈ 1.75 min + downloads + 3 blog posts).

## 4. Verifying the corpus before handing it to interns

After `complete_corpus.py` finishes:

```bash
# Every document in the manifest should have a corresponding file on disk
python3 -c "
import json, pathlib
m = json.load(open('manifest.json'))
missing = [d['local_path'] for d in m['documents'] if not pathlib.Path(d['local_path']).exists()]
print(f'Missing: {len(missing)}')
for p in missing: print(f'  {p}')
"
```

Expected output: `Missing: 0`.

## 5. Known caveats — please review

Two arXiv IDs in the manifest were flagged at curation time as "verify before use":

- `arxiv-2408.00118` — Gemma 2 technical report. Confirm the ID resolves to the paper titled "Gemma 2: Improving Open Language Models at a Practical Size".
- `arxiv-2501.12948` — DeepSeek-R1 paper. Confirm the ID resolves to "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning".

If either fails verification, edit `manifest.json` to point to the correct ID and re-run `complete_corpus.py`. Easy substitutes: Phi-4 technical report, Qwen2.5 technical report, or a Llama 3.1/3.2/3.3 follow-up.

Lil'Log posts contain some residual HTML-to-Markdown noise (share buttons, tag links, navigation cruft at the end of each post). This is realistic — the ingestion pipeline will need to handle some markdown noise anyway. If you prefer cleaner inputs, you can manually trim the trailing navigation/share blocks from each `lillog/*.md` file.

HF Learn course structures evolve over time. The chapter numbering and unit titles in `manifest.json` were correct as of the build date; if Hugging Face renames or restructures a chapter, the `source_url` in the manifest may 404. The local Markdown file is what counts for ingestion; the URL is only for citation rendering.

## 6. Re-distributing this corpus

The documents in this corpus are the work of their respective authors. They are included for the educational purpose of this internal intern assignment under fair-use understanding. **Do not re-distribute this package publicly.** When the system surfaces an answer to a learner, it must cite the original source URL, not paraphrase as if the wording were ours.
