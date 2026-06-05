# ADR-006: Similarity Threshold for Refusal

## Decision
Set similarity_threshold=0.525

## Context
RRF fusion scores are rank-based (not comparable across queries).
We use dense cosine similarity separately for confidence estimation.

## Evaluation
Tested on 20 queries (10 valid AI/ML, 10 out-of-domain):
- Best threshold: 0.525
- Accuracy: 90% (9/10 TP, 9/10 TN)
- 1 false positive, 1 false negative

## Consequences
Will be re-evaluated in M4 with 40-query golden set.
M3 router provides additional out-of-scope filtering before retrieval.