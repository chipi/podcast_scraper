# Embedding-provider comparison

- Ground truth: gi.json `SUPPORTED_BY` edges (Insight → Quote)
- Pairs: **1200** (1200 insights × 100 unique quotes)

## Retrieval metrics (higher is better)

| Metric | MiniLM (current default) | nomic-embed-text (Ollama) |
| --- | --- | --- |
| Recall@1 | 0.331 | 0.364 |
| Recall@5 | 0.536 | 0.631 |
| Recall@10 | 0.645 | 0.741 |
| Recall@20 | 0.755 | 0.845 |
| nDCG@1 | 0.331 | 0.364 |
| nDCG@5 | 0.436 | 0.506 |
| nDCG@10 | 0.471 | 0.542 |
| nDCG@20 | 0.499 | 0.568 |
| MRR | 0.431 | 0.491 |

## Operational characteristics

| Field | MiniLM (current default) | nomic-embed-text (Ollama) |
| --- | --- | --- |
| Provider | sentence_transformers | ollama |
| Model | sentence-transformers/all-MiniLM-L6-v2 | nomic-embed-text |
| Vector dim | 384 | 768 |
| Embed latency p50 (ms) | 5.8 | 18.2 |
| Embed latency p95 (ms) | 15.6 | 137.6 |

## Interpretation guide

- **Recall@1**: how often the model nails the exact supporting quote first try.
- **Recall@10**: how often it's in the user's top-10 (the practical UX bar).
- **MRR**: average rank-quality across all queries.
- A 5+ point Recall@10 delta is materially felt in retrieval UX.
