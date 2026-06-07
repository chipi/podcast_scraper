# Embedding-provider comparison

- Ground truth: gi.json `SUPPORTED_BY` edges (Insight → Quote)
- Pairs: **197** (100 insights × 197 unique quotes)

## Retrieval metrics (higher is better)

| Metric | MiniLM (current default) | nomic-embed-text (Ollama) |
| --- | --- | --- |
| Recall@1 | 0.360 | 0.365 |
| Recall@5 | 0.645 | 0.665 |
| Recall@10 | 0.726 | 0.751 |
| Recall@20 | 0.817 | 0.838 |
| nDCG@1 | 0.360 | 0.365 |
| nDCG@5 | 0.507 | 0.520 |
| nDCG@10 | 0.533 | 0.548 |
| nDCG@20 | 0.556 | 0.570 |
| MRR | 0.483 | 0.494 |

## Operational characteristics

| Field | MiniLM (current default) | nomic-embed-text (Ollama) |
| --- | --- | --- |
| Provider | sentence_transformers | ollama |
| Model | sentence-transformers/all-MiniLM-L6-v2 | nomic-embed-text |
| Vector dim | 384 | 768 |
| Embed latency p50 (ms) | 5.6 | 18.9 |
| Embed latency p95 (ms) | 20.2 | 24.0 |

## Interpretation guide

- **Recall@1**: how often the model nails the exact supporting quote first try.
- **Recall@10**: how often it's in the user's top-10 (the practical UX bar).
- **MRR**: average rank-quality across all queries.
- A 5+ point Recall@10 delta is materially felt in retrieval UX.
