# Embedding-provider comparison

- Ground truth: gi.json `SUPPORTED_BY` edges (Insight → Quote)
- Pairs: **1200** (1200 insights × 100 unique quotes)

## Retrieval metrics (higher is better)

| Metric | MiniLM (current default) | nomic-embed-text (Ollama) |
| --- | --- | --- |
| Recall@1 | 0.621 | 0.573 |
| Recall@5 | 0.818 | 0.790 |
| Recall@10 | 0.898 | 0.879 |
| Recall@20 | 0.955 | 0.930 |
| nDCG@1 | 0.621 | 0.573 |
| nDCG@5 | 0.725 | 0.690 |
| nDCG@10 | 0.751 | 0.719 |
| nDCG@20 | 0.766 | 0.732 |
| MRR | 0.710 | 0.674 |

## Operational characteristics

| Field | MiniLM (current default) | nomic-embed-text (Ollama) |
| --- | --- | --- |
| Provider | sentence_transformers | ollama |
| Model | sentence-transformers/all-MiniLM-L6-v2 | nomic-embed-text |
| Vector dim | 384 | 768 |
| Embed latency p50 (ms) | 7.4 | 33.0 |
| Embed latency p95 (ms) | 11.3 | 40.6 |

## Interpretation guide

- **Recall@1**: how often the model nails the exact supporting quote first try.
- **Recall@10**: how often it's in the user's top-10 (the practical UX bar).
- **MRR**: average rank-quality across all queries.
- A 5+ point Recall@10 delta is materially felt in retrieval UX.
