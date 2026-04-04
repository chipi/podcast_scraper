# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen25_32b_smoke_paragraph_v1`
**Dataset ID:** `curated_5feeds_smoke_v1`
**Episode Count:** 5

## Intrinsic Metrics

### Quality Gates

- **Boilerplate Leak Rate:** 0.0%
- **Speaker Label Leak Rate:** 0.0%
- **Speaker Name Leak Rate (WARN):** 0.0%
- **Truncation Rate:** 0.0%
- **Failed Episodes:** None

### Length Metrics

- **Average Tokens:** 403.4000
- **Min Tokens:** 365.0000
- **Max Tokens:** 425.0000

### Performance Metrics

- **Average Latency:** 58526ms
- **Median Latency:** 57314ms
- **P95 Latency:** 62711ms
- **Avg Latency (excl. first episode):** 57479ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 54.3%
- ROUGE-2 F1: 16.8%
- ROUGE-L F1: 25.4%

**BLEU Score:** 9.1%

**Word Error Rate (WER):** 88.5%

**Embedding Cosine Similarity:** 76.6%
