# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_gemma2_9b_smoke_paragraph_v1`
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

- **Average Tokens:** 460.0000
- **Min Tokens:** 410.0000
- **Max Tokens:** 563.0000

### Performance Metrics

- **Average Latency:** 16449ms
- **Median Latency:** 16055ms
- **P95 Latency:** 18432ms
- **Avg Latency (excl. first episode):** 15953ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 50.4%
- ROUGE-2 F1: 13.0%
- ROUGE-L F1: 21.1%

**BLEU Score:** 7.1%

**Word Error Rate (WER):** 90.3%

**Embedding Cosine Similarity:** 78.0%
