# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_phi3_mini_smoke_bullets_v1`
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

- **Average Tokens:** 228.2000
- **Min Tokens:** 188.0000
- **Max Tokens:** 298.0000

### Performance Metrics

- **Average Latency:** 7985ms
- **Median Latency:** 7873ms
- **P95 Latency:** 9819ms
- **Avg Latency (excl. first episode):** 7526ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_bullets_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 53.4%
- ROUGE-2 F1: 25.1%
- ROUGE-L F1: 30.7%

**BLEU Score:** 19.6%

**Word Error Rate (WER):** 86.4%

**Embedding Cosine Similarity:** 78.4%
