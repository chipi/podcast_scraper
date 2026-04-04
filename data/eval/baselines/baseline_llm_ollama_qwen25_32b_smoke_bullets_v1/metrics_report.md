# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen25_32b_smoke_bullets_v1`
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

- **Average Tokens:** 196.4000
- **Min Tokens:** 180.0000
- **Max Tokens:** 225.0000

### Performance Metrics

- **Average Latency:** 40894ms
- **Median Latency:** 39732ms
- **P95 Latency:** 49325ms
- **Avg Latency (excl. first episode):** 38786ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_bullets_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 52.9%
- ROUGE-2 F1: 24.5%
- ROUGE-L F1: 30.1%

**BLEU Score:** 20.0%

**Word Error Rate (WER):** 83.8%

**Embedding Cosine Similarity:** 82.4%
