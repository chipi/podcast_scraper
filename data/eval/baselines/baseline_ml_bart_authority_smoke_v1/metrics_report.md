# Experiment Metrics Report

**Run ID:** `baseline_ml_dev_authority`
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

- **Average Tokens:** 217.2000
- **Min Tokens:** 209.0000
- **Max Tokens:** 227.0000

### Performance Metrics

- **Average Latency:** 30019ms
- **Median Latency:** 26897ms
- **P95 Latency:** 44079ms
- **Avg Latency (excl. first episode):** 26504ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 35.0%
- ROUGE-2 F1: 12.7%
- ROUGE-L F1: 18.1%

**BLEU Score:** 4.0%

**Word Error Rate (WER):** 92.4%

**Embedding Cosine Similarity:** 72.4%
