# Experiment Metrics Report

**Run ID:** `hybrid_ml_tier2_qwen25_32b_smoke_paragraph_v1`
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

- **Average Tokens:** 412.4000
- **Min Tokens:** 285.0000
- **Max Tokens:** 461.0000

### Performance Metrics

- **Average Latency:** 42547ms
- **Median Latency:** 44359ms
- **P95 Latency:** 50041ms
- **Avg Latency (excl. first episode):** 40673ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 48.5%
- ROUGE-2 F1: 14.8%
- ROUGE-L F1: 18.2%

**BLEU Score:** 7.7%

**Word Error Rate (WER):** 92.6%

**Embedding Cosine Similarity:** 81.2%
