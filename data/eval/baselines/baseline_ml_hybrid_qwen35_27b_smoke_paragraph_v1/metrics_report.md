# Experiment Metrics Report

**Run ID:** `hybrid_ml_tier2_qwen35_27b_smoke_paragraph_v1`
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

- **Average Tokens:** 738.8000
- **Min Tokens:** 607.0000
- **Max Tokens:** 947.0000

### Performance Metrics

- **Average Latency:** 81017ms
- **Median Latency:** 81197ms
- **P95 Latency:** 106687ms
- **Avg Latency (excl. first episode):** 80972ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 54.2%
- ROUGE-2 F1: 15.8%
- ROUGE-L F1: 20.4%

**BLEU Score:** 11.9%

**Word Error Rate (WER):** 109.0%

**Embedding Cosine Similarity:** 76.5%
