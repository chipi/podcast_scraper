# Experiment Metrics Report

**Run ID:** `hybrid_ml_tier2_qwen35_9b_smoke_paragraph_v1`
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

- **Average Tokens:** 657.2000
- **Min Tokens:** 537.0000
- **Max Tokens:** 864.0000

### Performance Metrics

- **Average Latency:** 28763ms
- **Median Latency:** 28429ms
- **P95 Latency:** 32537ms
- **Avg Latency (excl. first episode):** 27820ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 52.6%
- ROUGE-2 F1: 15.6%
- ROUGE-L F1: 18.1%

**BLEU Score:** 11.5%

**Word Error Rate (WER):** 101.2%

**Embedding Cosine Similarity:** 76.8%
