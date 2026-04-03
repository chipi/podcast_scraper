# Experiment Metrics Report

**Run ID:** `hybrid_ml_tier2_qwen35_35b_smoke_paragraph_v1`
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

- **Average Tokens:** 605.0000
- **Min Tokens:** 528.0000
- **Max Tokens:** 689.0000

### Performance Metrics

- **Average Latency:** 25679ms
- **Median Latency:** 23233ms
- **P95 Latency:** 35930ms
- **Avg Latency (excl. first episode):** 23116ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 54.2%
- ROUGE-2 F1: 18.2%
- ROUGE-L F1: 21.2%

**BLEU Score:** 13.5%

**Word Error Rate (WER):** 94.7%

**Embedding Cosine Similarity:** 76.2%
