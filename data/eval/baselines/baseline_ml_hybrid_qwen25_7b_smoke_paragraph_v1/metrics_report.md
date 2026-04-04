# Experiment Metrics Report

**Run ID:** `hybrid_ml_tier2_qwen25_7b_smoke_paragraph_v1`
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

- **Average Tokens:** 397.2000
- **Min Tokens:** 355.0000
- **Max Tokens:** 418.0000

### Performance Metrics

- **Average Latency:** 15727ms
- **Median Latency:** 13559ms
- **P95 Latency:** 24927ms
- **Avg Latency (excl. first episode):** 13428ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 46.4%
- ROUGE-2 F1: 13.8%
- ROUGE-L F1: 18.4%

**BLEU Score:** 7.5%

**Word Error Rate (WER):** 93.1%

**Embedding Cosine Similarity:** 77.2%
