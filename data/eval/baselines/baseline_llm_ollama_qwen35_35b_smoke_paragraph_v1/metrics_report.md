# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen35_35b_smoke_paragraph_v1`
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

- **Average Tokens:** 598.8000
- **Min Tokens:** 491.0000
- **Max Tokens:** 707.0000

### Performance Metrics

- **Average Latency:** 20393ms
- **Median Latency:** 19834ms
- **P95 Latency:** 25097ms
- **Avg Latency (excl. first episode):** 19217ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 65.1%
- ROUGE-2 F1: 26.0%
- ROUGE-L F1: 29.9%

**BLEU Score:** 18.4%

**Word Error Rate (WER):** 91.7%

**Embedding Cosine Similarity:** 81.8%
