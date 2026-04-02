# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen35_9b_smoke_bullets_v1`
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

- **Average Tokens:** 241.2000
- **Min Tokens:** 210.0000
- **Max Tokens:** 263.0000

### Performance Metrics

- **Average Latency:** 15376ms
- **Median Latency:** 14743ms
- **P95 Latency:** 18448ms
- **Avg Latency (excl. first episode):** 14609ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_bullets_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 60.7%
- ROUGE-2 F1: 30.4%
- ROUGE-L F1: 33.3%

**BLEU Score:** 26.9%

**Word Error Rate (WER):** 83.9%

**Embedding Cosine Similarity:** 83.4%
