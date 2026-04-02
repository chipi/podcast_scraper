# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen25_7b_smoke_paragraph_v1`
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

- **Average Tokens:** 491.8000
- **Min Tokens:** 437.0000
- **Max Tokens:** 613.0000

### Performance Metrics

- **Average Latency:** 13774ms
- **Median Latency:** 13043ms
- **P95 Latency:** 18873ms
- **Avg Latency (excl. first episode):** 12500ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 55.1%
- ROUGE-2 F1: 17.4%
- ROUGE-L F1: 21.4%

**BLEU Score:** 10.8%

**Word Error Rate (WER):** 92.6%

**Embedding Cosine Similarity:** 78.4%
