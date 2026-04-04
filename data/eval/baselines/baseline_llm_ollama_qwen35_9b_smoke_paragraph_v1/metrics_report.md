# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen35_9b_smoke_paragraph_v1`
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

- **Average Tokens:** 528.2000
- **Min Tokens:** 422.0000
- **Max Tokens:** 562.0000

### Performance Metrics

- **Average Latency:** 23937ms
- **Median Latency:** 23485ms
- **P95 Latency:** 25573ms
- **Avg Latency (excl. first episode):** 23528ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 59.2%
- ROUGE-2 F1: 20.8%
- ROUGE-L F1: 24.7%

**BLEU Score:** 12.1%

**Word Error Rate (WER):** 92.1%

**Embedding Cosine Similarity:** 75.8%
