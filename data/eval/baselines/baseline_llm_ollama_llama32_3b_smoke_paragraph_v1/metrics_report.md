# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_llama32_3b_smoke_paragraph_v1`
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

- **Average Tokens:** 544.0000
- **Min Tokens:** 419.0000
- **Max Tokens:** 627.0000

### Performance Metrics

- **Average Latency:** 7336ms
- **Median Latency:** 7297ms
- **P95 Latency:** 8869ms
- **Avg Latency (excl. first episode):** 6953ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 56.3%
- ROUGE-2 F1: 18.5%
- ROUGE-L F1: 22.6%

**BLEU Score:** 12.4%

**Word Error Rate (WER):** 93.0%

**Embedding Cosine Similarity:** 78.9%
