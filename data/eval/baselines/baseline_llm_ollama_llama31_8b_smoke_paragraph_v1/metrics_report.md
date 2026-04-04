# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_llama31_8b_smoke_paragraph_v1`
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

- **Average Tokens:** 565.6000
- **Min Tokens:** 487.0000
- **Max Tokens:** 761.0000

### Performance Metrics

- **Average Latency:** 15724ms
- **Median Latency:** 14962ms
- **P95 Latency:** 18369ms
- **Avg Latency (excl. first episode):** 15309ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 57.4%
- ROUGE-2 F1: 18.3%
- ROUGE-L F1: 23.0%

**BLEU Score:** 13.4%

**Word Error Rate (WER):** 95.1%

**Embedding Cosine Similarity:** 77.9%
