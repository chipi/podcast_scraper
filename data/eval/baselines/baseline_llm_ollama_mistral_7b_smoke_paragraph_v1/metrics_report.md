# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_mistral_7b_smoke_paragraph_v1`
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

- **Average Tokens:** 499.0000
- **Min Tokens:** 374.0000
- **Max Tokens:** 602.0000

### Performance Metrics

- **Average Latency:** 16086ms
- **Median Latency:** 16493ms
- **P95 Latency:** 17438ms
- **Avg Latency (excl. first episode):** 16112ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 54.0%
- ROUGE-2 F1: 18.4%
- ROUGE-L F1: 22.3%

**BLEU Score:** 11.6%

**Word Error Rate (WER):** 91.3%

**Embedding Cosine Similarity:** 71.4%
