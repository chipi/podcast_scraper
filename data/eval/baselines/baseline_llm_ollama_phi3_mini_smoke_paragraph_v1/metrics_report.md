# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_phi3_mini_smoke_paragraph_v1`
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

- **Average Tokens:** 972.6000
- **Min Tokens:** 906.0000
- **Max Tokens:** 1044.0000

### Performance Metrics

- **Average Latency:** 16486ms
- **Median Latency:** 16322ms
- **P95 Latency:** 18881ms
- **Avg Latency (excl. first episode):** 15888ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 50.3%
- ROUGE-2 F1: 12.1%
- ROUGE-L F1: 18.0%

**BLEU Score:** 6.9%

**Word Error Rate (WER):** 141.1%

**Embedding Cosine Similarity:** 79.6%
