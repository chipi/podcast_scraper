# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_mistral_nemo_12b_smoke_bullets_v1`
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

- **Average Tokens:** 197.2000
- **Min Tokens:** 171.0000
- **Max Tokens:** 222.0000

### Performance Metrics

- **Average Latency:** 16188ms
- **Median Latency:** 15578ms
- **P95 Latency:** 19095ms
- **Avg Latency (excl. first episode):** 15462ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_bullets_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 53.1%
- ROUGE-2 F1: 25.8%
- ROUGE-L F1: 27.6%

**BLEU Score:** 19.6%

**Word Error Rate (WER):** 86.3%

**Embedding Cosine Similarity:** 77.9%
