# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_mistral_7b_smoke_bullets_v1`
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

- **Average Tokens:** 184.4000
- **Min Tokens:** 170.0000
- **Max Tokens:** 200.0000

### Performance Metrics

- **Average Latency:** 11150ms
- **Median Latency:** 11156ms
- **P95 Latency:** 12823ms
- **Avg Latency (excl. first episode):** 10731ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_bullets_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 52.5%
- ROUGE-2 F1: 28.0%
- ROUGE-L F1: 27.8%

**BLEU Score:** 20.1%

**Word Error Rate (WER):** 88.0%

**Embedding Cosine Similarity:** 81.2%
