# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_mistral_small3_2_smoke_paragraph_v1`
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

- **Average Tokens:** 470.2000
- **Min Tokens:** 421.0000
- **Max Tokens:** 511.0000

### Performance Metrics

- **Average Latency:** 45525ms
- **Median Latency:** 45791ms
- **P95 Latency:** 49234ms
- **Avg Latency (excl. first episode):** 44598ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 58.1%
- ROUGE-2 F1: 19.3%
- ROUGE-L F1: 25.2%

**BLEU Score:** 11.4%

**Word Error Rate (WER):** 89.0%

**Embedding Cosine Similarity:** 77.1%
