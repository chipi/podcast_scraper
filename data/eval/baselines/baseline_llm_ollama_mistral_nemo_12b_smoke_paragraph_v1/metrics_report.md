# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_mistral_nemo_12b_smoke_paragraph_v1`
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

- **Average Tokens:** 581.0000
- **Min Tokens:** 353.0000
- **Max Tokens:** 791.0000

### Performance Metrics

- **Average Latency:** 24586ms
- **Median Latency:** 25799ms
- **P95 Latency:** 27982ms
- **Avg Latency (excl. first episode):** 23918ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 53.4%
- ROUGE-2 F1: 18.8%
- ROUGE-L F1: 23.6%

**BLEU Score:** 12.2%

**Word Error Rate (WER):** 96.3%

**Embedding Cosine Similarity:** 77.2%
