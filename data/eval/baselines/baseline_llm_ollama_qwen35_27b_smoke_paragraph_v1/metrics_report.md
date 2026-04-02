# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen35_27b_smoke_paragraph_v1`
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

- **Average Tokens:** 661.2000
- **Min Tokens:** 569.0000
- **Max Tokens:** 724.0000

### Performance Metrics

- **Average Latency:** 82826ms
- **Median Latency:** 84128ms
- **P95 Latency:** 91128ms
- **Avg Latency (excl. first episode):** 80750ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 61.5%
- ROUGE-2 F1: 23.5%
- ROUGE-L F1: 29.2%

**BLEU Score:** 18.5%

**Word Error Rate (WER):** 96.9%

**Embedding Cosine Similarity:** 80.4%
