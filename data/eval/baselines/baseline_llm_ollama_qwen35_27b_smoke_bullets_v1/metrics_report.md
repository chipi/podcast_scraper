# Experiment Metrics Report

**Run ID:** `autoresearch_prompt_ollama_qwen35_27b_smoke_bullets_v1`
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

- **Average Tokens:** 261.4000
- **Min Tokens:** 228.0000
- **Max Tokens:** 284.0000

### Performance Metrics

- **Average Latency:** 53610ms
- **Median Latency:** 53002ms
- **P95 Latency:** 63474ms
- **Avg Latency (excl. first episode):** 51145ms

## vs Reference Metrics

### vs silver_sonnet46_smoke_bullets_v1

**Reference Quality:** silver

**ROUGE Scores:**

- ROUGE-1 F1: 63.4%
- ROUGE-2 F1: 33.3%
- ROUGE-L F1: 33.5%

**BLEU Score:** 30.2%

**Word Error Rate (WER):** 84.5%

**Embedding Cosine Similarity:** 84.7%
