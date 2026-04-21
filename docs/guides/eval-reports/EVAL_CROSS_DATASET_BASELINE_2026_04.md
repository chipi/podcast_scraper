# Cross-Dataset Scoring Baseline (April 2026)

> **Purpose:** Establish baseline scores across providers, datasets, and pipeline
> layers (summary, GI, KG) for measuring future improvements.

| Field | Value |
| ----- | ----- |
| **Date** | 2026-04-19 |
| **Datasets** | Synthetic benchmark (5 eps), QMSum (35 meetings) |
| **Silver refs** | Sonnet 4.6 (summary bullets, GI insights, KG topics) |
| **Providers** | gemini-2.5-flash-lite, gpt-4o-mini, claude-haiku-4-5, qwen3.5:9b, BART+LED |

## Summary quality (synthetic benchmark, bundled bullets)

| Provider | ROUGE-L | Embed cosine | Coverage | GI insight cov | KG topic cov |
| -------- | :-----: | :----------: | :------: | :------------: | :----------: |
| gemini-2.5-flash-lite | 36.8% | 84.3% | 77.4% | 80% | 31% |
| gpt-4o-mini | 32.1% | 76.9% | 60.3% | 65% | 31% |
| claude-haiku-4-5 | 39.3% | 84.9% | 103.5% | 75% | 38% |
| qwen3.5:9b (Ollama) | 35.8% | 83.4% | 101.5% | 70% | 38% |
| BART+LED (local ML) | 19.4% | 61.0% | 39.0% | 8% | 0% |

### Key observations

- **GI insight coverage** ranges 65-80% for LLM providers. Gemini leads at 80%.
  BART+LED at 8% — local ML summaries miss most insights because they're
  extractive, not abstractive.
- **KG topic coverage (from summary bullets)** is 31-38% — but this is misleading.
  See actual KG pipeline scores below.
- **Coverage > 100%** for claude/qwen means their summaries are longer than the
  silver reference, producing more bullets that happen to match.
- **Embedding cosine** is the most stable metric (61-85%), less affected by
  length mismatch than ROUGE-L.

## KG topic coverage (actual KG pipeline, not summary bullets)

The table above used summary bullets as topic proxies (31-38%). When scoring
**actual KG pipeline output** from pipeline validation runs, coverage is much higher:

| Provider | KG topic cov | avg similarity |
| -------- | :----------: | :------------: |
| qwen3.5:9b (Ollama) | **79%** | 0.779 |
| mistral-small (API) | 71% | 0.733 |
| gemini-2.5-flash-lite | 65% | 0.710 |
| gpt-4o-mini | 65% | 0.721 |
| claude-haiku-4-5 | 65% | 0.752 |
| gemma2:9b (Ollama) | 48% | 0.600 |
| mistral:7b (Ollama) | 46% | 0.599 |

### Key findings

- **KG v2 prompt works well for 9b+ models.** qwen3.5:9b leads at 79%.
- **Cloud providers cluster at 65%.** Consistent across gemini/openai/anthropic.
- **Smaller Ollama models (7b) lag at 46-48%.** These may need prompt adaptation
  or are fundamentally limited by model capacity.
- **Summary-bullet proxy underestimates KG quality by ~2x.** The 31-38% scores
  from the first table are not the real KG baseline.

## QMSum cross-dataset (meetings, not podcasts)

| Provider | Format | ROUGE-L |
| -------- | ------ | :-----: |
| gemini-2.5-flash-lite | Bullets | 15.4% |
| gemini-2.5-flash-lite | Paragraph | 14.0% |

### Why QMSum scores are low

1. **Length mismatch:** QMSum gold refs are 100-300 words; our prompts produce
   300-600 words. ROUGE-L penalizes heavily.
2. **Domain shift:** Prompts tuned for podcast conversations, not academic meetings.
3. **Structure:** Meeting transcripts lack podcast structure (host/guest, topics).

QMSum is useful for testing generalization but not for absolute quality comparison.
See [EVAL_TIER2_QMSUM_2026_04.md](EVAL_TIER2_QMSUM_2026_04.md) for detailed analysis.

## Metric definitions

| Metric | What it measures | Good for |
| ------ | ---------------- | -------- |
| ROUGE-L | Longest common subsequence overlap | Summary faithfulness |
| Embed cosine | Semantic similarity (all-MiniLM-L6-v2) | Meaning preservation |
| Coverage | Bullet count ratio (pred/silver) | Length calibration |
| GI insight cov | % of silver insights matched by summary bullets (cos > 0.65) | Insight capture quality |
| KG topic cov | % of silver KG topics matched by summary-derived topics (cos > 0.65) | Topic extraction quality |

## What this enables

- **KG prompt tuning (#590):** actual KG topic coverage is 65-79% for strong models.
  Small Ollama models (7b) at 46-48% may benefit from prompt tuning. Cloud providers
  at 65% could potentially improve to 75%+ with better prompts.
- **SummLlama (#571):** benchmark against BART+LED baseline (19.4% ROUGE-L, 8% GI coverage).
- **Registry (#593):** data-backed defaults — qwen3.5:9b at 79% KG + gemini at 80% GI.

## How to reproduce

```bash
# GI insight coverage
python scripts/eval/score/score_gi_insight_coverage.py \
  --run-id <run> --silver silver_sonnet46_gi_benchmark_v2 \
  --dataset curated_5feeds_benchmark_v2

# KG topic coverage
python scripts/eval/score/score_kg_topic_coverage.py \
  --run-id <run> --silver silver_sonnet46_kg_benchmark_v2 \
  --dataset curated_5feeds_benchmark_v2
```
