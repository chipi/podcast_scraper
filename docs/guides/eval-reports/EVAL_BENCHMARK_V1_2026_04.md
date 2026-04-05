# Evaluation Report: Benchmark v1 (April 2026)

> **First full benchmark sweep** — 6 cloud APIs + 12 Ollama local models, both paragraph
> and bullet JSON output, on the 10-episode benchmark dataset. Uses the new Sonnet 4.6
> silver references. Extends the [April 2026 smoke report](EVAL_SMOKE_V1_2026_04.md)
> to production-scale: 10 episodes gives more stable rankings than 5.

| Field | Value |
| ----- | ----- |
| **Date** | April 2026 |
| **Dataset** | `curated_5feeds_benchmark_v1` (10 episodes, 5 feeds) |
| **Reference (paragraphs)** | `silver_sonnet46_benchmark_v1` (Claude Sonnet 4.6) |
| **Reference (bullets)** | `silver_sonnet46_benchmark_bullets_v1` (Claude Sonnet 4.6) |
| **Schema** | `metrics_summarization_v2` |
| **Paragraph configs** | `data/eval/configs/summarization/autoresearch_prompt_*_benchmark_paragraph_v1.yaml` |
| **Bullet configs** | `data/eval/configs/summarization_bullets/autoresearch_prompt_*_benchmark_bullets_v1.yaml` |
| **Previous report** | [Smoke v1 (2026-04)](EVAL_SMOKE_V1_2026_04.md) |

For metric definitions and interpretation guidance, see the
[Evaluation Reports index](index.md).

---

## What changed vs. the April 2026 smoke report

### Scale

The smoke report used 5 episodes; this benchmark report uses 10 episodes from the same
5 feeds. Numbers are more stable — single-episode outliers have less influence. The
overall ranking order is consistent with smoke, but individual scores shift by 1-3%.

### New Ollama exclusion: llama3.3:70b

`llama3.3:70b-instruct-q3_K_M` was present in the smoke report (showing OOM). Its
benchmark configs have been deleted — it crashes before completing episode 1 on this
hardware (Apple M-series, model runner OOM error). The 4 configs are removed from the
eval matrix.

### Latency anomalies in large Ollama models

`qwen3.5:27b` (414 s/ep for paragraphs) and `qwen3.5:9b` (226 s/ep) are being
CPU-offloaded in the paragraph benchmark — likely a VRAM pressure issue during the
longer 10-episode run. Their bullet latencies are normal (63 s and 17 s respectively).
Do not use the paragraph latency figures for these two models for planning; re-run on
your hardware.

---

## Key takeaways

1. **Anthropic Haiku 4.5** leads cloud paragraphs (33.7% ROUGE-L, 86.2% embed) and
   cloud bullets (38.6% ROUGE-L). Rankings are stable vs. smoke.
2. **DeepSeek** is the strongest non-Anthropic provider: 29.5% ROUGE-L (paragraphs) and
   38.0% ROUGE-L with the best embedding similarity across all cloud (85.7%, bullets).
3. **qwen3.5:35b** is the best on-prem model for paragraphs (31.9% ROUGE-L, 20.8s/ep)
   — it ties with OpenAI/Gemini at benchmark scale, comfortably within the cloud range.
4. **qwen3.5:35b** also leads on-prem bullets (36.2% ROUGE-L, 14.1s/ep). Unlike smoke
   where llama3.2:3b led bullets, the 35b model takes the top spot at benchmark scale.
5. **qwen3.5:27b** shows strong semantic alignment (88.4% embedding, bullets) but is
   impractical for paragraph inference (414 s/ep CPU-offload on this hardware).
6. **llama3.2:3b** remains the best fast on-prem option: 24.4% ROUGE-L paragraphs
   (8.5s/ep), 33.6% bullets (5.2s/ep) — 3B params, 2 GB disk.
7. **phi3:mini** is unsuitable for paragraph summaries: 157.7% WER, 175.5% coverage —
   hallucinates / repeats extensively. Its bullet score is mediocre but usable (28.3%).
8. **qwen2.5:7b** breaks bullet JSON at benchmark scale (19.5% ROUGE-L, 65.0% embed) —
   do not use for structured output.

---

## Paragraph summaries — full metrics

### Cloud providers (sorted by ROUGE-L)

| Provider | Model | Lat/ep | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Embed | Coverage | WER |
| -------- | ----- | ------ | ------- | ------- | ------- | ---- | ----- | -------- | --- |
| **Anthropic** | claude-haiku-4-5 | 5.0s | **65.3%** | **30.5%** | **33.7%** | **24.3%** | **86.2%** | 99.8% | 89.9% |
| DeepSeek | deepseek-chat | 8.9s | 62.7% | 24.2% | 29.5% | 17.6% | 83.6% | 89.3% | 89.2% |
| Gemini | gemini-2.0-flash | **2.7s** | 57.3% | 23.4% | 28.7% | 16.3% | 82.5% | 78.8% | 87.2% |
| Mistral | mistral-small-latest | 4.6s | 61.3% | 23.7% | 28.0% | 18.9% | 82.3% | 108.4% | 96.5% |
| OpenAI | gpt-4o-mini | 8.5s | 57.7% | 22.4% | 26.8% | 16.9% | 84.1% | 96.0% | 92.6% |
| Grok | grok-3-mini | 7.5s | 57.5% | 20.9% | 26.7% | 14.9% | 81.7% | 90.5% | 91.0% |

### Local Ollama — paragraphs (sorted by ROUGE-L)

Hardware: Apple M-series. Re-run on your machine for latency decisions.
`*` = anomalous latency (CPU offload detected during this run; re-run to confirm).

| Model | Tag | Lat/ep | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Embed | Coverage | WER |
| ----- | --- | ------ | ------- | ------- | ------- | ---- | ----- | -------- | --- |
| **qwen3.5:35b** | qwen3.5:35b | 20.8s | **63.7%** | **24.4%** | **31.9%** | **18.6%** | **81.5%** | 108.0% | 96.7% |
| qwen3.5:27b | qwen3.5:27b | 414.3s\* | 61.5% | 24.1% | 27.9% | 17.8% | 82.4% | 120.3% | 104.4% |
| mistral-small3.2 | mistral-small3.2:latest | 89.2s | 58.7% | 21.7% | 28.1% | 16.4% | 81.4% | 84.4% | 90.5% |
| mistral:7b | mistral:7b | 18.3s | 56.8% | 24.4% | 27.2% | 18.3% | 76.7% | 89.5% | 93.0% |
| llama3.1:8b | llama3.1:8b | 17.9s | 59.9% | 25.7% | 26.8% | 20.9% | 79.0% | 103.9% | 100.1% |
| mistral-nemo:12b | mistral-nemo:12b | 25.1s | 56.9% | 23.9% | 26.8% | 16.8% | 79.5% | 105.1% | 98.0% |
| qwen3.5:9b | qwen3.5:9b | 226.1s\* | 59.6% | 21.4% | 25.7% | 14.4% | 78.0% | 99.6% | 95.6% |
| qwen2.5:32b | qwen2.5:32b | 78.3s | 55.5% | 19.9% | 24.6% | 12.4% | 80.7% | 77.4% | 90.0% |
| llama3.2:3b | llama3.2:3b | **8.5s** | 55.3% | 21.5% | 24.4% | 17.3% | 78.6% | 107.6% | 103.5% |
| qwen2.5:7b | qwen2.5:7b | 17.1s | 55.2% | 19.5% | 23.6% | 13.9% | 78.9% | 89.9% | 93.6% |
| gemma2:9b | gemma2:9b | 19.5s | 50.2% | 14.5% | 22.4% | 9.6% | 79.1% | 82.4% | 91.3% |
| phi3:mini | phi3:mini | 16.2s | 47.7% | 13.2% | 19.2% | 7.5% | 77.7% | **175.5%** | **157.7%** |
| llama3.3:70b-q3km | — | — | — | — | — | — | — | — | — |

> `llama3.3:70b-instruct-q3_K_M`: model runner crashed with OOM on this hardware.
> Configs deleted — see note above.

---

## Bullet JSON summaries — full metrics

Bullet ROUGE scores are computed against `silver_sonnet46_benchmark_bullets_v1`. Not
comparable to paragraph scores.

### Cloud providers — bullets (sorted by ROUGE-L)

| Provider | Model | Lat/ep | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Embed | Coverage | WER |
| -------- | ----- | ------ | ------- | ------- | ------- | ---- | ----- | -------- | --- |
| **Anthropic** | claude-haiku-4-5 | 3.5s | 64.7% | **33.3%** | **38.6%** | **34.3%** | 84.2% | 86.9% | 84.1% |
| DeepSeek | deepseek-chat | 5.6s | 63.5% | 35.4% | 38.0% | 33.3% | **85.7%** | 73.1% | 83.2% |
| Gemini | gemini-2.0-flash | **1.6s** | **61.8%** | 33.9% | 34.9% | 30.6% | 79.9% | 64.4% | 84.2% |
| Mistral | mistral-small-latest | 2.0s | 60.1% | 28.8% | 33.8% | 30.3% | 85.1% | 78.6% | 87.6% |
| OpenAI | gpt-4o-mini | 5.7s | 59.6% | 31.5% | 32.1% | 29.9% | 83.8% | 67.4% | 82.8% |
| Grok | grok-3-mini | 8.3s | 55.3% | 26.9% | 29.2% | 24.9% | 80.9% | 65.6% | 88.2% |

### Local Ollama — bullets (sorted by ROUGE-L)

| Model | Tag | Lat/ep | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Embed | Coverage | WER |
| ----- | --- | ------ | ------- | ------- | ------- | ---- | ----- | -------- | --- |
| **qwen3.5:35b** | qwen3.5:35b | **14.1s** | 63.6% | 33.4% | **36.2%** | 32.2% | 87.3% | 77.4% | 83.3% |
| qwen3.5:27b | qwen3.5:27b | 63.2s | **65.6%** | 32.4% | 35.2% | 33.7% | **88.4%** | 89.0% | 87.0% |
| mistral-small3.2 | mistral-small3.2:latest | 39.9s | 63.9% | **35.6%** | 34.2% | **33.6%** | 84.3% | 67.5% | 83.8% |
| llama3.2:3b | llama3.2:3b | **5.2s** | 57.6% | 31.5% | 33.6% | 25.5% | 82.9% | 54.5% | 81.6% |
| qwen3.5:9b | qwen3.5:9b | 16.7s | 60.0% | 28.8% | 32.6% | 27.1% | 83.5% | 69.7% | 85.2% |
| llama3.1:8b | llama3.1:8b | 10.1s | 52.9% | 27.4% | 31.5% | 19.4% | 78.5% | 50.4% | 86.2% |
| qwen2.5:32b | qwen2.5:32b | 50.9s | 56.7% | 26.9% | 30.5% | 25.6% | 84.9% | 62.1% | 84.7% |
| mistral-nemo:12b | mistral-nemo:12b | 16.8s | 48.0% | 23.8% | 28.9% | 16.9% | 73.6% | 50.3% | 85.5% |
| mistral:7b | mistral:7b | 13.0s | 52.8% | 27.8% | 28.6% | 20.6% | 76.1% | 49.9% | 87.1% |
| phi3:mini | phi3:mini | 8.2s | 50.7% | 23.0% | 28.3% | 21.0% | 74.3% | 66.9% | 89.8% |
| gemma2:9b | gemma2:9b | 11.1s | 45.7% | 20.9% | 27.3% | 16.4% | 78.3% | 47.0% | 88.0% |
| qwen2.5:7b | qwen2.5:7b | 9.2s | 31.1% | 13.0% | 19.5% | 6.4% | 65.0% | 28.7% | 91.2% |
| llama3.3:70b-q3km | — | — | — | — | OOM | — | — | — | — |

> `qwen2.5:7b` (19.5% ROUGE-L, 65.0% embed) does not reliably follow the JSON bullet
> format — inspect predictions before using. All other models produce valid JSON output.

---

## On-prem Ollama vs cloud API

> **Decision guide for teams choosing between local and cloud summarization.**

### Quality (ROUGE-L, paragraphs)

```text
Anthropic Haiku 4.5 (cloud):    33.7%  ██████████████████████████████████
DeepSeek (cloud):               29.5%  █████████████████████████████
─── on-prem competitive zone ───────────────────────────────────────────
qwen3.5:35b (local, 21s):       31.9%  ████████████████████████████████
─── cloud mid-tier ──────────────────────────────────────────────────────
Gemini (cloud):                 28.7%  ████████████████████████████
Mistral (cloud):                28.0%  ████████████████████████████
qwen3.5:27b (local, 414s):      27.9%  ███████████████████████████
OpenAI (cloud):                 26.8%  ██████████████████████████
Grok (cloud):                   26.7%  ██████████████████████████
mistral-small3.2 (local, 89s):  28.1%  ████████████████████████████
─── below cloud floor ───────────────────────────────────────────────────
llama3.2:3b (local, 8.5s):      24.4%  ████████████████████████
```

**qwen3.5:35b** (31.9%) is the only on-prem model above the cloud median. It
outperforms OpenAI gpt-4o-mini and Grok, and ties with Gemini 2.0 Flash at
benchmark scale.

### Latency

- **Cloud leaders:** Gemini 2.0 Flash (2.7s/ep paragraphs, 1.6s/ep bullets); Mistral
  API (4.6s/ep, 2.0s/ep bullets); Anthropic (5.0s/ep, 3.5s/ep bullets).
- **On-prem fast:** llama3.2:3b (8.5s/ep paragraphs, 5.2s/ep bullets) is the fastest
  on-prem option. qwen3.5:35b (21s/ep paragraphs, 14s/ep bullets) is the best
  quality/latency balance.

### Use-case decision table

| Scenario | Recommendation |
| -------- | -------------- |
| Production, quality-sensitive | Anthropic Haiku 4.5 (cloud) — leads all providers |
| Production, cost-sensitive | DeepSeek — 2nd best quality at ~$0.02/100 eps |
| Production, speed-sensitive | Gemini 2.0 Flash — fastest cloud (1.6–2.7s/ep) |
| On-prem required, quality first | qwen3.5:35b (21s/ep, competitive with cloud mid-tier) |
| On-prem required, speed/quality | llama3.2:3b (8.5s/ep, 2 GB) — best fast on-prem option |
| Bullet JSON, cloud | Anthropic (highest ROUGE-L) or DeepSeek (best semantic embed) |
| Bullet JSON, on-prem, quality | qwen3.5:35b (36.2% ROUGE-L, 14s/ep) |
| Bullet JSON, on-prem, fast | llama3.2:3b (33.6% ROUGE-L, 5.2s/ep) |

---

## Smoke vs. benchmark — score delta

Scores at benchmark scale (10 eps) compared to smoke (5 eps):

| Provider | Smoke ROUGE-L | Benchmark ROUGE-L | Delta |
| -------- | ------------- | ----------------- | ----- |
| Anthropic | 30.5% | 33.7% | +3.2% |
| DeepSeek | 29.4% | 29.5% | +0.1% |
| Gemini | 27.9% | 28.7% | +0.8% |
| OpenAI | 26.6% | 26.8% | +0.2% |
| Mistral | 26.6% | 28.0% | +1.4% |
| Grok | 22.9% | 26.7% | +3.8% |
| qwen3.5:35b | 29.9% | 31.9% | +2.0% |
| llama3.2:3b | 22.6% | 24.4% | +1.8% |

> Positive deltas are expected: smoke picks 5 episodes that may skew harder or easier.
> Large positive swings (Grok +3.8%, Anthropic +3.2%) suggest those providers had
> an unlucky episode in smoke. Rankings are stable.

---

## Model IDs (acceptance-tested)

| Provider | Model ID used | Notes |
| -------- | ------------- | ----- |
| Anthropic | `claude-haiku-4-5` | Fastest Anthropic option; Sonnet 4.6 used for silver only |
| OpenAI | `gpt-4o-mini` | `max_completion_tokens` required for gpt-5 series |
| DeepSeek | `deepseek-chat` | — |
| Gemini | `gemini-2.0-flash` | `gemini-2.5-pro` / `gemini-3.1-pro-preview` are thinking models |
| Grok | `grok-3-mini` | — |
| Mistral | `mistral-small-latest` | — |

---

## How to reproduce

```bash
# Cloud paragraph (any provider)
make experiment-run \
  CONFIG=data/eval/configs/summarization/autoresearch_prompt_anthropic_benchmark_paragraph_v1.yaml \
  REFERENCE=silver_sonnet46_benchmark_v1

# Cloud bullets (any provider)
make experiment-run \
  CONFIG=data/eval/configs/summarization_bullets/autoresearch_prompt_anthropic_benchmark_bullets_v1.yaml \
  REFERENCE=silver_sonnet46_benchmark_bullets_v1

# Ollama paragraph
make experiment-run \
  CONFIG=data/eval/configs/summarization/autoresearch_prompt_ollama_qwen35_35b_benchmark_paragraph_v1.yaml \
  REFERENCE=silver_sonnet46_benchmark_v1

# Ollama bullets
make experiment-run \
  CONFIG=data/eval/configs/summarization_bullets/autoresearch_prompt_ollama_qwen35_35b_benchmark_bullets_v1.yaml \
  REFERENCE=silver_sonnet46_benchmark_bullets_v1
```

---

## Related

- [Smoke v1 (2026-04)](EVAL_SMOKE_V1_2026_04.md) — smoke report with same silver references
- [Smoke v1 (2026-03)](EVAL_SMOKE_V1_2026_03.md) — March report with GPT-4o silver
- [Evaluation Reports index](index.md) — methodology and metric definitions
- [AI Provider Comparison Guide](../AI_PROVIDER_COMPARISON_GUIDE.md) — decision guide
- [Ollama Provider Guide](../OLLAMA_PROVIDER_GUIDE.md) — Ollama setup and model reference
