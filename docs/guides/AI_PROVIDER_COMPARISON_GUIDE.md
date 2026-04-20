# AI Provider Comparison Guide

> **Authoritative v2 reference**: [`eval-reports/EVAL_HELDOUT_V2_2026_04.md`](eval-reports/EVAL_HELDOUT_V2_2026_04.md) — 6 cloud APIs + 11 Ollama local models, 100+ held-out cells under the v2 framework ([RFC-073](../rfc/RFC-073-autoresearch-v2-framework.md)), compound-scored on quality × latency × cost.
> v1 benchmark numbers later in this guide are **superseded** by the v2 report above.

---

## ⭐ Use these two. Everything else is a tradeoff

After evaluating 20 model variants across 100+ held-out cells, two picks cover ~95% of real podcast_scraper deployments. Pick one based on whether you need cloud or local.

### 🌐 Cloud / dev / prod / corpus building → `gemini-2.5-flash-lite` **non-bundled**

```yaml
backend:
  type: "gemini"
  model: "gemini-2.5-flash-lite"
prompts:
  system: "shared/summarization/system_bullets_v1"   # for bullets
  user: "shared/summarization/bullets_json_v1"
  # (or gemini/summarization/{system_v1,long_v1} for paragraph)
params:
  temperature: 0.0
```

**Use this for dev, test, corpus building, and most production traffic.**

- Quality: 0.564 bullets / 0.479 paragraph held-out (within 4% of the absolute best)
- Latency: **1.5s per episode** — 2-7× faster than any alternative
- Cost: **~$0.00047 per episode** — ~$0.47 per 1,000 episodes
- Why non-bundled: bundled mode costs 5-12% quality on Gemini and OpenAI for no real gain. Just make two API calls (one for bullets, one for paragraph) — still only 3s and $0.00094 total per episode.

**Upgrade to `deepseek-chat` non-bundled only if** quality matters more than throughput (0.586/0.541 vs 0.564/0.479, but ~7× slower at 10s/ep). **Switch to `claude-haiku-4-5` bundled only if** you specifically need title+summary+bullets in a single API call (bundled is viable on Anthropic; it's the only cloud provider where that's true).

### 🏠 Local / offline / privacy → `qwen3.5:9b` **bundled**

```yaml
backend:
  type: "ollama"
  model: "qwen3.5:9b"
llm_pipeline_mode: "bundled"
params:
  temperature: 0.0
```

**Use this for any fully-local, offline, or privacy-constrained deployment.**

- Quality: 0.529 bullets / 0.509 paragraph held-out (free, matches mid-tier cloud)
- Latency: ~44s per episode for BOTH outputs (vs ~72s for two separate non-bundled calls)
- Cost: $0 (your hardware)
- Why bundled: local paragraph contests on 5 of 11 Ollama models non-bundled; bundled mode's JSON schema stabilises structure and makes paragraph output reliable. Single call returns title + summary + bullets.

**Upgrade to `qwen3.5:9b` non-bundled for bullets only** if you need absolute max bullets quality locally (0.580 vs 0.529, +10%). **Fallback to `llama3.2:3b` non-bundled bullets** if latency matters more than quality (12s vs 33s, quality drops to 0.501).

---

## Full reference — v2 matrix details

The two picks above cover most deployments. Everything below is the detailed reference for
edge cases and the data that backs those recommendations.

All 6 cloud LLM providers evaluated under the v2 framework: dev/held-out split, fraction-based
judge contestation, champion prompts ported across providers. Held-out ROUGE-L vs Sonnet 4.6
silver on `curated_5feeds_benchmark_v2` (5 unseen episodes, ~32 min each):

| Track | OpenAI | Anthropic | Gemini | Mistral | **DeepSeek** | Grok |
| ----- | :----: | :-------: | :----: | :-----: | :----------: | :--: |
| Bullets non-bundled | 39.6% | 40.7% | 40.1% | 37.3% | **43.1%** | 38.6% |
| Bullets bundled | 33.2% | **39.3%** | 28.5% | 30.4% | 34.4% | 30.5% |
| Paragraph non-bundled | 31.7% | 36.4% | 29.3% | 27.8% | **40.7%** | 31.3% |
| Paragraph bundled | 29.5% | **39.2%** | 26.6% | 30.3% | 35.9% | 28.9% |

**Cell winners:**

- **Non-bundled (either track):** DeepSeek (`deepseek-chat`). Cheapest cloud non-local
  option that also leads on both quality metrics — the clear sweet spot.
- **Bundled (either track):** Anthropic (`claude-haiku-4-5`). Only provider where bundled
  is competitive with non-bundled; bundled paragraph even beats non-bundled paragraph.

### Local (Ollama) — v2 held-out, 11 models evaluated → Core 5 standardized

> **ADR-077 (2026-04-18):** Standardized on 5 models for regular sweeps and
> pipeline validation. One per family + one large-scale reference:
> **qwen3.5:9b** (champion), **llama3.2:3b** (speed), **mistral:7b** (mid-tier),
> **gemma2:9b** (diversity), **qwen3.5:35b** (scale reference).
> Dropped 6 models that were same-family duplicates or structurally unsuitable.
> See [ADR-077](../adr/ADR-077-local-ollama-model-selection.md) for rationale.

**Top 6 local bullets non-bundled (held-out):**

| Rank | Model | Size | Bullets final |
| :--: | ----- | :--: | :-----------: |
| **1** | **qwen3.5:9b** | 9B | **0.580** |
| 2 | qwen3.5:35b | 35B | 0.576 |
| 3 | qwen3.5:27b | 27B | 0.543 |
| 4 | mistral-small3.2 | ~22B | 0.536 |
| 5 | mistral:7b | 7B | 0.526 |
| 6 | llama3.1:8b | 8B | 0.518 |

**Cross-matrix ranking (bullets non-bundled held-out):**

| Rank | Provider+model | Final |
| :--: | -------------- | :---: |
| 1 | DeepSeek (cloud) | 0.586 |
| **2** | **Ollama qwen3.5:9b (local, free)** | **0.580** |
| 3 | Ollama qwen3.5:35b | 0.576 |
| 4 | Anthropic haiku-4.5 | 0.570 |
| 5 | OpenAI gpt-4o | 0.566 |

**Paragraph — use bundled on local** (big finding):

| Model | Non-bundled paragraph | Bundled paragraph |
| ----- | :-------------------: | :---------------: |
| qwen3.5:9b | 0.505 (2/5 contested) | **0.509** (uncontested) |
| qwen3.5:35b | 0.325 (contested → ROUGE-only) | **0.492** |
| mistral-small3.2 | 0.288 (contested → ROUGE-only) | **0.468** |

Non-bundled paragraph contests on 3 of 4 local models (long transcripts → inconsistent
structure → judge disagreement). **Bundled paragraph doesn't contest at all** — the JSON
schema stabilises output. Bundled is the correct local-deployment choice for paragraph.

**Local picks**:

- **Bullets (either mode)** → `qwen3.5:9b` non-bundled (0.580).
- **Paragraph** → `qwen3.5:9b` **bundled** (0.509). Same single call produces both.
- **Single call, title+summary+bullets** → `qwen3.5:9b` bundled. Loses ~9% on bullets vs
  non-bundled but gains reliability and cost efficiency.
- **No-Ollama-daemon alternative** → `DISLab/SummLlama3.2-3B` via HF transformers directly.
  v2 held-out **paragraph 0.485** (uncontested 0/5, dev 0.442), **bullets 0.416** (uncontested
  0/5, dev 0.467). Runs via `run_summllama_v2.py` (HF `transformers` + MPS / CUDA). DPO-tuned
  on faithfulness/completeness/conciseness — the same Llama-3.2-3B base that scores only
  0.270 paragraph standalone lifts to 0.485 with alignment. **Paragraph-strong, bullets-
  weaker** (DPO was prose-shaped, not list-shaped). Latency 60-156s/ep on Apple MPS, slower
  than Ollama but operationally simpler (no daemon, one Python process). **Pick this for
  paragraph-first deployments or when Ollama can't be run.** For bullet-heavy workloads,
  qwen3.5:9b bundled stays the better local pick. See
  [Held-out v2 report §6a](eval-reports/EVAL_HELDOUT_V2_2026_04.md#6a-ml-transformers-standalone-hf-not-ollama--2026-04-16).

**Default picks by use case** (compound-scored across quality, latency, cost — see
[Held-out v2 report §Compound analysis](eval-reports/EVAL_HELDOUT_V2_2026_04.md#compound-analysis--pareto-frontier)):

| Priority | Best pick | Why |
| :------- | :-------- | :-- |
| **Balanced default** | **Gemini 2.5-flash-lite non-bundled** | 0.564 / 0.479, 1.5s, ~$0.00047/ep. New 2026-04-16 — strict upgrade over 2.0-flash. |
| **Quality first** | DeepSeek non-bundled | 0.586 (#1), 10s, $0.0005/ep. Top quality at near-bottom cost. |
| **Single-call bundled** | Anthropic haiku-4.5 bundled | 0.552 bullets / 0.548 paragraph, 7s, $0.006/ep. Only provider where bundled is competitive. |
| **Throughput / real-time** | Gemini 2.5-flash-lite | 1.5s/ep — fastest in the matrix. 2× faster than Gemini 2.0-flash at same quality tier. |
| **OpenAI-ecosystem (cost-sensitive)** | gpt-4o-mini | 0.540 / 0.469, 6.6s, ~$0.00074/ep. 16× cheaper than gpt-4o for 4% quality hit. |
| **Privacy / offline** | Ollama qwen3.5:9b | 0.580 (local #1), 33s/ep, free. Matches DeepSeek quality offline. |

**Avoid / dominated (unless ecosystem-locked):**

- **OpenAI gpt-4o**: Anthropic haiku-4.5 is better on quality AND ~3× cheaper.
- **Grok**: slower than Anthropic/Gemini without compensating quality.
- **OpenAI bundled, Gemini bundled, Mistral non-bundled paragraph, local paragraph on long transcripts**: structural weak spots visible in the matrix.
- **Ollama qwen3.5:27b / qwen3.5:35b**: larger but not better than qwen3.5:9b.

See [v2 eval report](eval-reports/EVAL_HELDOUT_V2_2026_04.md) for blended scores, dev numbers, generalisation analysis, and provider-specific quirks.

### Full pipeline validation (2026-04-18, PR #603)

> All providers tested through the complete pipeline (summary → GI → KG →
> bridge) on 5 held-out episodes. `make pipeline-validate` verifies each
> stage produces valid output.

| Provider | Summary | GI | Grounding | KG | Bridge |
| -------- | :-----: | :-: | :-------: | :-: | :----: |
| openai/gpt-4o-mini | ✅ | ✅ | 100% | ✅ | ✅ |
| gemini/flash-lite | ✅ | ✅ | 98% | ✅ | ✅ |
| anthropic/haiku-4.5 | ✅ | ✅ | 100% | ✅ | ✅ |
| deepseek/deepseek-chat | ✅ | ✅ | 100% | ✅ | ✅ |
| mistral/mistral-small | ✅ | ✅ | 100% | ✅ | ✅ |
| grok/grok-3-mini | ✅ | ✅ | 100% | ✅ | ✅ |
| ollama/qwen3.5:9b | ✅ | ✅ | 100% | ✅ | ✅ |
| ollama/llama3.1:8b | ✅ | ✅ | ✅ | ✅ | ✅ |
| ollama/mistral:7b | ✅ | ✅ | 98% | ✅ | ✅ |
| ollama/gemma2:9b | ✅ | ⚠️ | 95% | ✅ | ✅ |
| ollama/qwen3.5:35b | ✅ | ✅ | 98% | ✅ | ✅ |

All 11 providers pass the full pipeline. gemma2:9b is borderline on GI
insight count (7.8/ep vs 8 threshold, instruction-following gap). Minimum
viable model size for full pipeline: **7-8B** (3B models fail on KG entity
extraction). See [ADR-077](../adr/ADR-077-local-ollama-model-selection.md).

---

## Implementation Status

All providers below are **implemented and acceptance-tested** (v2.4.0+).

| Provider | Status | RFC | Notes |
| ---------- | :------: | :---: | ------- |
| **Local ML** | Implemented | - | Default provider (Whisper + spaCy + Transformers): transcription, speaker detection, summarization |
| **Hybrid ML** | Implemented | RFC-042 | Summarization only: MAP (LongT5) + REDUCE (transformers / Ollama / llama_cpp) |
| **OpenAI** | Implemented | RFC-013 | Transcription + summarization (Whisper API + GPT API) |
| **Gemini** | Implemented | RFC-035 | Transcription + summarization (no speaker detection) |
| **Mistral** | Implemented | RFC-033 | Summarization only (EU data residency) |
| **Anthropic** | Implemented | RFC-032 | Summarization only (no transcription or speaker detection) |
| **DeepSeek** | Implemented | RFC-034 | Summarization only; ultra low-cost |
| **Grok** | Implemented | RFC-036 | Summarization only; real-time information access |
| **Ollama** | Implemented | RFC-037 | Transcription, speaker detection, summarization; local self-hosted LLMs, zero cost, complete privacy |

For hybrid_ml (MAP-REDUCE) configuration and REDUCE backends (Ollama, llama_cpp,
transformers), see [ML Provider Reference](ML_PROVIDER_REFERENCE.md) and
[Configuration API](../api/CONFIGURATION.md). **Transcript cleaning:** `hybrid_ml` honors
`transcript_cleaning_strategy` like API providers; with **`pattern`**, use
`hybrid_internal_preprocessing_after_pattern` (CLI: `--hybrid-internal-preprocessing-after-pattern`)
to control internal MAP preprocessing after workflow cleaning ([RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md#layered-transcript-cleaning-issue-419)).
**LLM / hybrid LLM stage:** long transcripts can hit the **length-ratio guard** (pattern-cleaned
input ≥2000 chars and LLM output **below 20%** of that length is discarded); see
[CONFIGURATION — LLM cleaning length guard](../api/CONFIGURATION.md#llm-cleaning-length-guard-issue-564).

---

## Key Statistics at a Glance

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROVIDER LANDSCAPE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  9 Summarization Options  │  (Hybrid = MAP+REDUCE)  │  3 Full-Stack Ready  │
│  ════════════════════     │  ═══════════════════════ │  ═══════════════     │
│  Local ML              │  Hybrid ML (RFC-042)  │  Local ML          │
│  Hybrid ML             │  MAP + Ollama/llama_cpp │  OpenAI (tx+sum)  │
│  OpenAI                │  or transformers REDUCE  │  Ollama            │
│  Gemini                │                         │                      │
│  Mistral               │                         │                      │
│  Anthropic / DeepSeek  │                         │                      │
│  Grok / Ollama         │                         │                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                           COST SPECTRUM (per 100 episodes)                  │
│                                                                             │
│  $0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ $37  │
│  │                                                                     │   │
│  ▼                                                                     ▼   │
│  Local/Ollama                                               OpenAI (full) │
│  ($0)                                                             ($37)    │
│                                                                             │
│  DeepSeek ─── Grok ─── Anthropic ─── Gemini ─── OpenAI (text) ─── OpenAI  │
│   ($0.02)    ($0.03)    ($0.40)      ($0.95)    ($0.55)           ($37)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Decision Matrix

> **Updated picks (v2 data, 2026-04-16)**: Default local-LLM is now **qwen3.5:9b** (not llama3.2:3b).
> Hybrid ML is **demoted** from default recommendation to narrow niche — v2 showed standalone
> qwen3.5:9b (0.509 paragraph held-out) beats hybrid bart+qwen3.5:9b (0.448) for our workload.

| If you need... | Choose | Why |
| :------------- | :----: | :-- |
| **Complete Privacy** | Local ML / Ollama | Data never leaves your device |
| **Lowest Cost** | Local ML / Ollama | $0 (just electricity) |
| **Air-gapped (no Ollama)** | Local ML (`ml_bart_led_autoresearch_v1`) | v2 held-out 0.206 (weak but zero-deps floor) |
| **Air-gapped + Ollama** | **Ollama (`qwen3.5:9b` bundled)** | v2 held-out 0.509 paragraph / 0.529 bullets — the recommended local pick |
| **Highest Quality (cloud)** | DeepSeek (non-bundled) | v2 held-out 0.586 bullets / 0.541 paragraph — top of matrix |
| **Fastest Cloud** | Gemini 2.5-flash-lite | 1.5s/ep — fastest in v2 matrix |
| **On-prem, quality first** | **Ollama `qwen3.5:9b` bundled** | Best local quality (0.509 paragraph, uncontested) |
| **On-prem, speed first** | Ollama (`llama3.2:3b`) | 12s/ep, quality floor 0.501 bullets |
| **On-prem, bullets-only max quality** | Ollama (`qwen3.5:9b` non-bundled) | 0.580 held-out (2nd overall, beats most cloud) |
| **On-prem, no Ollama daemon, paragraph-first** | HF transformers (`DISLab/SummLlama3.2-3B`) | 0.485 held-out paragraph / 0.416 bullets — DPO-tuned 3B via HF transformers on MPS/CUDA directly; operationally simpler than Ollama. Paragraph-strong, bullets-weaker (DPO was prose-shaped). |
| **Full Capabilities** | OpenAI / Local ML | All 3 capabilities (transcription + speaker + summary) |
| **Hybrid MAP-REDUCE** | Hybrid ML (Ollama/llama_cpp) | Retained as niche option (RFC-042); not recommended as default — see v2 findings |
| **Real-Time Info** | Grok | Real-time information access (RFC-036) |
| **Lowest Cloud Cost** | Gemini 2.5-flash-lite / DeepSeek | ~$0.0005/ep both — comparable |
| **EU Data Residency** | Mistral | European servers (RFC-033) |
| **Huge Context** | Gemini | 2 million token window (RFC-035) |
| **Free Development** | Gemini / Grok | Generous free tiers (RFC-035, RFC-036) |
| **Self-Hosted** | Ollama | Offline/air-gapped (RFC-037) |

### Screenplay formatting vs transcription (GitHub #562)

| `transcription_provider` | `screenplay: true` in transcript body |
| :----------------------- | :------------------------------------ |
| **`whisper`** (local) | Yes — gap-based speaker labels via `MLProvider` |
| **`openai`** (`whisper-1` API) | No — plain text transcript; segment JSON may still exist |
| **`gemini`** / **`mistral`** | No — plain text in **our** integration (provider wiring here) |

Validation coerces truthy **`screenplay`** to **`false`** when the transcription provider is not **`whisper`**, with a **single** INFO while a process gate is set (the gate resets when each **`run_pipeline`** finishes so another **`Config`** build can log again). Details: [CONFIGURATION.md — Screenplay vs transcription](../api/CONFIGURATION.md).

---

## Empirical Highlights

All claims below are backed by measured data. For the full metrics tables, methodology,
and metric definitions, see the [Evaluation Reports](eval-reports/index.md).

> **Note on the silver reference:** Results were re-measured in April 2026 against
> `silver_sonnet46_benchmark_v1` (Claude Sonnet 4.6, 10-episode benchmark scale).
> Rankings shifted significantly from March 2026 — see
> [why the rankings changed](eval-reports/EVAL_SMOKE_V1_2026_04.md#why-the-rankings-changed-vs-march-2026).
> The March 2026 numbers (vs GPT-4o silver) are preserved in the
> [March report](eval-reports/EVAL_SMOKE_V1_2026_03.md) for reference.

### Full quality ladder — all four tiers

Every summarization option in one view, ordered by ROUGE-L. ML/hybrid numbers are
smoke-scale (5 eps); cloud and Ollama numbers are benchmark-scale (10 eps).

All numbers benchmark-scale (10 eps, `curated_5feeds_benchmark_v1` vs
`silver_sonnet46_benchmark_v1`).

| Tier | Mode | ROUGE-L | Embed | Lat/ep | Dependencies |
| :--- | :--- | ------: | ----: | -----: | :----------- |
| 1 — ML Dev | `ml_small_authority` | 19.1% | 70.0% | fast | None (CI safe) |
| 2 — ML Prod | `ml_bart_led_autoresearch_v1` | 20.5% | 68.2% | 26s | None (air-gap safe) |
| — Hybrid | `ml_hybrid_bart_llama32_3b_autoresearch_v1` | 21.1% | 76.6% | 15s | Ollama (3B only) |
| 3 — LLM Local (small) | `llama3.2:3b` direct | 24.4% | 78.6% | 8.5s | Ollama |
| 3 — LLM Local (large) | `qwen3.5:35b` direct | 31.9% | 81.5% | 21s | Ollama |
| 4 — LLM Cloud (mid) | Gemini 2.0 Flash | 28.7% | 82.5% | 2.7s | API key |
| 4 — LLM Cloud (mid) | DeepSeek | 29.5% | 83.6% | 8.9s | API key |
| 4 — LLM Cloud (best) | Anthropic Haiku 4.5 | **33.7%** | **86.2%** | 5.0s | API key |

**Key observations:**

- The jump from ML-prod (20.5%) to direct-LLM (24.4%) is ~4 ROUGE-L points. The hybrid
  (21.1%) closes only ~1.5 of those points at benchmark scale — less than smoke
  suggested (23.7%). The gap exists because temperature=0.5 sampling variance averages
  down over 10 episodes. The hybrid is still valuable for long transcripts (BART MAP
  chunks arbitrary-length input).
- qwen3.5:35b (31.9%) is the only on-prem model in the cloud quality range — it
  exceeds Gemini, OpenAI, and Grok.
- The hybrid is the right choice when: transcripts exceed LLM context windows, Ollama
  is available but only a small model fits in VRAM, or quality must improve over
  ML-prod without paying for cloud.

### Cloud providers — paragraphs (vs Sonnet 4.6 silver, April 2026)

**Best cloud provider:** **Anthropic** (`claude-haiku-4-5`) — leads on ROUGE-L and
embedding similarity across both smoke (5 eps) and benchmark (10 eps) runs. **Gemini**
(`gemini-2.0-flash`) remains the fastest cloud option. Numbers below are benchmark
(10-episode, more stable).

| Provider | ROUGE-L | Embed | Latency |
| -------- | ------- | ----- | ------- |
| **Anthropic** | **33.7%** | **86.2%** | 5.0s |
| DeepSeek | 29.5% | 83.6% | 8.9s |
| Gemini | 28.7% | 82.5% | **2.7s** |
| Mistral | 28.0% | 82.3% | 4.6s |
| OpenAI | 26.8% | 84.1% | 8.5s |
| Grok | 26.7% | 81.7% | 7.5s |

> The Anthropic model used here is `claude-haiku-4-5` (smallest/fastest Haiku). The
> silver reference is Claude Sonnet 4.6 — Anthropic scores well partly because the
> models share a generation family.

Full table:
[Benchmark v1 report — Cloud providers](eval-reports/EVAL_BENCHMARK_V1_2026_04.md#cloud-providers-sorted-by-rouge-l)

### Local Ollama — paragraphs (vs Sonnet 4.6 silver, April 2026)

**Best local model:** **Qwen 3.5:35b** at 31.9% ROUGE-L (21s/ep) — the only on-prem
model above the cloud median, competitive with Gemini and Mistral API. **Mistral
Small 3.2** (28.1%, 89s/ep) is a strong mid-tier option. **llama3.2:3b** (24.4%, 8.5s)
is the best fast/low-resource choice. Numbers below are benchmark scale (10 eps).

| Model | ROUGE-L | Embed | Latency |
| ----- | ------- | ----- | ------- |
| qwen3.5:35b | **31.9%** | **81.5%** | 21s |
| mistral-small3.2 | 28.1% | 81.4% | 89s |
| qwen2.5:32b | 24.6% | 80.7% | 78s |
| qwen3.5:9b | 25.7% | 78.0% | 226s† |
| llama3.2:3b | 24.4% | 78.6% | **8.5s** |

> Latencies are hardware-dependent (Apple M-series). Re-run on your machine.
> †qwen3.5:9b and qwen3.5:27b showed CPU-offload latency anomalies in the benchmark run.

### Local Ollama — bullets (vs Sonnet 4.6 bullets silver, April 2026)

For bullet JSON output, **qwen3.5:35b** leads at benchmark scale (36.2% ROUGE-L,
14s/ep). **llama3.2:3b** is the fastest option (33.6% ROUGE-L, 5.2s/ep). **qwen2.5:7b**
does not reliably follow the JSON format — avoid it for the bullets track.

| Model | ROUGE-L | Embed | Latency |
| ----- | ------- | ----- | ------- |
| qwen3.5:35b | **36.2%** | 87.3% | **14.1s** |
| qwen3.5:27b | 35.2% | **88.4%** | 63.2s |
| mistral-small3.2 | 34.2% | 84.3% | 39.9s |
| llama3.2:3b | 33.6% | 82.9% | **5.2s** |
| qwen3.5:9b | 32.6% | 83.5% | 16.7s |

Full tables:
[Benchmark v1 report (April 2026)](eval-reports/EVAL_BENCHMARK_V1_2026_04.md)

---

## Detailed Cost Analysis

### Per 100 Episodes — Complete Breakdown

| Provider | Transcription | Speaker | Summary | **Total** | vs OpenAI |
| :------- | :-----------: | :-----: | :-----: | :-------: | :-------: |
| **Local ML** | $0 | $0 | $0 | **$0** | -100% |
| **Ollama** | N/A | $0 | $0 | **$0** | -100% |
| **DeepSeek** | N/A | N/A | $0.016 | **$0.016** | -97% |
| **Grok (beta)** | N/A | N/A | $0.00 | **$0.00** | -100% |
| **Mistral (Small)** | N/A | N/A | $0.11 | **$0.11** | -80% |
| **Anthropic (Haiku)** | N/A | N/A | $0.40 | **$0.40** | -27% |
| **Gemini (Flash)** | $0.90 | N/A | $0.05 | **$0.95** | +73% |
| **OpenAI (Nano)** | $36.00 | N/A | $0.28 | **$36.28** | baseline |
| **OpenAI (Mini)** | $36.00 | N/A | $1.40 | **$37.40** | +3% |
| **Mistral (Large)** | N/A | N/A | $9.00 | **$9.00** | -75% |

### Monthly Cost Projections

```text
Monthly costs at different scales
═══════════════════════════════════════════════════════════════════════════

                    100 ep/month        1,000 ep/month      10,000 ep/month
                    ────────────        ──────────────      ───────────────
Local ML            $0                  $0                  $0
DeepSeek            $0.02               $0.16               $1.60
Grok                $0.03               $0.26               $2.60
Anthropic           $0.40               $4.00               $40.00
OpenAI (text only)  $0.55               $5.50               $55.00
OpenAI (full)       $37.40              $374.00             $3,740.00

  At 10,000 episodes/month, OpenAI full stack costs $3,740!
    Using local transcription + DeepSeek: $1.60 (99.96% savings)
```

> **Key insight:** Transcription dominates cloud costs (90%+). Use local Whisper +
> cloud text processing to save massively.

---

## Decision Flowchart

```text
                            START
                              │
                              ▼
                    ┌─────────────────┐
                    │  What's your    │
                    │  TOP priority?  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ PRIVACY │         │  COST   │         │ QUALITY │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                   │
        ▼                   ▼                   ▼
   Need transcription?  Need transcription?  Budget matters?
        │                   │                   │
   ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
   │Yes  │No │         │Yes  │No │         │Yes  │No │
   ▼     ▼   ▼         ▼     ▼   ▼         ▼     ▼   ▼
┌──────┐ ┌──────┐  ┌──────┐ ┌──────┐  ┌──────┐ ┌──────┐
│LOCAL │ │OLLAMA│  │LOCAL │ │DEEP  │  │GPT-5 │ │GPT-5 │
│  ML  │ │      │  │Whisper│ │SEEK  │  │ Mini │ │      │
│      │ │      │  │  +    │ │      │  │      │ │      │
│      │ │      │  │DeepSk │ │      │  │      │ │      │
└──────┘ └──────┘  └──────┘ └──────┘  └──────┘ └──────┘

        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │  SPEED  │         │ CONTEXT │         │   EU    │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │  GROK   │         │ GEMINI  │         │ MISTRAL │
   │         │         │   Pro   │         │         │
   │ Real-Time│        │   2M    │         │  Full   │
   │ faster  │         │ tokens  │         │  Stack  │
   └─────────┘         └─────────┘         └─────────┘
```

---

## Autoresearch-derived defaults (2026-04)

These are the research-backed settings used across `config/profiles/` (main
profiles + `capture_e2e_*.yaml` per-provider captures). Source data in
`docs/guides/eval-reports/` and `data/eval/runs/ner_*_smoke_v1/`.

Since 2026-04, the defaults are split across **named presets** so a single
edit updates every deployment profile that references them (GitHub #634):

- **Audio preprocessing**: [`config/profiles/audio/speech_optimal_v1.yaml`](../../config/profiles/audio/speech_optimal_v1.yaml)
  — referenced by all 5 deployment profiles via `audio_preprocessing_profile: speech_optimal_v1`.
- **Text cleaning (ML-only)**: `ml_preprocessing_profile` on `Config` (e.g.
  `cleaning_v4`), overrides the `mode_cfg.preprocessing_profile` default in
  the ML summary registry. Cloud LLM and Ollama providers send raw transcripts
  and ignore this field.

### NER / speaker detection

Smoke eval (5 episodes, 15 gold entities, `data/eval/runs/ner_*_smoke_v1/`):

| Backend | F1 | Notes |
| --- | :-: | --- |
| **spaCy trf** (`en_core_web_trf`) | **1.000** | Free, local, deterministic |
| spaCy sm (`en_core_web_sm`) | 0.966 | Faster, slightly lower recall |
| OpenAI, Anthropic, Gemini, Mistral, DeepSeek, Grok | 1.000 | All six cloud LLMs tie spaCy trf |
| Ollama `qwen3.5:9b` | 0.750 | Misses 6/15 entities; weakest tested |

**Default for main profiles**: `speaker_detector_provider: spacy` +
`ner_model: en_core_web_trf`. Ties every cloud LLM on the available data while
saving the API call. Post-reingestion validation on a larger/harder corpus is
tracked in `POST_REINGESTION_PLAN.md` Step 6.

**Exception — per-provider capture profiles** (`capture_e2e_<name>.yaml`):
these intentionally route NER through `<provider>` to exercise each provider's
full surface in profile/cost capture runs.

**All 6 cloud providers + Ollama have `detect_speakers` wired** (see
`src/podcast_scraper/providers/*/...`). Earlier "summarization only" notes in
this guide are obsolete.

### Grounded Insights (GI)

- `gi_insight_source: provider` (not `summary_bullets`) → **+10pp** insight
  coverage vs silver. Used by `cloud_balanced`, `cloud_quality`, `local`.
  `airgapped` keeps `summary_bullets` because SummLlama3.2-3B is summary-only,
  not chat.
- `gi_max_insights: 12` — autoresearch sweet spot; default is 20, which is
  long-tail and hurts precision.
- `gi_require_grounding: true` (default) — drops insights without grounded
  quote evidence.

### Knowledge Graph (KG)

- `kg_extraction_source: provider` → **+37pp** topic coverage vs silver
  (measured on KG pipeline, not summary-bullet proxy).
- `kg_max_topics: 10` — autoresearch sweet spot; default is 20.
- `kg_max_entities: 15` — matches default; standardised across profiles.
- **KG v3 prompt** (noun-phrase enforcement, promoted in #625 / PR #628) —
  already the default.

### Insight clustering

- Default threshold **0.75** (`src/podcast_scraper/search/insight_clusters.py`).
  Configured at call site, not via profile YAML.

### Pipeline mode

- `llm_pipeline_mode: bundled` — **only** for local `qwen3.5:9b`: bundled
  schema stabilises paragraph output for Ollama (+structure reliability).
- `llm_pipeline_mode: staged` (default) — for Gemini, OpenAI, Anthropic (except
  Haiku bundled edge case), DeepSeek, Mistral, Grok. On Gemini and OpenAI,
  bundled costs 5-12% quality for no real gain.

### Transcription

- `whisper_model: small.en` — 9.5% WER sweet spot, 148s/ep on CPU. Use this
  whenever the provider doesn't offer its own transcription.
- `whisper_model: base.en` — 40s/ep, ~13% WER. Faster but noticeably worse
  quality; reserved for `dev` profile.
- Providers with native transcription: **OpenAI** (`whisper-1`), **Gemini**
  (`gemini-2.5-flash-lite`), **Mistral** (`voxtral-mini-latest`). For these,
  per-provider capture profiles use the provider's own transcription path.
- Anthropic, DeepSeek, Grok, Ollama have no transcription — fall back to
  local Whisper (`small.en`).

### Transcription head-to-head (#577 Exp 3, 2026-04-20, 5 NPR eps ~30 min each)

Audio fixed at 32 kbps / 16 kHz / mono (Exp 1 winner). Reference transcripts
from `curated_5feeds_benchmark_v2`. Local Whisper small.en baseline WER ~11%.

| Provider / Model | WER (avg) | $/ep (32-min) | Wall | Notes |
| ---------------- | :-------: | :-----------: | :--: | ----- |
| `openai/whisper-1` | 8.2% | $0.20 | 68s | No caps; hard-coded anti-loop filters make it the most stable cloud option |
| `mistral/voxtral-mini-latest` | 8.6% (clean) | **$0.034** | **21s** | 6× cheaper than whisper-1 but **1/5 eps hallucinated** (109K-char loop); no native anti-loop filters. Requires `temperature=0.0` (applied) + output-length sanity check (applied) + pre-chunk to <25 min. New model `voxtral-mini-transcribe-2` (Feb 2026) ships with diarization/timestamps and is expected to replace voxtral-mini-latest for batch transcription — upgrade tracked separately. |
| `openai/gpt-4o-transcribe` | — | — | — | **Hard 1400s (23 min) duration cap.** All 5 eps failed. Needs chunking (see #286). |
| `openai/gpt-4o-mini-transcribe` | — | — | — | **Token budget cap** (narrower than the 1400s cap). All 5 eps failed. Needs chunking. |
| `gemini/gemini-2.5-flash-lite` | 72–931% | $0.01 | 16–121s | **Not suitable for verbatim transcription.** LLM-based audio without anti-loop filters. At default `max_output_tokens=8192` silently truncates → summary-style (72% WER). At raised cap, runs the hallucination loop (931% WER). Use `gemini-2.5-pro` or `-flash` with thinking for better results if Gemini is required; otherwise prefer Whisper/Voxtral. |

### Cost lever hierarchy for Whisper API path

1. **File-size vs duration** — API cost is duration-based. Bitrate affects file
   size (upload speed, 25 MB cap) but **not** per-minute cost. Lower bitrate is
   still worth it for smaller files (Exp 1: 32 kbps is the sweet spot).
2. **Silence trim** — direct duration cut. On tightly-edited benchmark fixtures
   the filter removed 0% (fixtures have no silences > 1 s at any threshold).
   On production NPR audio (#577 Exp 2-prod, 2026-04-20, 2 × 32-min episodes):
   `-50 dB / 2.0 s` (the previous default) trims 0%. `-30 dB / 0.5 s` trims
   **3.6%** with < 1% transcript char drop. `-25 dB / 0.5 s` trims **6.7%**
   with similar char drop (held pending validation on more diverse podcast
   types). `speech_optimal_v1.yaml` now uses `-30 dB / 0.5 s` as the moderate
   sweet spot — across all 5 deployment profiles that's a direct 3.6% API $
   saving on top of whatever provider they pick.
3. **Cheaper model** — `voxtral-mini-latest` is 6× cheaper at similar clean-run
   quality (pending hallucination mitigations). `gpt-4o-mini-transcribe` is 50%
   cheaper than `whisper-1` but blocked on chunking.

### Cross-provider caps and constraints

| Model | File size | Duration | Token budget | Notes |
| ----- | :-------: | :------: | :----------: | ----- |
| `openai/whisper-1` | 25 MB | — | — | Most permissive OpenAI option |
| `openai/gpt-4o-transcribe` | — | 1400s | — | Hard duration cap |
| `openai/gpt-4o-mini-transcribe` | — | — | yes (tighter than 1400s) | Needs chunking for any real podcast ep |
| `mistral/voxtral-mini-latest` | n/a | **30 min** (documented ceiling) | — | Exceeding 30 min triggers hallucination loops |
| `gemini/gemini-2.5-flash-lite` | inline ~20 MB (Files API ≥ 20 MB) | — | 8192 output default, 65536 cap | Silent summary-truncation at default cap |

### Local vs API breakeven — when to pick which (#577 Exp 4)

Inputs (measured, 32-min NPR episode, 32 kbps input):

| Path | Wall per ep | $ per ep | Throughput (parallel) |
| ---- | :---------: | :------: | :-------------------: |
| Local Whisper `small.en` on MPS | ~100s | $0.00 | 1 ep at a time (MPS shared) |
| Cloud `whisper-1` | ~68s | $0.20 | 50+ concurrent (tier-1 rate limit) |
| Cloud `voxtral-mini-latest` (clean runs) | ~21s | $0.034 | 10+ concurrent |

Decision rules:

- **Volume < 100 eps/day AND you're not time-sensitive** → local wins. Zero
  cost; the 100s/ep on MPS is fine to run overnight. API $ savings (at most
  $20/day on whisper-1) don't justify the API complexity or hallucination risk.
- **Volume 100–1,000 eps/day OR you need results inside an hour** → cloud API
  wins on wall-clock time. Pick `whisper-1` for reliability (no chunking
  needed, no hallucination surprises). Expect `$20–$200/day`.
- **Volume > 1,000 eps/day OR cost-sensitive at scale** → `voxtral-mini` with
  hallucination mitigations applied (`temperature=0.0`, pre-chunk to < 25 min,
  post-hoc length sanity check with fallback to `whisper-1`). 6× cheaper than
  `whisper-1`; expected ~`$35/1,000 eps` net after fallback overhead.

**Pathological case — a single ~3,000-word episode as fast as possible:**
`voxtral-mini-latest` wins wall-time (~21s). `whisper-1` is the conservative
alternative (~68s, predictable). Both beat local Whisper (~100s sequential).

**Break-even table (rough, 32-min NPR avg episode):**

| Daily volume | Local (1× MPS) | whisper-1 (50 parallel) | voxtral-mini (10 parallel + fallback) |
| :----------: | :------------: | :---------------------: | :----------------------------------: |
| 10 eps | 17 min, $0 | 14s, $2 | 21s, $0.40 |
| 100 eps | 2.8 hours, $0 | 2.3 min, $20 | 3.5 min, $4 |
| 1,000 eps | 28 hours, $0 | 23 min, $200 | 35 min, $40 |
| 10,000 eps | 11.5 days, $0 | 3.8 hours, $2,000 | 5.8 hours, $400 |

Rule of thumb: once you need throughput above 1 ep/min sustained, local on a
single Mac is out. Cloud API cost becomes the binding constraint, which is
why `voxtral-mini` with mitigations is the cost-optimal long-horizon pick.

---

## Recommended Configurations

### Configuration 1: Ultra-Budget ($0.016/100 episodes)

```yaml
# 97% cheaper than OpenAI. DeepSeek has detect_speakers wired but NER smoke
# has spaCy trf tied at F1=1.000, so local spaCy is cheaper and equivalent.
transcription_provider: whisper             # Free (local; DeepSeek no transcription API)
whisper_model: small.en                     # Production quality
speaker_detector_provider: spacy            # Free, local; F1=1.000 smoke
ner_model: en_core_web_trf
summary_provider: deepseek                  # $0.016/100, leads v2 bullets
deepseek_summary_model: deepseek-chat
deepseek_api_key: ${DEEPSEEK_API_KEY}
```

### Configuration 2: Quality-First (~$42/100 episodes)

```yaml
# Maximum quality via OpenAI (transcription + summarization).
transcription_provider: openai
openai_transcription_model: whisper-1
speaker_detector_provider: spacy            # F1 tied; skip the API call
ner_model: en_core_web_trf
summary_provider: openai
openai_summary_model: gpt-4o-mini           # v2 eval model; gpt-4o for top quality
openai_api_key: ${OPENAI_API_KEY}
```

### Configuration 3: Privacy-First ($0)

```yaml
# Data never leaves your device.
transcription_provider: whisper             # Local
whisper_model: small.en                     # 9.5% WER sweet spot
speaker_detector_provider: spacy            # Local spaCy trf > qwen3.5:9b NER (F1 1.0 vs 0.75)
ner_model: en_core_web_trf
summary_provider: ollama                    # Local Ollama
ollama_summary_model: qwen3.5:9b            # v2 local champion
llm_pipeline_mode: bundled                  # Stabilises Ollama JSON output
```

### Configuration 4: Speed-First (~$0.25/100 episodes)

```yaml
# Fast cloud summarization via Grok.
transcription_provider: whisper             # Local (Grok no transcription API)
whisper_model: small.en
speaker_detector_provider: spacy            # Free, tied F1 on smoke
ner_model: en_core_web_trf
summary_provider: grok
grok_summary_model: grok-3-mini             # Eval-validated; grok-2/grok-beta are stale IDs
grok_api_key: ${GROK_API_KEY}
```

### Configuration 5: EU Compliant (Mistral)

```yaml
# European data residency end-to-end (Mistral has its own transcription).
transcription_provider: mistral
mistral_transcription_model: voxtral-mini-latest
speaker_detector_provider: spacy            # Free, tied F1 on smoke
ner_model: en_core_web_trf
summary_provider: mistral
mistral_summary_model: mistral-large-latest
mistral_api_key: ${MISTRAL_API_KEY}
```

### Configuration 6: Free Development (~$0)

```yaml
# Maximize free tiers. Gemini free tier covers transcription + summary.
transcription_provider: gemini              # Gemini has audio input
gemini_transcription_model: gemini-2.5-flash-lite
speaker_detector_provider: spacy            # Free, tied F1 on smoke
ner_model: en_core_web_trf
summary_provider: gemini
gemini_summary_model: gemini-2.5-flash-lite
gemini_api_key: ${GEMINI_API_KEY}
```

---

## Summary — v2 held-out key takeaways

See [`eval-reports/EVAL_HELDOUT_V2_2026_04.md`](eval-reports/EVAL_HELDOUT_V2_2026_04.md)
for the full matrix. Headline findings:

| Axis | Winner | Score / note |
| --- | --- | --- |
| Cheapest cloud | **DeepSeek** | `$0.016/100 eps`, leads bullets non-bundled (43.1%) |
| Compound-scored cloud default | **Gemini 2.5-flash-lite** | Non-bundled: 0.564/0.479 bullets/para, 1.5s/ep, `$0.47/1k eps` |
| Best cloud bundled | **Anthropic** `claude-haiku-4-5` | Only provider where bundled is competitive (39.3% bullets, 39.2% para) |
| Best local | **qwen3.5:9b bundled** | 0.529/0.509 bullets/para, ~44s/ep, `$0` |
| EU residency | **Mistral** | End-to-end: `voxtral-mini-latest` transcription + `mistral-large-latest` |
| Real-time info | **Grok** | X/Twitter integration |
| Local bullets leader (non-bundled) | **qwen3.5:9b** | 0.580 — beats qwen3.5:35b (0.576) at 1/4 the size |

**Cost insight:** transcription is 90%+ of cloud pipeline cost. Local Whisper
`small.en` + cloud summarization is the high-leverage combination. Per-provider
cost numbers are in [EVAL_HELDOUT_V2](eval-reports/EVAL_HELDOUT_V2_2026_04.md);
older v1 numbers in this guide are superseded.

**Rankings change when the silver reference changes** — Sonnet 4.6 silver
favours verbose paragraph style; different silvers may produce different
orderings. See [eval methodology](eval-reports/index.md) for detail.

---

## Related Documentation

- [Provider Deep Dives](PROVIDER_DEEP_DIVES.md) — per-provider cards, magic quadrant,
  visual comparisons
- [Evaluation Reports](eval-reports/index.md) — methodology, metrics, and full
  comparison data
- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md)
- [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) — complete Ollama setup and
  troubleshooting
- [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)
- [ML Provider Reference](ML_PROVIDER_REFERENCE.md)
- [PRD-006: OpenAI Provider](../prd/PRD-006-openai-provider-integration.md)
- [PRD-009: Anthropic Provider](../prd/PRD-009-anthropic-provider-integration.md)
- [PRD-010: Mistral Provider](../prd/PRD-010-mistral-provider-integration.md)
- [PRD-011: DeepSeek Provider](../prd/PRD-011-deepseek-provider-integration.md)
- [PRD-012: Gemini Provider](../prd/PRD-012-gemini-provider-integration.md)
- [PRD-013: Grok Provider (xAI)](../prd/PRD-013-grok-provider-integration.md)
- [PRD-014: Ollama Provider](../prd/PRD-014-ollama-provider-integration.md)
