# AI Provider Comparison Guide

> **Your decision-making resource for choosing the right AI provider.**

A focused analysis of summarization and capability providers supported by
podcast_scraper: local ML, **hybrid MAP-REDUCE** (hybrid_ml), and 7 LLM providers.
This guide answers **"which provider should I pick?"** with decision matrices, cost
analysis, and empirical conclusions.

**Companion pages:**

- [Provider Deep Dives](PROVIDER_DEEP_DIVES.md) — per-provider reference cards, magic
  quadrant, visual comparisons
- [Evaluation Reports](eval-reports/index.md) — methodology, metric definitions, and
  the full library of measured comparison reports

---

## Implementation Status

All providers below are **implemented and acceptance-tested** (v2.4.0+).

| Provider | Status | RFC | Notes |
| ---------- | :------: | :---: | ------- |
| **Local ML** | ✅ Implemented | - | Default provider (Whisper + spaCy + Transformers): transcription, speaker detection, summarization |
| **Hybrid ML** | ✅ Implemented | RFC-042 | Summarization only: MAP (LongT5) + REDUCE (transformers / Ollama / llama_cpp) |
| **OpenAI** | ✅ Implemented | RFC-013 | Transcription + summarization (Whisper API + GPT API) |
| **Gemini** | ✅ Implemented | RFC-035 | Transcription + summarization (no speaker detection) |
| **Mistral** | ✅ Implemented | RFC-033 | Summarization only (EU data residency) |
| **Anthropic** | ✅ Implemented | RFC-032 | Summarization only (no transcription or speaker detection) |
| **DeepSeek** | ✅ Implemented | RFC-034 | Summarization only; ultra low-cost |
| **Grok** | ✅ Implemented | RFC-036 | Summarization only; real-time information access |
| **Ollama** | ✅ Implemented | RFC-037 | Transcription, speaker detection, summarization; local self-hosted LLMs, zero cost, complete privacy |

For hybrid_ml (MAP-REDUCE) configuration and REDUCE backends (Ollama, llama_cpp,
transformers), see [ML Provider Reference](ML_PROVIDER_REFERENCE.md) and
[Configuration API](../api/CONFIGURATION.md).

---

## Key Statistics at a Glance

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROVIDER LANDSCAPE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  9 Summarization Options  │  (Hybrid = MAP+REDUCE)  │  3 Full-Stack Ready  │
│  ════════════════════     │  ═══════════════════════ │  ═══════════════     │
│  ✅ Local ML              │  ✅ Hybrid ML (RFC-042)  │  ✅ Local ML          │
│  ✅ Hybrid ML             │  MAP + Ollama/llama_cpp │  ✅ OpenAI (tx+sum)  │
│  ✅ OpenAI                │  or transformers REDUCE  │  ✅ Ollama            │
│  ✅ Gemini                │                         │                      │
│  ✅ Mistral               │                         │                      │
│  ✅ Anthropic / DeepSeek  │                         │                      │
│  ✅ Grok / Ollama         │                         │                      │
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

| If you need... | Choose | Why |
| :------------- | :----: | :-- |
| **Complete Privacy** | Local ML / Hybrid ML / Ollama | Data never leaves your device |
| **Lowest Cost** | Local ML / Hybrid ML / Ollama | $0 (just electricity) |
| **Highest Quality** | OpenAI | Industry leader ([measured](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)) |
| **Full Capabilities** | Local ML / Ollama | All 3 capabilities (transcription + speaker detection + summarization) |
| **Local MAP + LLM REDUCE** | Hybrid ML (Ollama/llama_cpp) | LongT5 MAP + local LLM synthesis (RFC-042) |
| **Real-Time Info** | Grok | Real-time information access (RFC-036) |
| **Lowest Cloud Cost** | DeepSeek | 95% cheaper than OpenAI (RFC-034) |
| **EU Data Residency** | Mistral | European servers (RFC-033) |
| **Huge Context** | Gemini | 2 million token window (RFC-035) |
| **Free Development** | Gemini / Grok | Generous free tiers (RFC-035, RFC-036) |
| **Self-Hosted** | Ollama | Offline/air-gapped (RFC-037) |

---

## Empirical Highlights

All claims below are backed by measured data. For the full metrics tables, methodology,
and metric definitions, see the [Evaluation Reports](eval-reports/index.md).

### Cloud providers (vs silver GPT-4o reference)

**Best non-OpenAI cloud:** **Gemini** (`gemini-2.0-flash`) — highest ROUGE-L (33.3%)
and embedding similarity (87.3%) among non-OpenAI providers, with the fastest latency
(2.7s/ep). **Mistral** (`mistral-small-latest`) is a close second (32.5% ROUGE-L,
2.8s/ep).

| Provider | ROUGE-L | Embed | Latency |
| -------- | ------- | ----- | ------- |
| OpenAI (GPT-4o) | 58.8% | 92.7% | 15.4s |
| Gemini | 33.3% | 87.3% | 2.7s |
| Mistral | 32.5% | 84.8% | 2.8s |
| Grok | 29.5% | 85.4% | 13.2s |
| Anthropic | 29.4% | 81.8% | 4.8s |
| DeepSeek | 26.3% | 85.0% | 14.2s |

> OpenAI scores highest because the silver reference is GPT-4o. Compare non-OpenAI
> providers against each other for a fairer picture.

Full table:
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)

### Local Ollama (vs silver GPT-4o reference)

**Best local models:** **Mistral Small 3.2** and **Qwen 2.5:32b** tie at **38.4%
ROUGE-L** — both outperform every cloud provider except OpenAI. Mistral Small 3.2
leads on ROUGE-1, BLEU, and embedding similarity. **Qwen 3.5:9b** (with
`reasoning_effort: none`) is the best smaller model (~30% ROUGE-L, 85.2% embed).

| Model | ROUGE-L | Embed | Latency |
| ----- | ------- | ----- | ------- |
| mistral-small3.2:latest | 38.4% | 85.8% | 48.6s |
| qwen2.5:32b | 38.4% | 85.2% | 54.8s |
| mistral:7b | 32.8% | 80.4% | 17.4s |
| qwen3.5:9b | 30.3% | 85.2% | 21.9s |
| qwen2.5:7b | 28.3% | 84.9% | 12.1s |

> Ollama latencies are hardware-dependent. Re-run on your machine before making
> decisions.

Full table:
[Smoke v1 report — Local Ollama](eval-reports/EVAL_SMOKE_V1_2026_03.md#local-ollama-ranked-by-rouge-l)

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

⚠️  At 10,000 episodes/month, OpenAI full stack costs $3,740!
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

## Recommended Configurations

### Configuration 1: Ultra-Budget ($0.016/100 episodes)

```yaml
# 97% cheaper than OpenAI
transcription_provider: whisper       # Free (local)
speaker_detector_provider: spacy      # Free (local; DeepSeek: summarization only)
summary_provider: deepseek            # $0.016/100
deepseek_api_key: ${DEEPSEEK_API_KEY}
```

### Configuration 2: Quality-First (~$42/100 episodes)

```yaml
# Maximum quality
transcription_provider: openai
speaker_detector_provider: spacy      # OpenAI: summarization only (no speaker detection)
summary_provider: openai
openai_summary_model: gpt-5
openai_api_key: ${OPENAI_API_KEY}
```

### Configuration 3: Privacy-First ($0)

```yaml
# Data never leaves your device
transcription_provider: whisper       # Local
speaker_detector_provider: ollama     # Local Ollama
summary_provider: ollama              # Local Ollama
ollama_speaker_model: llama3.1:8b
ollama_summary_model: llama3.1:8b
# For better quality (12-16 GB RAM):
# ollama_speaker_model: llama3.3:latest
# ollama_summary_model: llama3.3:latest
```

### Configuration 4: Speed-First (~$0.25/100 episodes)

```yaml
# Fast cloud summarization
transcription_provider: whisper       # Local
speaker_detector_provider: spacy      # Local (Grok: summarization only)
summary_provider: grok
grok_summary_model: grok-2
grok_api_key: ${GROK_API_KEY}
```

### Configuration 5: EU Compliant (Mistral Summarization)

```yaml
# European data residency for summarization; local for other capabilities
transcription_provider: whisper            # Local (Mistral: summarization only)
speaker_detector_provider: spacy           # Local (Mistral: summarization only)
summary_provider: mistral
mistral_summary_model: mistral-large-latest
mistral_api_key: ${MISTRAL_API_KEY}
```

### Configuration 6: Free Development (~$0)

```yaml
# Maximize free tiers
transcription_provider: whisper       # Local
speaker_detector_provider: spacy      # Local (Gemini/Grok don't support speaker detection)
summary_provider: grok                # Free tier
grok_summary_model: grok-beta
```

---

## Summary

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KEY TAKEAWAYS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🥇 CHEAPEST CLOUD:      DeepSeek         $0.016/100 episodes (97% off)    │
│  🥇 HIGHEST QUALITY:     OpenAI GPT-4o    Industry benchmark               │
│  🥇 LARGEST CONTEXT:     Gemini Pro       2,000,000 tokens                 │
│  🥇 BEST FREE TIER:      Gemini/Grok      Generous limits                  │
│  🥇 REAL-TIME INFO:      Grok             X/Twitter integration            │
│  🥇 EU COMPLIANT:        Mistral          European summarization provider  │
│  🥇 COMPLETE PRIVACY:    Local/Ollama     Data never leaves device         │
│  🥇 BEST LOCAL MODEL:    Mistral Small 3.2 / Qwen 2.5:32b (38.4% ROUGE-L)│
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📈 COST INSIGHT:                                                          │
│     Transcription = 90%+ of cloud costs                                    │
│     → Use local Whisper + cloud text = massive savings                     │
│                                                                             │
│  📊 EVAL INSIGHT:                                                          │
│     Gemini is best non-OpenAI cloud (33.3% ROUGE-L, 87.3% embed)          │
│     Ollama top models beat all cloud except OpenAI (38.4% ROUGE-L)         │
│     → See eval reports for full data                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
