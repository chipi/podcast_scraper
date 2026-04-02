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
| **Highest Quality (cloud)** | Anthropic | Leads cloud ROUGE-L (32.6%) vs Sonnet 4.6 silver ([measured](eval-reports/EVAL_SMOKE_V1_2026_04.md#cloud-providers-sorted-by-rouge-l)) |
| **Fastest Cloud** | Gemini | 2.9s/ep paragraphs, 1.9s/ep bullets |
| **On-prem, quality first** | Ollama (qwen3.5:35b) | 29.9% ROUGE-L, competitive with cloud mid-tier |
| **On-prem, speed/quality** | Ollama (mistral-small3.2) | Best embedding (85.8% bullets), 30s/ep |
| **On-prem, low resource** | Ollama (llama3.2:3b) | 7.3s/ep paragraphs, 4.8s/ep bullets, only 2GB |
| **Full Capabilities** | OpenAI / Local ML | All 3 capabilities |
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

> **Note on the silver reference:** Results were re-measured in April 2026 against
> `silver_sonnet46_smoke_v1` (Claude Sonnet 4.6, selected via pairwise LLM judge).
> Rankings shifted significantly — see
> [why the rankings changed](eval-reports/EVAL_SMOKE_V1_2026_04.md#why-the-rankings-changed-vs-march-2026).
> The March 2026 numbers (vs GPT-4o silver) are preserved in the
> [March report](eval-reports/EVAL_SMOKE_V1_2026_03.md) for reference.

### Cloud providers — paragraphs (vs Sonnet 4.6 silver, April 2026)

**Best cloud provider:** **Anthropic** (`claude-haiku-4-5`) — leads on ROUGE-L (32.6%)
and embedding similarity (86.8%), driven by round-2 prompt tuning (+82% gain from
baseline). **Gemini** (`gemini-2.0-flash`) remains the fastest cloud option (2.9s/ep).

| Provider | ROUGE-L | Embed | Latency |
| -------- | ------- | ----- | ------- |
| **Anthropic** | **32.6%** | **86.8%** | 5.1s |
| DeepSeek | 28.3% | 81.5% | 9.8s |
| Gemini | 27.6% | 81.6% | **2.9s** |
| OpenAI | 25.7% | 82.5% | 8.2s |
| Mistral | 25.7% | 80.1% | 4.0s |
| Grok | 25.1% | 77.8% | 8.9s |

> The Anthropic model used here is `claude-haiku-4-5` (smallest/fastest Haiku). The
> silver reference is Claude Sonnet 4.6 — Anthropic scores well partly because the
> models share a generation family.

Full table:
[Smoke v1 report — Cloud providers](eval-reports/EVAL_SMOKE_V1_2026_04.md#cloud-providers-sorted-by-rouge-l)

### Local Ollama — paragraphs (vs Sonnet 4.6 silver, April 2026)

**Best local model:** **Qwen 3.5:35b** at 29.9% ROUGE-L (20s/ep) — competitive with
cloud mid-tier. **Qwen 3.5:27b** is close behind (29.2%) but nearly 4× slower (83s/ep).
**Mistral Small 3.2** is the best speed/quality balance for large-model budgets (25.2%,
46s/ep). **llama3.2:3b** (22.6%, 7.3s/ep) is the best fast/low-resource option.

| Model | ROUGE-L | Embed | Latency |
| ----- | ------- | ----- | ------- |
| qwen3.5:35b | **29.9%** | **81.8%** | 20.4s |
| qwen3.5:27b | 29.2% | 80.4% | 82.8s |
| qwen2.5:32b | 25.4% | 76.6% | 58.5s |
| mistral-small3.2 | 25.2% | 77.1% | 45.5s |
| qwen3.5:9b | 24.7% | 75.8% | 23.9s |
| llama3.2:3b | 22.6% | 78.9% | **7.3s** |

> Latencies are hardware-dependent (Apple M-series). Re-run on your machine.

### Local Ollama — bullets (vs Sonnet 4.6 bullets silver, April 2026)

For bullet JSON output, **llama3.2:3b** leads on ROUGE-L (35.5%, 4.8s/ep — fastest).
**Mistral Small 3.2** leads on embedding similarity (85.8%). **qwen2.5:7b** does not
reliably follow the JSON format — avoid it for the bullets track.

| Model | ROUGE-L | Embed | Latency |
| ----- | ------- | ----- | ------- |
| llama3.2:3b | **35.5%** | 79.8% | **4.8s** |
| mistral-small3.2 | 34.9% | **85.8%** | 30.5s |
| qwen3.5:27b | 33.5% | 84.7% | 53.6s |
| qwen3.5:9b | 33.3% | 83.4% | 15.4s |

Full tables:
[Smoke v1 report (April 2026)](eval-reports/EVAL_SMOKE_V1_2026_04.md)

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
│  🥇 BEST CLOUD QUALITY:  Anthropic Haiku  32.6% ROUGE-L (Apr 2026)        │
│  🥇 FASTEST CLOUD:       Gemini Flash     2.9s/ep paragraphs               │
│  🥇 LARGEST CONTEXT:     Gemini Pro       2,000,000 tokens                 │
│  🥇 BEST FREE TIER:      Gemini/Grok      Generous limits                  │
│  🥇 REAL-TIME INFO:      Grok             X/Twitter integration            │
│  🥇 EU COMPLIANT:        Mistral          European summarization provider  │
│  🥇 COMPLETE PRIVACY:    Local/Ollama     Data never leaves device         │
│  🥇 BEST LOCAL (para):   qwen3.5:35b      29.9% ROUGE-L, 20s/ep           │
│  🥇 BEST LOCAL (bullets):llama3.2:3b      35.5% ROUGE-L, 4.8s/ep          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📈 COST INSIGHT:                                                          │
│     Transcription = 90%+ of cloud costs                                    │
│     → Use local Whisper + cloud text = massive savings                     │
│                                                                             │
│  📊 EVAL INSIGHT (Apr 2026, vs Sonnet 4.6 silver):                        │
│     Anthropic Haiku leads cloud paragraphs (32.6% ROUGE-L, 86.8% embed)   │
│     qwen3.5:35b is the only on-prem model in cloud quality range (29.9%)   │
│     Rankings change when the silver reference changes — see eval reports   │
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
