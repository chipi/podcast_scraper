# Provider Deep Dives

> **Per-provider reference cards, magic quadrant analysis, and visual comparisons.**

This page is the detailed reference for each provider's capabilities, pricing, models,
and strategic positioning. For the decision-oriented summary, see the
[AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md). For measured
performance numbers, see the [Evaluation Reports](eval-reports/index.md).

---

## Provider Reference Cards

### 1. Local ML (Default)

```text
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL ML PROVIDERS                                             │
│  ═══════════════════                                            │
│                                                                 │
│   Cost:     $0 (just electricity)                            │
│   Speed:    Moderate (GPU dependent)                          │
│   Quality:  Good                                              │
│   Privacy:  ████████████████████ 100% (complete)              │
│                                                                 │
│  Components:                                                    │
│  ├──  Transcription: OpenAI Whisper (local)                  │
│  ├──  Speaker Det:   spaCy NER models                         │
│  └──  Summarization: Hugging Face BART/LED                    │
│                                                                 │
│  Best For: Privacy, offline use, zero ongoing cost              │
└─────────────────────────────────────────────────────────────────┘
```

> For detailed ML model comparisons, see
> [ML Model Comparison Guide](ML_MODEL_COMPARISON_GUIDE.md).

**Hardware Requirements:**

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| RAM | 8 GB | 16 GB+ |
| GPU VRAM | None (CPU) | 8 GB+ |
| Storage | 5 GB | 20 GB |

---

### 2. OpenAI

```text
┌─────────────────────────────────────────────────────────────────┐
│  OPENAI                                        Industry Leader  │
│  ══════                                                         │
│                                                                 │
│   Cost:     $$$ (Premium pricing)                             │
│   Speed:    Fast (100 tok/s)                                  │
│   Quality:  ████████████████████ Best                         │
│   Privacy:  ████████████░░░░░░░░ Standard (US)                │
│   Tracking: Built-in token/audio usage metrics for cost eval  │
│   Limit:    25 MB audio file size limit for Whisper API       │
│                                                                 │
│  Models:                                                        │
│  ├── GPT-4o       $5.00/$15.00  │ Best quality                 │
│  ├── GPT-4o-mini  $0.15/$0.60   │  Production recommended     │
│  └── Whisper      $0.006/min    │ Transcription                │
│                                                                 │
│  Best For: Quality-critical production, reliable workflows      │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)
(58.8% ROUGE-L, 92.7% embed — silver reference bias applies).

---

### 3. Anthropic (Claude)

```text
┌─────────────────────────────────────────────────────────────────┐
│  ANTHROPIC CLAUDE                              Safety Focused   │
│  ═══════════════                                                │
│                                                                 │
│   Cost:     $$ (Competitive)                                  │
│   Speed:    Fast (100 tok/s)                                  │
│   Quality:  ███████████████████░ Excellent                    │
│   Privacy:  ████████████░░░░░░░░ Standard (US)                │
│    Summarization only (no transcription or speaker detection) │
│                                                                 │
│  Models:                                                        │
│  ├── Claude Haiku 4.5   $1/$5   │  Eval/acceptance alias      │
│  ├── claude-haiku-4-5            │ Anthropic alias (current)    │
│  └── claude-3-5-sonnet-20241022 │ Deprecated (404)             │
│                                                                 │
│  Best For: Quality text, nuanced content, safety alignment      │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)
(29.4% ROUGE-L, 81.8% embed, 4.8s/ep).

---

### 4. Mistral

```text
┌─────────────────────────────────────────────────────────────────┐
│  MISTRAL                                       European Leader  │
│  ═══════                                                        │
│                                                                 │
│   Cost:     $-$$ (Competitive)                                │
│   Speed:    Fast                                              │
│   Quality:  ██████████████████░░ Very Good                    │
│   Privacy:  ████████████████░░░░ High (EU servers)            │
│    Summarization only (no transcription or speaker detection) │
│                                                                 │
│  Models:                                                        │
│  ├── Large 3      $2/$6      │  Production                   │
│  └── Small 3.1    $0.10/$0.30│  Dev/test (cheapest!)         │
│                                                                 │
│  Best For: EU compliance, summarization with data residency     │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)
(32.5% ROUGE-L, 84.8% embed, 2.8s/ep — fastest cloud provider).

---

### 5. DeepSeek

```text
┌─────────────────────────────────────────────────────────────────┐
│  DEEPSEEK                                      Ultra Low Cost   │
│  ════════                                                       │
│                                                                 │
│   Cost:     $ (95% cheaper than OpenAI!)                      │
│   Speed:    Fast (150 tok/s)                                  │
│   Quality:  ██████████████░░░░░░ Good                         │
│   Privacy:  ████████░░░░░░░░░░░░ China servers                │
│    Summarization only (no transcription or speaker detection) │
│                                                                 │
│  Models:                                                        │
│  ├── DeepSeek Chat      $0.28/$0.42 (cache miss)               │
│  ├── DeepSeek Chat      $0.028/$0.42 (cache hit!)            │
│  └── DeepSeek Reasoner  Complex reasoning tasks                │
│                                                                 │
│   $0.016/100 episodes vs $0.55 OpenAI = 97% SAVINGS          │
│                                                                 │
│  Best For: Budget optimization, bulk processing, startups       │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)
(26.3% ROUGE-L, 85.0% embed, 14.2s/ep).

---

### 6. Google Gemini

```text
┌─────────────────────────────────────────────────────────────────┐
│  GOOGLE GEMINI                                 Massive Context  │
│  ═════════════                                                  │
│                                                                 │
│   Cost:     $ (Generous free tier!)                           │
│   Speed:    Fast                                              │
│   Quality:  ██████████████████░░ Very Good                    │
│   Privacy:  ████████████░░░░░░░░ Standard (Google)            │
│  Transcription + summarization (no speaker detection)        │
│                                                                 │
│  Models:                                                        │
│  ├── Gemini 2.0 Flash  $0.10/$0.40  │  Dev/test              │
│  ├── Gemini 1.5 Pro    $1.25/$5.00  │  Production            │
│  └── Gemini 1.5 Flash  $0.075/$0.30 │ Budget                   │
│                                                                 │
│   2 MILLION TOKEN CONTEXT - Process entire seasons!          │
│                                                                 │
│  FREE TIER: 15 RPM, 1M TPM, 1500 RPD                           │
│                                                                 │
│  Best For: Long content, free development, multimodal           │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)
(33.3% ROUGE-L, 87.3% embed, 2.7s/ep — best non-OpenAI cloud).

---

### 7. Grok

```text
┌─────────────────────────────────────────────────────────────────┐
│  GROK                                          Real-Time Info   │
│  ════                                                           │
│                                                                 │
│   Cost:     $ (Affordable)                                    │
│   Feature:  ████████████████████ Real-time X/Twitter access  │
│   Quality:  ██████████████░░░░░░ Good (xAI models)           │
│   Privacy:  ████████████░░░░░░░░ Standard (US)                │
│    Summarization only (no transcription or speaker detection) │
│                                                                 │
│  Models (xAI's Grok):                                           │
│  ├── grok-3-mini                │  Eval/acceptance (use this) │
│  └── grok-2                     │ 400 model not found           │
│                                                                 │
│   Access real-time information via X/Twitter integration!   │
│                                                                 │
│  FREE TIER: grok-beta available                                │
│                                                                 │
│  Best For: Real-time information, current events, X/Twitter     │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Cloud LLMs](eval-reports/EVAL_SMOKE_V1_2026_03.md#cloud-llms-ranked-by-rouge-l)
(29.5% ROUGE-L, 85.4% embed, 13.2s/ep).

---

### 8. Ollama (Local LLMs)

```text
┌─────────────────────────────────────────────────────────────────┐
│  OLLAMA                                        Self-Hosted      │
│  ══════                                                         │
│                                                                 │
│   Cost:     $0 per request (hardware only)                    │
│   Speed:    Slow-Medium (hardware dependent, ~30 tok/s)        │
│   Quality:  ██████████████░░░░░░ Good (model dependent)       │
│   Privacy:  ████████████████████ 100% Complete                │
│  Full-stack: transcription, speaker detection, summarization │
│                                                                 │
│  Recommended Models (by Use Case):                             │
│  ├── qwen2.5:7b        8 GB+ RAM  │  Best JSON, GIL          │
│  ├── llama3.1:8b       8 GB+ RAM  │  General purpose          │
│  ├── mistral:7b        8 GB+ RAM  │  Fastest inference        │
│  ├── gemma2:9b         12 GB+ RAM │  Balanced quality/speed   │
│  └── phi3:mini         4 GB+ RAM  │  Dev/test, lightweight   │
│                                                                 │
│  Setup Requirements:                                           │
│  ├── Install Ollama:   brew install ollama (macOS)            │
│  ├── Start server:     ollama serve (keep running)            │
│  └── Pull models:      ollama pull qwen2.5:7b (recommended)    │
│                                                                 │
│  Hardware Recommendations:                                      │
│  ├── 4 GB+ RAM:        phi3:mini (dev/test only)              │
│  ├── 8 GB+ RAM:        qwen2.5:7b, llama3.1:8b, mistral:7b    │
│  └── 12 GB+ RAM:       gemma2:9b (balanced quality/speed)     │
│                                                                 │
│   Zero API costs, unlimited usage, complete data privacy      │
│                                                                 │
│  Best For: Privacy-critical, offline/air-gapped, unlimited    │
│            processing, enterprises, cost optimization           │
│                                                                 │
│   See [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md)      │
└─────────────────────────────────────────────────────────────────┘
```

**Measured performance:** See
[Smoke v1 report — Local Ollama](eval-reports/EVAL_SMOKE_V1_2026_03.md#local-ollama-ranked-by-rouge-l)
(top: Mistral Small 3.2 / Qwen 2.5:32b at 38.4% ROUGE-L).

---

## Magic Quadrant

A Gartner-style analysis plotting all providers across two strategic dimensions:

- **X-Axis: Completeness of Vision** — full-stack capabilities, context window, free
  tiers, innovation
- **Y-Axis: Ability to Execute** — quality, speed, reliability, cost-effectiveness

```text
                           ABILITY TO EXECUTE
                                  ▲
                                  │
High │    CHALLENGERS     │      LEADERS
     │                    │
     │    ┌───────────┐   │   ┌─────────────┐
     │    │ Anthropic │   │   │   OpenAI    │ ← Quality benchmark
     │    │  Claude   │   │   │   GPT-4o    │
     │    └───────────┘   │   └─────────────┘
             │         ▲          │          ▲
             │         │          │   ┌──────┴──────┐
             │    High quality    │   │   Gemini    │ ← 2M context + free tier
             │    but text-only   │   │             │
             │                    │   └─────────────┘
             │   ┌─────────┐      │          ▲
             │   │  Grok   │──────┼──────────┤
             │   │  Real │      │   ┌──────┴──────┐
             │   └─────────┘      │   │   Mistral   │ ← EU summarization
             │       ▲            │   │           │
             │   Speed champion   │   └─────────────┘
             │                    │
      ───────┼────────────────────┼──────────────────────────────►
             │                    │              COMPLETENESS OF VISION
             │                    │
             │    NICHE PLAYERS   │     VISIONARIES
             │                    │
             │   ┌───────────┐    │   ┌─────────────┐
             │   │  Ollama   │    │   │  Local ML   │ ← Zero cost + full stack
             │   │         │    │   │   (Default) │
             │   └───────────┘    │   └─────────────┘
             │       ▲            │          ▲
             │   Offline/private  │   Hardware required
             │   but needs HW     │   but complete control
             │                    │
             │   ┌───────────┐    │
             │   │ DeepSeek  │────┼───► Extreme value
             │   │   97%   │    │      but China-based
             │   └───────────┘    │
        Low  │                    │
             │                    │
             └────────────────────┴────────────────────────────────
                    Limited                              Comprehensive
```

### Quadrant Analysis

| Quadrant | Providers | Characteristics | Best For |
| -------- | --------- | --------------- | -------- |
| **Leaders** | OpenAI, Gemini, Mistral | High quality, proven reliability, broad capabilities | Production workloads, quality-critical apps |
| **Challengers** | Anthropic, Grok | Excellent execution but limited scope | Text-only processing, real-time information |
| **Visionaries** | Local ML, DeepSeek | Innovative value proposition, some trade-offs | Cost optimization, privacy, experimentation |
| **Niche Players** | Ollama | Specialized use case, strong in specific domain | Offline, enterprise security, self-hosted |

### Provider Scores (0-10)

| Provider | Vision Score | Execution Score | Quadrant | Key Strength |
| -------- | :----------: | :-------------: | -------- | ------------ |
| **OpenAI** | 9 | 10 | Leader | Quality benchmark |
| **Gemini** | 10 | 8 | Leader | 2M context + free tier |
| **Mistral** | 8 | 7 | Leader | EU compliance + summarization |
| **Anthropic** | 5 | 9 | Challenger | Safety + quality |
| **Grok** | 5 | 8 | Challenger | Real-time info |
| **Local ML** | 8 | 5 | Visionary | Zero cost + privacy |
| **DeepSeek** | 4 | 7 | Visionary | 97% cost savings |
| **Ollama** | 4 | 6 | Niche | Offline + self-hosted |

### Movement Predictions (2026)

```text
                                    ▲ Ability to Execute
                                    │
                                    │     ┌──────────────┐
                                    │     │   OpenAI     │ ← Maintains lead
                                    │     │   ●──────●   │
                                    │     └──────────────┘
                                    │
                                    │     ┌──────────────┐
                                    │     │   Gemini     │ ← Rising challenger
                                    │     │      ●═══▶   │
                                    │     └──────────────┘
                                    │
                                    │     ┌──────────────┐
                                    │     │   Grok       │ ← Adding capabilities?
                                    │     │   ●═══▶      │
                                    │     └──────────────┘
                                    │
                                    │     ┌──────────────┐
                                    │     │  DeepSeek    │ ← Quality improving
                                    │     │      ●       │
                                    │     │      ║       │
                                    │     │      ▼       │
                                    │     └──────────────┘
                    ────────────────┼──────────────────────────────►
                                    │                    Completeness of Vision

    Legend:  ● Current position    ═══▶ Predicted movement
```

### Strategic Recommendations by Quadrant

**LEADERS (OpenAI, Gemini, Mistral)**

> *"Safe bets for production. Choose based on specific needs."*

| Provider | Choose When... |
| -------- | -------------- |
| OpenAI | Quality is paramount, budget available |
| Gemini | Need huge context (2M), want free tier |
| Mistral | EU data residency required (summarization only) |

**CHALLENGERS (Anthropic, Grok)**

> *"Excellent at what they do, but not full-stack."*

| Provider | Choose When... |
| -------- | -------------- |
| Anthropic | Text quality matters, safety-first |
| Grok | Real-time information access needed |

**VISIONARIES (Local ML, DeepSeek)**

> *"Trade-offs for significant advantages."*

| Provider | Choose When... |
| -------- | -------------- |
| Local ML | Zero cost + privacy + offline |
| DeepSeek | Extreme budget constraints (97% savings) |

**NICHE PLAYERS (Ollama)**

> *"Perfect for specific use cases."*

| Provider | Choose When... |
| -------- | -------------- |
| Ollama | Enterprise security, air-gapped, unlimited processing |

---

## Visual Comparisons

### Cost Comparison (Text Processing per 100 Episodes)

```text
Cost Scale (logarithmic feel - lower is better)
═══════════════════════════════════════════════════════════════════════════

Local ML     $0.00 │
Ollama       $0.00 │
DeepSeek     $0.02 │▏
Grok         $0.03 │▎
Mistral      $0.11 │█
Anthropic    $0.40 │███
OpenAI       $0.55 │████
                   └────────────────────────────────────────────────────────
                   $0                                                   $0.60

 DeepSeek is 97% cheaper than OpenAI for text processing!
```

### Speed Comparison (Relative Performance)

```text
Inference Speed (tokens/second)
═══════════════════════════════════════════════════════════════════════════

Grok         100  │██████████                                           1x
DeepSeek     150  │███████████████                                      1.5x
OpenAI       100  │██████████                                           1x
Anthropic    100  │██████████                                           1x
Gemini       100  │██████████                                           1x
Mistral      100  │██████████                                           1x
Local GPU     50  │█████                                               0.5x
Ollama        30  │███                                                 0.3x
              0   └────────────────────────────────────────────────────────
                  0                     100                            150

 Grok provides real-time information access via X/Twitter integration!
```

### Quality Ranking (Subjective)

```text
Quality Score (1-10)
═══════════════════════════════════════════════════════════════════════════

OpenAI GPT-4o      │██████████████████████████████████████████████████│ 10
Claude Sonnet     │█████████████████████████████████████████████     │  9
Gemini Pro        │████████████████████████████████████████          │  8
Mistral Large     │███████████████████████████████████               │  7
Ollama 70B        │███████████████████████████████████               │  7
DeepSeek          │██████████████████████████████                    │  6
Grok              │██████████████████████████████                    │  6
Local BART        │█████████████████████████                         │  5
                  └────────────────────────────────────────────────────────
                  0                    5                              10

 OpenAI remains the quality leader, but alternatives close the gap!
```

### Privacy Level

```text
Privacy Scale (Higher = More Private)
═══════════════════════════════════════════════════════════════════════════

Local ML     │████████████████████████████████████████████████│ Complete
Ollama       │████████████████████████████████████████████████│ Complete
Mistral        │███████████████████████████████████████         │ EU Servers
OpenAI           │██████████████████████████████                  │ US Servers
Anthropic        │██████████████████████████████                  │ US Servers
Google           │██████████████████████████████                  │ Google Cloud
Grok             │██████████████████████████████                  │ US Servers
DeepSeek           │████████████████████                            │ China Servers
                      └────────────────────────────────────────────────────

 For maximum privacy, use Local ML or Ollama - data never leaves your device!
```

---

## Capability Matrix

```text
                    ┌─────────────────────────────────────────────────┐
                    │           CAPABILITY SUPPORT MATRIX              │
                    ├─────────────────────────────────────────────────┤
                    │  Provider      │ Status   │  Trans │  Speaker │  Summary │
                    ├────────────────┼──────────┼──────────┼────────────┼────────────┤
                    │  Local ML      │ Impl  │       │         │         │
                    │  Hybrid ML     │ Impl  │       │         │         │
                    │  OpenAI        │ Impl  │       │         │         │
                    │  Gemini        │ Impl  │       │         │         │
                    │  Ollama        │ Impl  │       │         │         │
                    ├────────────────┼──────────┼──────────┼────────────┼────────────┤
                    │  Anthropic     │ Impl  │       │         │         │
                    │  Mistral       │ Impl  │       │         │         │
                    │  DeepSeek      │ Impl  │       │         │         │
                    │  Grok          │ Impl  │       │         │         │
                    └─────────────────────────────────────────────────────────────────┘

    Implemented (9): Local ML, Hybrid ML, OpenAI, Gemini, Ollama, Anthropic, Mistral, DeepSeek, Grok
```

---

## Related Documentation

- [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) — decision matrix
  and recommended configurations
- [Evaluation Reports](eval-reports/index.md) — methodology, metrics, and measured
  performance
- [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) — complete Ollama setup and
  troubleshooting
- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md)
- [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)
- [ML Provider Reference](ML_PROVIDER_REFERENCE.md)
