# AI Provider Comparison Guide

> **Your complete decision-making resource for choosing the right AI provider**

A comprehensive analysis of summarization and capability providers supported by podcast_scraper:
local ML, **hybrid MAP-REDUCE** (hybrid_ml), and 7 LLM providers. This guide is the **single place**
for differences between providers: capabilities, cost, quality, speed, privacy, and
**empirical smoke evaluation** (metrics vs silver reference, latency, model IDs).

---

## ⚠️ Implementation Status

**Important:** This guide includes both **implemented** and **planned** providers.
Check the status below before making decisions.

### ✅ Implemented Providers (v2.4.0+)

| Provider | Status | RFC | Notes |
| ---------- | :------: | :---: | ------- |
| **Local ML** | ✅ **Implemented** | - | Default provider (Whisper + spaCy + Transformers) |
| **Hybrid ML** | ✅ **Implemented** | RFC-042 | Summarization only: MAP (LongT5) + REDUCE (transformers / Ollama / llama_cpp) |
| **OpenAI** | ✅ **Implemented** | RFC-013 | Full-stack: Whisper API + GPT API |
| **Gemini** | ✅ **Implemented** | RFC-035 | Full-stack: Gemini API for all capabilities |
| **Mistral** | ✅ **Implemented** | RFC-033 | Full-stack: Mistral API (EU data residency) |
| **Anthropic** | ✅ **Implemented** | RFC-032 | Speaker detection + summarization (transcription not supported) |
| **DeepSeek** | ✅ **Implemented** | RFC-034 | Speaker detection + summarization; ultra low-cost |
| **Grok** | ✅ **Implemented** | RFC-036 | Real-time information access, speaker detection + summarization |
| **Ollama** | ✅ **Implemented** | RFC-037 | Local self-hosted LLMs, zero cost, complete privacy |

### 📋 Planned Providers (RFCs in Draft Status)

None currently; all listed providers above are implemented.

**Note:** For hybrid_ml (MAP-REDUCE) configuration and REDUCE backends (Ollama, llama_cpp, transformers), see [ML Provider Reference](ML_PROVIDER_REFERENCE.md) and [Configuration API](../api/CONFIGURATION.md).

---

## 📊 Key Statistics at a Glance

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROVIDER LANDSCAPE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  9 Summarization Options  │  (Hybrid = MAP+REDUCE)  │  5 Full-Stack Ready  │
│  ════════════════════     │  ═══════════════════════ │  ═══════════════     │
│  ✅ Local ML              │  ✅ Hybrid ML (RFC-042)  │  ✅ Local ML          │
│  ✅ Hybrid ML             │  MAP + Ollama/llama_cpp │  ✅ OpenAI            │
│  ✅ OpenAI                │  or transformers REDUCE  │  ✅ Gemini            │
│  ✅ Gemini                │                         │  ✅ Mistral           │
│  ✅ Mistral               │                         │  ✅ Ollama            │
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
│  DeepSeek ─── Grok ─── Anthropic ─── Gemini ─── OpenAI (text) ─── OpenAI (full) │
│   ($0.02)    ($0.03)    ($0.40)      ($0.95)    ($0.55)           ($37)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Decision Matrix

| If you need... | Choose | Status | Why |
| :------------- | :----: | :----: | :-- |
| 🔒 **Complete Privacy** | Local ML / Hybrid ML / Ollama | ✅ Implemented | Data never leaves your device |
| 💰 **Lowest Cost** | Local ML / Hybrid ML / Ollama | ✅ Implemented | $0 (just electricity) |
| 🏆 **Highest Quality** | OpenAI | ✅ Implemented | Industry leader |
| 🌐 **Full Capabilities** | OpenAI / Local ML | ✅ Implemented | All 3 capabilities |
| 📝 **Local MAP + LLM REDUCE** | Hybrid ML (Ollama/llama_cpp) | ✅ Implemented | LongT5 MAP + local LLM synthesis (RFC-042) |
| 🌐 **Real-Time Info** | Grok | ✅ Implemented | Real-time information access (RFC-036) |
| 💰 **Lowest Cloud Cost** | DeepSeek | ✅ Implemented | 95% cheaper than OpenAI (RFC-034) |
| 🇪🇺 **EU Data Residency** | Mistral | ✅ Implemented | European servers (RFC-033) |
| 📚 **Huge Context** | Gemini | ✅ Implemented | 2 million token window (RFC-035) |
| 🆓 **Free Development** | Gemini / Grok | ✅ Implemented | Generous free tiers (RFC-035, RFC-036) |
| 🏠 **Self-Hosted** | Ollama | ✅ Implemented | Offline/air-gapped (RFC-037, Issue #196) |

---

## 📊 Empirical summarization comparison (smoke evaluation)

This section is the **single place** for measured differences between summarization providers. All metrics are vs a **silver reference** (GPT-4o on the same episodes). Dataset: `curated_5feeds_smoke_v1` (5 episodes). Reference: `silver_gpt4o_smoke_v1`.

**Interpretation:** Higher ROUGE / BLEU / Embed / Coverage = closer to reference; lower WER = better. Latency = seconds per episode (inference only).

**Best drop-in vs silver (this smoke):** After OpenAI (same family as the silver run), **Gemini** `gemini-2.0-flash` scores highest among measured cloud APIs on **ROUGE-L** and **embedding cosine**; **Mistral API** `mistral-small-latest` is next on ROUGE-L and the fastest in that set. On **Ollama**, **`mistral-small3.2:latest`** and **`qwen2.5:32b`** **tie** at **38.4%** ROUGE-L; Mistral Small 3.2 leads ROUGE-1/BLEU and embed on this run. **`qwen3.5:9b`** (with `reasoning_effort: none` in the Ollama path) is the best smaller Qwen 3.5 point (~30% ROUGE-L, strong embed). Values below come from `data/eval/runs/<run_id>/metrics.json` vs `silver_gpt4o_smoke_v1`.

### Full metrics table (vs silver reference)

Cloud APIs first (OpenAI baseline, then others by ROUGE-L), then local Ollama (by ROUGE-L).

| Run | Latency/ep | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Embed | Coverage | WER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_openai_smoke_v1 | 15.4s | 77.2% | 54.0% | 58.8% | 50.0% | 92.7% | 100.1% | 61.5% |
| llm_gemini_smoke_v1 | 2.7s | 61.3% | 23.4% | 33.3% | 16.3% | 87.3% | 78.1% | 83.8% |
| llm_mistral_smoke_v1 | 2.8s | 59.7% | 23.8% | 32.5% | 17.2% | 84.8% | 99.5% | 92.5% |
| llm_grok_smoke_v1 | 13.2s | 59.0% | 21.9% | 29.5% | 15.5% | 85.4% | 90.9% | 89.1% |
| llm_anthropic_smoke_v1 | 4.8s | 58.8% | 23.6% | 29.4% | 16.9% | 81.8% | 100.7% | 93.2% |
| llm_deepseek_smoke_v1 | 14.2s | 60.5% | 22.3% | 26.3% | 16.1% | 85.0% | 100.7% | 98.5% |
| llm_ollama_qwen25_32b_smoke_v1 | 54.8s | 65.6% | 30.8% | 38.4% | 22.0% | 85.2% | 78.4% | 81.3% |
| llm_ollama_mistral_small3_2_smoke_v1 | 48.6s | 69.4% | 33.0% | 38.4% | 27.8% | 85.8% | 91.4% | 83.2% |
| llm_ollama_mistral_7b_smoke_v1 | 17.4s | 61.4% | 24.8% | 32.8% | 18.9% | 80.4% | 97.0% | 92.2% |
| llm_ollama_mistral_nemo_12b_smoke_v1 | 22.8s | 59.2% | 23.4% | 30.7% | 16.0% | 82.5% | 112.9% | 102.9% |
| llm_ollama_qwen35_9b_smoke_v1 | 21.9s | 60.1% | 23.2% | 30.3% | 16.4% | 85.2% | 102.7% | 93.9% |
| llm_ollama_qwen35_27b_smoke_v1 | 81.5s | 61.2% | 24.8% | 30.2% | 17.2% | 82.4% | 128.5% | 113.1% |
| llm_ollama_qwen35_35b_smoke_v1 | 23.7s | 61.1% | 23.7% | 30.1% | 17.5% | 80.8% | 116.4% | 108.9% |
| llm_ollama_llama31_8b_smoke_v1 | 16.4s | 59.6% | 25.7% | 28.8% | 20.0% | 78.8% | 110.0% | 104.4% |
| llm_ollama_qwen25_7b_smoke_v1 | 12.1s | 63.0% | 24.1% | 28.3% | 17.7% | 84.9% | 95.6% | 95.3% |
| llm_ollama_gemma2_9b_smoke_v1 | 19.6s | 58.2% | 19.0% | 26.7% | 13.1% | 82.8% | 89.4% | 89.2% |
| llm_ollama_phi3_mini_smoke_v1 | 17.2s | 53.0% | 17.2% | 22.6% | 9.6% | 86.9% | 189.1% | 172.9% |

### Cloud LLMs (ranked by ROUGE-L vs silver)

| Rank | Provider | Model (eval config) | ROUGE-L | Embed | Latency | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **OpenAI** | GPT-4o | **58.8%** | **92.7%** | 15.4s | Same family as silver reference. |
| 2 | **Gemini** | gemini-2.0-flash | 33.3% | 87.3% | **2.7s** | Best non-OpenAI ROUGE-L + embed in this table. |
| 3 | Mistral | mistral-small-latest | 32.5% | 84.8% | 2.8s | Very fast; strong ROUGE-L. |
| 4 | Grok | grok-3-mini | 29.5% | 85.4% | 13.2s | Higher latency than Gemini/Mistral here. |
| 5 | Anthropic | claude-haiku-4-5 | 29.4% | 81.8% | 4.8s | Similar ROUGE-L to Grok; lowest embed in cloud set. |
| 6 | DeepSeek | (API default) | 26.3% | 85.0% | 14.2s | Lowest ROUGE-L among cloud rows; solid embed. |

### Local Ollama (vs silver)

Sorted by ROUGE-L (then run id). Hardware and Ollama builds vary; re-run on your machine for decisions.

| Run | ROUGE-L | Embed | Latency | Note |
| --- | --- | --- | --- | --- |
| llm_ollama_qwen25_32b_smoke_v1 | **38.4%** | 85.2% | 54.8s | Ties top ROUGE-L; `qwen2.5:32b`. |
| llm_ollama_mistral_small3_2_smoke_v1 | **38.4%** | **85.8%** | 48.6s | Ties top ROUGE-L; best ROUGE-1/BLEU/embed among the two; `mistral-small3.2:latest`. |
| llm_ollama_mistral_7b_smoke_v1 | 32.8% | 80.4% | 17.4s | Strong ROUGE-L for 7B; `mistral:7b`. |
| llm_ollama_mistral_nemo_12b_smoke_v1 | 30.7% | 82.5% | 22.8s | Good balance of speed vs quality; `mistral-nemo:12b`. |
| llm_ollama_qwen35_9b_smoke_v1 | 30.3% | 85.2% | 21.9s | Best Qwen 3.5 size/quality tradeoff here; use `reasoning_effort: none` for Qwen 3.5 on Ollama. |
| llm_ollama_qwen35_27b_smoke_v1 | 30.2% | 82.4% | 81.5s | Similar metrics to 9B/35B; slowest latency in this block. |
| llm_ollama_qwen35_35b_smoke_v1 | 30.1% | 80.8% | 23.7s | Similar ROUGE-L to other Qwen 3.5 smokes; faster wall time than 27B here. |
| llm_ollama_llama31_8b_smoke_v1 | 28.8% | 78.8% | 16.4s | `llama3.1:8b`. |
| llm_ollama_qwen25_7b_smoke_v1 | 28.3% | 84.9% | 12.1s | Fastest Ollama row; `qwen2.5:7b`. |
| llm_ollama_gemma2_9b_smoke_v1 | 26.7% | 82.8% | 19.6s | `gemma2:9b`. |
| llm_ollama_phi3_mini_smoke_v1 | 22.6% | 86.9% | 17.2s | High embed but very high coverage/WER vs reference; check output length. |

### Model IDs (acceptance-tested)

Use these model IDs in eval/acceptance configs to avoid API errors:

| Provider | Recommended model ID | Deprecated / not found |
| --- | --- | --- |
| Anthropic | `claude-haiku-4-5` | `claude-3-5-haiku-20241022` (404 deprecated) |
| Gemini | `gemini-2.0-flash` | — |
| Grok | `grok-3-mini` | `grok-2` (400 model not found) |

Eval configs: `data/eval/configs/llm_anthropic_smoke_v1.yaml`, `llm_gemini_smoke_v1.yaml`, `llm_grok_smoke_v1.yaml`, and siblings under `data/eval/configs/`. Acceptance configs: `config/acceptance/summarization/acceptance_planet_money_anthropic.yaml`, `acceptance_planet_money_gemini.yaml`, `acceptance_planet_money_grok.yaml`.

### How to re-run or add runs

From repo root, with the required API key set:

```bash
make experiment-run CONFIG=data/eval/configs/llm_<provider>_smoke_v1.yaml REFERENCE=silver_gpt4o_smoke_v1
```

To regenerate a multi-run report, use the eval multi-run report script (see [Experiment Guide](EXPERIMENT_GUIDE.md)).

---

## 🔮 Provider Magic Quadrant

A Gartner-style analysis plotting all 8 providers across two strategic dimensions:

- **X-Axis: Completeness of Vision** — Full-stack capabilities, context window, free tiers, innovation
- **Y-Axis: Ability to Execute** — Quality, speed, reliability, cost-effectiveness

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
             │   │  🌐Real │      │   ┌──────┴──────┐
             │   └─────────┘      │   │   Mistral   │ ← EU + Full stack
             │       ▲            │   │     🇪🇺      │
             │   Speed champion   │   └─────────────┘
             │                    │
      ───────┼────────────────────┼────────────────────────────────────►
             │                    │              COMPLETENESS OF VISION
             │                    │
             │    NICHE PLAYERS   │     VISIONARIES
             │                    │
             │   ┌───────────┐    │   ┌─────────────┐
             │   │  Ollama   │    │   │  Local ML   │ ← Zero cost + full stack
             │   │    🏠     │    │   │   (Default) │
             │   └───────────┘    │   └─────────────┘
             │       ▲            │          ▲
             │   Offline/private  │   Hardware required
             │   but needs HW     │   but complete control
             │                    │
             │   ┌───────────┐    │
             │   │ DeepSeek  │────┼───► Extreme value
             │   │   💰97%   │    │      but China-based
             │   └───────────┘    │
        Low  │                    │
             │                    │
             └────────────────────┴────────────────────────────────────
                    Limited                              Comprehensive
```yaml

### Quadrant Analysis

| Quadrant | Providers | Characteristics | Best For |
| -------- | --------- | --------------- | -------- |
| **🏆 Leaders** | OpenAI, Gemini, Mistral | Full capabilities, high quality, proven reliability | Production workloads, quality-critical apps |
| **💪 Challengers** | Anthropic, Grok | Excellent execution but limited scope | Text-only processing, real-time information |
| **🔭 Visionaries** | Local ML, DeepSeek | Innovative value proposition, some trade-offs | Cost optimization, privacy, experimentation |
| **🎯 Niche Players** | Ollama | Specialized use case, strong in specific domain | Offline, enterprise security, self-hosted |

### Provider Scores (0-10)

| Provider | Vision Score | Execution Score | Quadrant | Key Strength |
| -------- | :----------: | :-------------: | -------- | ------------ |
| **OpenAI** | 9 | 10 | Leader | Quality benchmark |
| **Gemini** | 10 | 8 | Leader | 2M context + free tier |
| **Mistral** | 8 | 7 | Leader | EU compliance + full stack |
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
                    ────────────────┼────────────────────────────────────►
                                    │                    Completeness of Vision

    Legend:  ● Current position    ═══▶ Predicted movement
```yaml

### Strategic Recommendations by Quadrant

**🏆 LEADERS (OpenAI, Gemini, Mistral)**

> *"Safe bets for production. Choose based on specific needs."*

| Provider | Choose When... |
| -------- | -------------- |
| OpenAI | Quality is paramount, budget available |
| Gemini | Need huge context (2M), want free tier |
| Mistral | EU data residency required |

**💪 CHALLENGERS (Anthropic, Grok)**

> *"Excellent at what they do, but not full-stack."*

| Provider | Choose When... |
| -------- | -------------- |
| Anthropic | Text quality matters, safety-first |
| Grok | Real-time information access needed |

**🔭 VISIONARIES (Local ML, DeepSeek)**

> *"Trade-offs for significant advantages."*

| Provider | Choose When... |
| -------- | -------------- |
| Local ML | Zero cost + privacy + offline |
| DeepSeek | Extreme budget constraints (97% savings) |

**🎯 NICHE PLAYERS (Ollama)**

> *"Perfect for specific use cases."*

| Provider | Choose When... |
| -------- | -------------- |
| Ollama | Enterprise security, air-gapped, unlimited processing |

---

## 📈 Visual Comparisons

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

💡 DeepSeek is 97% cheaper than OpenAI for text processing!
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

🌐 Grok provides real-time information access via X/Twitter integration!
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

🏆 OpenAI remains the quality leader, but alternatives close the gap!
```

### Privacy Level

```text
Privacy Scale (Higher = More Private)
═══════════════════════════════════════════════════════════════════════════

Local ML    🔒🔒🔒🔒🔒 │████████████████████████████████████████████████│ Complete
Ollama      🔒🔒🔒🔒🔒 │████████████████████████████████████████████████│ Complete
Mistral     🔒🔒🔒🔒   │███████████████████████████████████████         │ EU Servers
OpenAI      🔒🔒🔒     │██████████████████████████████                  │ US Servers
Anthropic   🔒🔒🔒     │██████████████████████████████                  │ US Servers
Google      🔒🔒🔒     │██████████████████████████████                  │ Google Cloud
Grok        🔒🔒🔒     │██████████████████████████████                  │ US Servers
DeepSeek    🔒🔒       │████████████████████                            │ China Servers
                      └────────────────────────────────────────────────────

🔒 For maximum privacy, use Local ML or Ollama - data never leaves your device!
```yaml

---

## 🎛️ Capability Matrix

```text
                    ┌─────────────────────────────────────────────────┐
                    │           CAPABILITY SUPPORT MATRIX              │
                    ├─────────────────────────────────────────────────┤
                    │  Provider      │ Status   │ 🎙️ Trans │ 👤 Speaker │ 📝 Summary │
                    ├────────────────┼──────────┼──────────┼────────────┼────────────┤
                    │  Local ML      │ ✅ Impl  │    ✅    │     ✅     │     ✅     │
                    │  OpenAI        │ ✅ Impl  │    ✅    │     ✅     │     ✅     │
                    ├────────────────┼──────────┼──────────┼────────────┼────────────┤
                    │  Mistral       │ ✅ Impl  │    ✅    │     ✅     │     ✅     │
                    │  Gemini        │ ✅ Impl  │    ✅    │     ✅     │     ✅     │
                    │  Anthropic     │ ✅ Impl  │    ❌    │     ✅     │     ✅     │
                    │  Grok          │ ✅ Impl  │    ❌    │     ✅     │     ✅     │
                    │  Ollama        │ ✅ Impl  │    ❌    │     ✅     │     ✅     │
                    │  DeepSeek      │ ✅ Impl  │    ❌    │     ✅     │     ✅     │
                    └─────────────────────────────────────────────────────────────────┘

    ✅ Implemented (8): Local ML, OpenAI, Gemini, Mistral, Anthropic, Grok, Ollama, DeepSeek
```yaml

---

## 💵 Detailed Cost Analysis

### Per 100 Episodes - Complete Breakdown

| Provider | Transcription | Speaker | Summary | **Total** | vs OpenAI |
| :------- | :-----------: | :-----: | :-----: | :-------: | :-------: |
| **Local ML** | $0 | $0 | $0 | **$0** | -100% |
| **Ollama** | ❌ | $0 | $0 | **$0** | -100% |
| **DeepSeek** | ❌ | $0.004 | $0.012 | **$0.016** | -97% |
| **Grok (beta)** | ❌ | $0.00 | $0.00 | **$0.00** | -100% |
| **Mistral (Small)** | ❌ | $0.03 | $0.08 | **$0.11** | -80% |
| **Anthropic (Haiku)** | ❌ | $0.10 | $0.30 | **$0.40** | -27% |
| **Gemini (Flash)** | $0.90 | $0.01 | $0.04 | **$0.95** | +73% |
| **OpenAI (Nano)** | $36.00 | $0.08 | $0.20 | **$36.28** | baseline |
| **OpenAI (Mini)** | $36.00 | $0.40 | $1.00 | **$37.40** | +3% |
| **Mistral (Full)** | $60.00 | $4.00 | $5.00 | **$69.00** | +90% |

### 📊 Cost Distribution Chart

```text
Where does the money go? (Full cloud processing)
═══════════════════════════════════════════════════════════════════════════

OpenAI ($37.40 total)
├── Transcription ███████████████████████████████████████████████░ 96% ($36)
├── Speaker Det.  ░                                                 1% ($0.40)
└── Summarization █░                                                3% ($1.00)

Mistral ($69 total)
├── Transcription ████████████████████████████████████████████░░░░ 87% ($60)
├── Speaker Det.  ██░                                               6% ($4)
└── Summarization ███░                                              7% ($5)

💡 INSIGHT: Transcription dominates cloud costs!
   Use local Whisper + cloud text processing to save 90%+
```

### 💰 Monthly Cost Projections

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
Mistral (full)      $69.00              $690.00             $6,900.00

⚠️  At 10,000 episodes/month, OpenAI full stack costs $3,740!
    Using local transcription + DeepSeek: $1.60 (99.96% savings)
```yaml

---

## 🔬 Provider Deep Dives

> **Note:** **Local ML**, **OpenAI**, **Gemini**, **Mistral**, **Anthropic**, **Grok**, and **Ollama** are fully implemented. **DeepSeek** is documented based on design specifications (RFC-034) and may change during implementation.

### 1. 🏠 Local ML Providers (Default) ✅ **Implemented**

```text
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL ML PROVIDERS                                             │
│  ═══════════════════                                            │
│                                                                 │
│  💰 Cost:     $0 (just electricity)                            │
│  ⚡ Speed:    Moderate (GPU dependent)                          │
│  🏆 Quality:  Good                                              │
│  🔒 Privacy:  ████████████████████ 100% (complete)              │
│                                                                 │
│  Components:                                                    │
│  ├── 🎙️ Transcription: OpenAI Whisper (local)                  │
│  ├── 👤 Speaker Det:   spaCy NER models                         │
│  └── 📝 Summarization: Hugging Face BART/LED                    │
│                                                                 │
│  Best For: Privacy, offline use, zero ongoing cost              │
└─────────────────────────────────────────────────────────────────┘
```yaml

> **📖 For detailed ML model comparisons, see [ML Model Comparison Guide](ML_MODEL_COMPARISON_GUIDE.md)**
>
> The ML Model Comparison Guide provides comprehensive analysis of:
> - Whisper model options (tiny, base, small, medium, large)
> - spaCy NER model options (sm, md, lg)
> - Transformers model options (BART, LED, PEGASUS, DistilBART)
> - Hardware requirements and recommendations
> - Performance benchmarks and quality comparisons

**Hardware Requirements:**

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| RAM | 8GB | 16GB+ |
| GPU VRAM | None (CPU) | 8GB+ |
| Storage | 5GB | 20GB |

---

### 2. 🤖 OpenAI ✅ **Implemented**

```text
┌─────────────────────────────────────────────────────────────────┐
│  OPENAI                                        Industry Leader  │
│  ══════                                                         │
│                                                                 │
│  💰 Cost:     $$$ (Premium pricing)                             │
│  ⚡ Speed:    Fast (100 tok/s)                                  │
│  🏆 Quality:  ████████████████████ Best                         │
│  🔒 Privacy:  ████████████░░░░░░░░ Standard (US)                │
│  📊 Tracking: Built-in token/audio usage metrics for cost eval  │
  ⚠️ Limit:    25MB audio file size limit for Whisper API        │
│                                                                 │
│  Models:                                                        │
│  ├── GPT-4o       $5.00/$15.00  │ Best quality                 │
│  ├── GPT-4o-mini  $0.15/$0.60   │ ⭐ Production recommended     │
│  └── Whisper      $0.006/min    │ Transcription                │
│                                                                 │
│  Best For: Quality-critical production, reliable workflows      │
└─────────────────────────────────────────────────────────────────┘
```yaml

---

### 3. 🧠 Anthropic (Claude) ✅ **Implemented** (RFC-032, Issue #106)

```text
┌─────────────────────────────────────────────────────────────────┐
│  ANTHROPIC CLAUDE                              Safety Focused   │
│  ═══════════════                                                │
│                                                                 │
│  💰 Cost:     $$ (Competitive)                                  │
│  ⚡ Speed:    Fast (100 tok/s)                                  │
│  🏆 Quality:  ███████████████████░ Excellent                    │
│  🔒 Privacy:  ████████████░░░░░░░░ Standard (US)                │
│  ⚠️  No transcription support                                   │
│                                                                 │
│  Models:                                                        │
│  ├── Claude Haiku 4.5   $1/$5   │ ⭐ Eval/acceptance alias `claude-haiku-4-5` │
│  ├── claude-haiku-4-5            │ Anthropic alias (current Haiku) │
│  └── claude-3-5-sonnet-20241022 │ Deprecated (404); use newer Sonnet IDs │
│                                                                 │
│  Best For: Quality text, nuanced content, safety alignment      │
└─────────────────────────────────────────────────────────────────┘
```yaml

---

### 4. 🇪🇺 Mistral ✅ **Implemented** (RFC-033, Issue #106)

```text
┌─────────────────────────────────────────────────────────────────┐
│  MISTRAL                                       European Leader  │
│  ═══════                                                        │
│                                                                 │
│  💰 Cost:     $-$$ (Competitive)                                │
│  ⚡ Speed:    Fast                                              │
│  🏆 Quality:  ██████████████████░░ Very Good                    │
│  🔒 Privacy:  ████████████████░░░░ High (EU servers)            │
│  ✅ FULL STACK - Only non-US alternative to OpenAI!            │
│                                                                 │
│  Models:                                                        │
│  ├── Large 3      $2/$6      │ ⭐ Production                   │
│  ├── Small 3.1    $0.10/$0.30│ ⭐ Dev/test (cheapest!)         │
│  └── Voxtral      ~$0.01/min │ Transcription                   │
│                                                                 │
│  Best For: EU compliance, full OpenAI alternative               │
└─────────────────────────────────────────────────────────────────┘
```yaml

---

### 5. 💎 DeepSeek 📋 **Planned** (RFC-034)

```text
┌─────────────────────────────────────────────────────────────────┐
│  DEEPSEEK                                      Ultra Low Cost   │
│  ════════                                                       │
│                                                                 │
│  💰 Cost:     $ (95% cheaper than OpenAI!)                      │
│  ⚡ Speed:    Fast (150 tok/s)                                  │
│  🏆 Quality:  ██████████████░░░░░░ Good                         │
│  🔒 Privacy:  ████████░░░░░░░░░░░░ China servers                │
│  ⚠️  No transcription support                                   │
│                                                                 │
│  Models:                                                        │
│  ├── DeepSeek Chat      $0.28/$0.42 (cache miss)               │
│  ├── DeepSeek Chat      $0.028/$0.42 (cache hit!) 💰           │
│  └── DeepSeek Reasoner  Complex reasoning tasks                │
│                                                                 │
│  🔥 $0.016/100 episodes vs $0.55 OpenAI = 97% SAVINGS          │
│                                                                 │
│  Best For: Budget optimization, bulk processing, startups       │
└─────────────────────────────────────────────────────────────────┘
```yaml

---

### 6. 🌈 Google Gemini 📋 **Planned** (RFC-035)

```text
┌─────────────────────────────────────────────────────────────────┐
│  GOOGLE GEMINI                                 Massive Context  │
│  ═════════════                                                  │
│                                                                 │
│  💰 Cost:     $ (Generous free tier!)                           │
│  ⚡ Speed:    Fast                                              │
│  🏆 Quality:  ██████████████████░░ Very Good                    │
│  🔒 Privacy:  ████████████░░░░░░░░ Standard (Google)            │
│  ✅ FULL STACK with native audio understanding                  │
│                                                                 │
│  Models:                                                        │
│  ├── Gemini 2.0 Flash  $0.10/$0.40  │ ⭐ Dev/test              │
│  ├── Gemini 1.5 Pro    $1.25/$5.00  │ ⭐ Production            │
│  └── Gemini 1.5 Flash  $0.075/$0.30 │ Budget                   │
│                                                                 │
│  🔥 2 MILLION TOKEN CONTEXT - Process entire seasons!          │
│                                                                 │
│  FREE TIER: 15 RPM, 1M TPM, 1500 RPD                           │
│                                                                 │
│  Best For: Long content, free development, multimodal           │
└─────────────────────────────────────────────────────────────────┘
```yaml

---

### 7. 🌐 Grok ✅ **Implemented** (RFC-036, Issue #1095)

```text
┌─────────────────────────────────────────────────────────────────┐
│  GROK                                          Real-Time Info   │
│  ════                                                           │
│                                                                 │
│  💰 Cost:     $ (Affordable)                                    │
│  🌐 Feature:  ████████████████████ Real-time X/Twitter access  │
│  🏆 Quality:  ██████████████░░░░░░ Good (xAI models)           │
│  🔒 Privacy:  ████████████░░░░░░░░ Standard (US)                │
│  ⚠️  No transcription support                                   │
│                                                                 │
│  Models (xAI's Grok):                                           │
│  ├── grok-3-mini                │ ⭐ Eval/acceptance (use this ID) │
│  └── grok-2                     │ API returns 400 model not found │
│                                                                 │
│  🌐 Access real-time information via X/Twitter integration!   │
│                                                                 │
│  FREE TIER: grok-beta available                                │
│                                                                 │
│  Best For: Real-time information, current events, X/Twitter data│
└─────────────────────────────────────────────────────────────────┘
```yaml

---

### 8. 🏠 Ollama (Local LLMs) ✅ **Implemented** (RFC-037, Issue #196)

```text
┌─────────────────────────────────────────────────────────────────┐
│  OLLAMA                                        Self-Hosted      │
│  ══════                                                         │
│                                                                 │
│  💰 Cost:     $0 per request (hardware only)                    │
│  ⚡ Speed:    Slow-Medium (hardware dependent, ~30 tok/s)        │
│  🏆 Quality:  ██████████████░░░░░░ Good (model dependent)       │
│  🔒 Privacy:  ████████████████████ 100% Complete                │
│  ⚠️  No transcription support                                   │
│                                                                 │
│  Recommended Models (by Use Case):                             │
│  ├── qwen2.5:7b        8GB+ RAM  │ ⭐ Best JSON, GIL extraction │
│  ├── llama3.1:8b       8GB+ RAM  │ ⭐ General purpose (default)│
│  ├── mistral:7b        8GB+ RAM  │ ⭐ Fastest inference        │
│  ├── gemma2:9b         12GB+ RAM │ ⭐ Balanced quality/speed   │
│  └── phi3:mini         4GB+ RAM  │ ⭐ Dev/test, lightweight   │
│                                                                 │
│  Setup Requirements:                                           │
│  ├── Install Ollama:   brew install ollama (macOS)            │
│  ├── Start server:     ollama serve (keep running)            │
│  └── Pull models:      ollama pull qwen2.5:7b (recommended)    │
│                                                                 │
│  Hardware Recommendations:                                      │
│  ├── 4GB+ RAM:         phi3:mini (dev/test only)             │
│  ├── 8GB+ RAM:         qwen2.5:7b, llama3.1:8b, mistral:7b    │
│  └── 12GB+ RAM:        gemma2:9b (balanced quality/speed)     │
│                                                                 │
│  💡 Zero API costs, unlimited usage, complete data privacy      │
│                                                                 │
│  Best For: Privacy-critical, offline/air-gapped, unlimited    │
│            processing, enterprises, cost optimization           │
│                                                                 │
│  📖 See [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) for │
│     detailed installation, setup, and troubleshooting          │
└─────────────────────────────────────────────────────────────────┘
```yaml

---

## 🗺️ Decision Flowchart

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
```yaml

---

## 🎨 Recommended Configurations

### Configuration 1: 💰 Ultra-Budget ($0.016/100 episodes)

```yaml

# 97% cheaper than OpenAI

transcription_provider: whisper       # Free (local)
speaker_detector_provider: deepseek   # $0.004/100
summary_provider: deepseek            # $0.012/100

# Model settings

deepseek_api_key: ${DEEPSEEK_API_KEY}
```bash

**Savings:** $37.38 per 100 episodes vs OpenAI

---

## Configuration 2: 🏆 Quality-First (~$42/100 episodes)

```yaml

# Maximum quality

transcription_provider: openai
speaker_detector_provider: openai
summary_provider: openai

# Model settings

openai_speaker_model: gpt-5
openai_summary_model: gpt-5
openai_api_key: ${OPENAI_API_KEY}
```yaml

---

## Configuration 3: 🔒 Privacy-First ($0)

```yaml

# Data never leaves your device

transcription_provider: whisper       # Local
speaker_detector_provider: ollama     # Local Ollama (or spacy for NER)
summary_provider: ollama              # Local Ollama (or transformers for BART/LED)

# Ollama configuration
ollama_speaker_model: llama3.1:8b    # For 6-8GB RAM systems
ollama_summary_model: llama3.1:8b    # For 6-8GB RAM systems
# OR for better quality (12-16GB RAM):
# ollama_speaker_model: llama3.3:latest
# ollama_summary_model: llama3.3:latest
```yaml

---

## Configuration 4: ⚡ Speed-First (~$0.25/100 episodes)

```yaml

# 10x faster processing

transcription_provider: whisper       # Local
speaker_detector_provider: grok
summary_provider: grok

# Model settings

grok_speaker_model: grok-2
grok_summary_model: grok-2
grok_api_key: ${GROK_API_KEY}
```yaml

---

## Configuration 5: 🇪🇺 EU Compliant (~$65/100 episodes)

```yaml

# European data residency

transcription_provider: mistral
speaker_detector_provider: mistral
summary_provider: mistral

# Model settings

mistral_speaker_model: mistral-large-latest
mistral_summary_model: mistral-large-latest
mistral_api_key: ${MISTRAL_API_KEY}
```yaml

---

## Configuration 6: 🆓 Free Development (~$0)

```yaml

# Maximize free tiers

transcription_provider: whisper       # Local
speaker_detector_provider: gemini     # Free tier
summary_provider: grok                # Free tier

gemini_speaker_model: gemini-2.0-flash
grok_summary_model: grok-beta
```yaml

---

## 📊 Summary Statistics

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
│  🥇 EU COMPLIANT:        Mistral          Only European full-stack         │
│  🥇 COMPLETE PRIVACY:    Local/Ollama     Data never leaves device         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📈 COST INSIGHT:                                                          │
│     Transcription = 90%+ of cloud costs                                    │
│     → Use local Whisper + cloud text = massive savings                     │
│                                                                             │
│  🌐 REAL-TIME INSIGHT:                                                      │
│     Grok provides real-time information via X/Twitter integration         │
│     → Access current events and trending topics                            │
│                                                                             │
│  🔒 PRIVACY INSIGHT:                                                        │
│     Only Local ML and Ollama guarantee 100% privacy                        │
│     → All cloud providers process data on their servers                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```yaml

---

## 📚 Related Documentation

- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md)
- [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) - Complete Ollama setup, installation, troubleshooting
- [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)
- [PRD-006: OpenAI Provider](../prd/PRD-006-openai-provider-integration.md)
- [PRD-009: Anthropic Provider](../prd/PRD-009-anthropic-provider-integration.md)
- [PRD-010: Mistral Provider](../prd/PRD-010-mistral-provider-integration.md)
- [PRD-011: DeepSeek Provider](../prd/PRD-011-deepseek-provider-integration.md)
- [PRD-012: Gemini Provider](../prd/PRD-012-gemini-provider-integration.md)
- [PRD-013: Grok Provider (xAI)](../prd/PRD-013-grok-provider-integration.md)
- [PRD-014: Ollama Provider](../prd/PRD-014-ollama-provider-integration.md)
