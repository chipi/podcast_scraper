# ML Model Comparison Guide

> **Your complete decision-making resource for choosing the right ML models**

A comprehensive analysis of all ML models supported by podcast_scraper's local ML provider to help you choose the right models based on quality, speed, memory requirements, and hardware capabilities.

---

## ⚠️ Implementation Status

**Important:** This guide includes both **implemented** and **planned** models.
Check the status below before making decisions.

### ✅ Implemented Models (v2.4.0)

| Category | Models | Status | Notes |
| :------- | :----: | :----: | :---- |
| **Whisper** | 6 models | ✅ **Implemented** | All sizes from tiny to large |
| **spaCy** | 3 models | ✅ **Implemented** | sm, md, lg variants |
| **Transformers (Classic)** | 6 models | ✅ **Implemented** | BART, LED, PEGASUS, DistilBART |

### 📋 Planned Models (RFC-042, v2.5)

The following models are **designed but not yet implemented** for the Hybrid MAP-REDUCE provider.
They are documented here for planning purposes and future reference.

| Category | Models | Status | RFC | Implementation Status |
| :------- | :----: | :----: | :--: | :---------------------: |
| **LongT5 (MAP)** | 2 models | 📋 **Planned** | RFC-042 | Design complete, not implemented |
| **Instruction LLMs (REDUCE)** | 5 models | 📋 **Planned** | RFC-042 | Design complete, not implemented |

**Note:** Hybrid MAP-REDUCE models are documented based on design specifications (RFC-042) and may change during implementation.

---

## 📊 Key Statistics at a Glance

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ML MODEL LANDSCAPE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  3 Model Categories │  20+ Model Options    │  Hardware-Aware Selection    │
│  ════════════════   │  ═══════════════════  │  ═══════════════════════     │
│  🎙️ Whisper (6)     │  Transcription        │  Auto-detects MPS/CUDA/CPU   │
│  👤 spaCy (3)       │  Speaker Detection   │  Optimized per device        │
│  📝 Classic (6)     │  Summarization        │  Memory-aware loading        │
│  🔄 Hybrid (7)      │  Summarization        │  Planned for v2.5            │
├─────────────────────────────────────────────────────────────────────────────┤
│                           MEMORY SPECTRUM (Model Sizes)                      │
│                                                                             │
│  ~100MB ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ~14GB │
│  │                                                                     │   │
│  ▼                                                                     ▼   │
│  tiny.en                                                      Qwen2.5-14B │
│  (39MB)                                                           (14GB)    │
│                                                                             │
│  base.en ─── small.en ─── medium.en ─── large ─── BART-large ─── LED-large │
│  (74MB)    (244MB)      (769MB)      (1.5GB)      (2GB)         (2.5GB)     │
│                                                                             │
│  ─── Qwen2.5-7B ─── LLaMA-3-8B ─── Mistral-7B ─── Qwen2.5-14B             │
│  (7GB)      (8GB)        (7GB)         (14GB)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Decision Matrix

| If you need... | Choose | Why |
| :------------- | :----: | :-- |
| 🚀 **Fastest Transcription** | `tiny.en` | Smallest model, fastest inference |
| 🏆 **Best Transcription Quality** | `large` or `large-v3` | Highest accuracy, handles accents/noise |
| ⚖️ **Quality/Speed Balance** | `base.en` | ⭐ **Production default** - good quality, reasonable speed |
| 💾 **Lowest Memory** | `tiny.en` + `bart-small` | Minimal RAM/VRAM requirements |
| 🎯 **Best Summarization** | `bart-large` + `led-large-16384` | ⭐ **Production default** - highest quality |
| ⚡ **Fast Summarization** | `bart-small` + `led-base-16384` | Faster inference, lower memory |
| 📏 **Long Transcripts** | `led-large-16384` | 16k token context, no chunking needed |
| 🔍 **Best Speaker Detection** | `en_core_web_lg` | Largest vocabulary, best accuracy |
| ⚖️ **Balanced Speaker Detection** | `en_core_web_sm` | ⭐ **Default** - good accuracy, fast |

---

## 🎙️ Transcription Models (Whisper)

The podcast scraper uses **OpenAI Whisper** for local transcription. Whisper models come in multiple sizes, with English-only (`.en`) variants that are faster and more accurate for English content.

### Model Comparison

| Model | Size | VRAM | Speed | Quality | Best For |
| :---- | :--: | :--: | :---: | :-----: | :------- |
| **tiny.en** | 39MB | ~1GB | ⚡⚡⚡ Fastest | ⭐⭐ Good | Testing, quick iterations |
| **base.en** | 74MB | ~1GB | ⚡⚡ Fast | ⭐⭐⭐ Very Good | ⭐ **Production default** |
| **small.en** | 244MB | ~2GB | ⚡ Moderate | ⭐⭐⭐⭐ Excellent | High-quality production |
| **medium.en** | 769MB | ~5GB | 🐌 Slow | ⭐⭐⭐⭐⭐ Best | Maximum accuracy |
| **large** | 1.5GB | ~10GB | 🐌🐌 Slowest | ⭐⭐⭐⭐⭐ Best | Multilingual, best quality |
| **large-v3** | 1.5GB | ~10GB | 🐌🐌 Slowest | ⭐⭐⭐⭐⭐ Best | Latest, best accuracy |

### Key Characteristics

**English-Only Models (`.en` suffix):**

- **Faster**: Optimized for English, no multilingual overhead
- **More Accurate**: Trained specifically on English content
- **Smaller**: Reduced vocabulary size
- **Recommended**: Use `.en` variants for English podcasts

**Multilingual Models (no suffix):**

- **Universal**: Supports 99+ languages
- **Larger**: Full vocabulary for all languages
- **Slower**: More processing overhead
- **Use When**: Processing non-English content

### Performance by Hardware

```text
Transcription Speed (Relative to base.en on CPU)
═══════════════════════════════════════════════════════════════════════════

Apple Silicon (MPS)
├── tiny.en      │████████████████████████████████████████████████████ 10x
├── base.en      │████████████████████████████████████████████████     8x
├── small.en     │████████████████████████████████████                 5x
└── large        │████████████████                                     2x

NVIDIA CUDA
├── tiny.en      │████████████████████████████████████████████████████ 12x
├── base.en      │████████████████████████████████████████████████     10x
├── small.en     │████████████████████████████████████                 6x
└── large        │████████████████                                     3x

CPU (FP32)
├── tiny.en      │████████████████████████████████████████████████████ 1x
├── base.en      │████████████████████████████████████████████████     0.8x
├── small.en     │████████████████████████████████████                 0.5x
└── large        │████████████████                                     0.2x
```

### Quality vs. Speed Trade-off

```text
Quality Score (1-10) vs Speed (relative)
═══════════════════════════════════════════════════════════════════════════

large-v3        │██████████████████████████████████████████████████│ 10
                │                                                  │
large           │██████████████████████████████████████████████    │  9
                │                                                  │
medium.en       │████████████████████████████████████████          │  8
                │                                                  │
small.en        │██████████████████████████████████                │  7
                │                                                  │
base.en ⭐      │███████████████████████████████                   │  6
                │                                                  │
tiny.en         │███████████████████████                            │  5
                └────────────────────────────────────────────────────────
                0                   5                              10
                Slowest                                          Fastest
```

### Recommended Whisper Configurations

**Development/Testing:**

```yaml
whisper_model: tiny.en  # Fastest, good enough for testing
```

**Production (Balanced):**

```yaml
whisper_model: base.en  # ⭐ Default - best quality/speed balance
```

**Production (High Quality):**

```yaml
whisper_model: small.en  # Better accuracy, still reasonable speed
```

**Maximum Quality:**

```yaml
whisper_model: large-v3  # Best accuracy, requires powerful hardware
```

---

## 👤 Speaker Detection Models (spaCy NER)

The system uses **spaCy Named Entity Recognition (NER)** models to identify hosts and guests from episode metadata. spaCy models come in three sizes, trading vocabulary size and accuracy for speed and memory.

### spaCy Model Comparison

| Model | Size | Memory | Speed | Accuracy | Best For |
| :---- | :--: | :----: | :---: | :------: | :------- |
| **en_core_web_sm** | 12MB | ~50MB | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | ⭐ **Default** - balanced |
| **en_core_web_md** | 40MB | ~200MB | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | Better accuracy needed |
| **en_core_web_lg** | 560MB | ~500MB | ⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Maximum accuracy |

### spaCy Model Characteristics

**Small Model (`en_core_web_sm`):**

- **Fastest**: Minimal processing overhead
- **Good Accuracy**: Handles common names well
- **Low Memory**: Suitable for resource-constrained environments
- **Recommended**: Default choice for most use cases

**Medium Model (`en_core_web_md`):**

- **Better Vocabulary**: Includes word vectors (300k unique vectors)
- **Improved Accuracy**: Better at identifying less common names
- **Moderate Memory**: Good balance for production

**Large Model (`en_core_web_lg`):**

- **Best Vocabulary**: Includes word vectors (685k unique vectors)
- **Highest Accuracy**: Best at identifying names in noisy descriptions
- **High Memory**: Requires more RAM but provides best results

### Accuracy Comparison

```text
Speaker Detection Accuracy (on test dataset)
═══════════════════════════════════════════════════════════════════════════

en_core_web_lg  │████████████████████████████████████████████████████│ 95%
en_core_web_md  │██████████████████████████████████████████████      │ 88%
en_core_web_sm  │████████████████████████████████████████            │ 82%
                └────────────────────────────────────────────────────────
                0                   50                              100%
```

### When to Upgrade spaCy Models

**Stick with `en_core_web_sm` (default) if:**

- Processing common English names
- Memory is constrained
- Speed is important
- Most episodes have clear host/guest mentions

**Upgrade to `en_core_web_md` if:**

- Missing some less common names
- Processing international names
- Have moderate memory available

**Upgrade to `en_core_web_lg` if:**

- Maximum accuracy is critical
- Processing very noisy metadata
- Names are frequently missed
- Memory is not a concern

---

## 📝 Summarization Models (Transformers)

The system uses **Hugging Face Transformers** for summarization, with a **MAP-REDUCE** architecture to handle long transcripts. Models are categorized into **MAP models** (for chunk summarization) and **REDUCE models** (for final synthesis).

### Content Size Suitability

Different models excel at different content lengths and styles:

| Content Type | Length | Best Models | Why |
| :----------- | :----: | :---------: | :-- |
| **Short News-Style** | < 1k tokens | BART, DistilBART | Trained on CNN/DailyMail, excel at concise news |
| **Medium Conversational** | 1k-4k tokens | BART-large, PEGASUS | Good balance, handle dialogue well |
| **Long Abstract** | 4k-16k tokens | LED, LongT5 | Large context windows, preserve detail |
| **Very Long Structured** | 16k+ tokens | LED-large, Hybrid LLMs | Maximum context, instruction-following |

---

## 📝 Classic Summarization Models (Current Implementation) ✅

### MAP Models (Chunk Summarization)

MAP models summarize individual transcript chunks. They have limited context windows (typically 1024 tokens) and are optimized for compression.

| Model | Alias | Size | Memory | Context | Speed | Quality | Content Fit | Best For |
| :---- | :---: | :--: | :----: | :-----: | :---: | :-----: | :---------: | :------- |
| **BART-base** | `bart-small` | 500MB | ~1GB | 1024 | ⚡⚡ Fast | ⭐⭐⭐ Good | Short-Medium | Testing, dev |
| **BART-large-cnn** | `bart-large` | 2GB | ~3GB | 1024 | ⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Short-Medium | ⭐ **Production** |
| **DistilBART** | `fast` | 300MB | ~500MB | 1024 | ⚡⚡⚡ Fastest | ⭐⭐ Fair | Short | Speed-critical |
| **PEGASUS-large** | `pegasus` | 2.5GB | ~3GB | 1024 | ⚡ Moderate | ⭐⭐⭐⭐ Very Good | Short-Medium | Alternative to BART |
| **PEGASUS-xsum** | `pegasus-xsum` | 2.5GB | ~3GB | 1024 | ⚡ Moderate | ⭐⭐⭐⭐ Very Good | Short | Short summaries |

**Content Size Characteristics:**

- **BART Models**: Best for **short to medium** content (news-style, concise). Trained on CNN/DailyMail, excel at extracting key facts from structured prose.
- **PEGASUS Models**: Similar to BART, optimized for **short summaries**. `pegasus-xsum` specifically targets very concise outputs.
- **DistilBART**: Fastest but lower quality, suitable for **short content** where speed matters more than quality.

### REDUCE Models (Final Synthesis)

REDUCE models combine chunk summaries into final output. LED models support 16k token context, eliminating the need for chunking in the reduce phase.

| Model | Alias | Size | Memory | Context | Speed | Quality | Content Fit | Best For |
| :---- | :---: | :--: | :----: | :-----: | :---: | :-----: | :---------: | :------- |
| **LED-base-16384** | `long-fast` | 1GB | ~2GB | 16,384 | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | Long | Faster processing |
| **LED-large-16384** | `long` | 2.5GB | ~4GB | 16,384 | ⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Very Long | ⭐ **Production** |

**Content Size Characteristics:**

- **LED Models**: Designed for **long documents** (up to 16k tokens). Preserve more detail than BART (65-75% compression is normal). Best for **abstract, long-form content** where detail retention matters.

### Classic Model Characteristics

**BART Models:**

- **Architecture**: Encoder-decoder, trained on CNN/DailyMail
- **Content Fit**: **Short to medium** (news-style, structured prose)
- **Strength**: Excellent at abstractive summarization of concise content
- **Weakness**: 1024 token limit requires chunking, struggles with very long/abstract content
- **Use Case**: Standard summarization tasks, news-style transcripts

**LED Models:**

- **Architecture**: Longformer-based, designed for long documents
- **Content Fit**: **Long to very long** (abstract, detailed content)
- **Strength**: 16k token context, no chunking needed, preserves detail
- **Weakness**: Preserves more detail (lower compression ratios) - this is by design
- **Use Case**: Long transcripts, reduce phase synthesis, abstract content
- **Note**: LED models preserve 65-75% compression (normal), not a failure (see [Issue #283](https://github.com/chipi/podcast_scraper/issues/283))

**PEGASUS Models:**

- **Architecture**: Pre-trained specifically for summarization
- **Content Fit**: **Short** (optimized for concise summaries)
- **Strength**: Trained on summarization tasks, good for short outputs
- **Weakness**: Similar to BART, 1024 token limit
- **Use Case**: Alternative to BART for different style, very short summaries

**DistilBART:**

- **Architecture**: Distilled version of BART
- **Content Fit**: **Short** (fast processing of concise content)
- **Strength**: Fastest, lowest memory
- **Weakness**: Lower quality than full BART
- **Use Case**: Resource-constrained environments, speed-critical short content

### MAP-REDUCE Strategy

The system uses different strategies based on combined chunk summary length:

| Strategy | Token Range (BART) | Token Range (LED) | Approach |
| :------- | :----------------: | :---------------: | :------- |
| **Direct** | < 1024 | < 16,384 | Single-pass summarization |
| **Single-Pass Reduce** | < 800 | < 800 | Direct reduce of all chunks |
| **Hierarchical Reduce** | 800 - 3,500 | 800 - 5,500 | Recursive chunking and summarizing |
| **Transition Zone** | 3,500 - 4,500 | 5,500 - 6,500 | Smooth transition to extractive |
| **Extractive Fallback** | > 4,500 | > 6,500 | Select representative chunks |

**Key Insight from Issue #283:**

- LED models have **higher token ceilings** (6k vs 4k for BART) because they preserve more detail by design
- LED compression ratios of 65-75% are **normal**, not a failure
- Model-specific thresholds prevent false warnings and optimize quality

### Quality Comparison

```text
Summarization Quality (Subjective, based on test evaluations)
═══════════════════════════════════════════════════════════════════════════

BART-large + LED-large  │████████████████████████████████████████████████│ 10
BART-large + LED-base    │████████████████████████████████████████████    │  9
BART-base + LED-large    │█████████████████████████████████████████       │  8
PEGASUS + LED-large      │██████████████████████████████████████          │  7
BART-base + LED-base     │██████████████████████████████████              │  6
DistilBART + LED-base    │███████████████████████                         │  4
                        └────────────────────────────────────────────────────────
                        0                   5                              10
```

### Performance Characteristics

**Direct Summarization** (< 1024 tokens):

- **Time**: < 5 seconds
- **Memory**: Model size + ~500MB overhead
- **Quality**: Excellent (no chunking artifacts)

**MAP-REDUCE** (long transcripts):

- **MAP Phase**: ~3 seconds per chunk (varies by hardware)
- **REDUCE Phase**: ~5-10 seconds (depends on combined length)
- **Memory**: Model size + chunk buffers
- **Quality**: Very good (minimal information loss)

**Hardware Impact:**

- **Apple Silicon (MPS)**: Excellent performance, unified memory allows larger models
- **NVIDIA CUDA**: Best throughput, can run multiple models in parallel
- **CPU**: Slower but functional, uses ThreadPoolExecutor for parallel chunks

### Recommended Transformers Configurations

**Development/Testing:**

```yaml
summary_model: bart-small        # Fast, lower memory
summary_reduce_model: long-fast  # Faster reduce phase
```

**Production (Balanced):**

```yaml
summary_model: bart-large        # ⭐ Best quality MAP
summary_reduce_model: long       # ⭐ Best quality REDUCE
```

**Production (Speed-Optimized):**

```yaml
summary_model: fast              # DistilBART - fastest
summary_reduce_model: long-fast  # LED-base - faster
```

**Production (Quality-Optimized):**

```yaml
summary_model: bart-large        # Best MAP quality
summary_reduce_model: long       # Best REDUCE quality (handles 16k tokens)
```

---

## 🔄 Hybrid MAP-REDUCE Models (Planned for v2.5) 📋

> **Status**: These models are **planned** for RFC-042 implementation in v2.5.
> They are documented here for planning purposes and future reference.

The Hybrid MAP-REDUCE architecture separates **compression** (MAP phase) from **reasoning and structuring** (REDUCE phase), using each model class for what it does best. This addresses quality issues in classic-only summarization by leveraging instruction-tuned LLMs for abstraction and structure adherence.

### Hybrid MAP Models (Compression Phase)

MAP models compress transcript chunks into factual notes. Classic summarizers excel at this compression task.

| Model | Status | Size | Memory | Context | Speed | Content Fit | Best For |
| :---- | :----: | :--: | :----: | :-----: | :---: | :---------: | :------- |
| **LED-base-16384** | ✅ Implemented | 1GB | ~2GB | 16,384 | ⚡⚡ Fast | Long | Fast compression |
| **LED-large-16384** | ✅ Implemented | 2.5GB | ~4GB | 16,384 | ⚡ Moderate | Very Long | ⭐ Best quality |
| **LongT5-base** | 📋 Planned | ~1GB | ~2GB | 8,192 | ⚡⚡ Fast | Medium-Long | Alternative to LED |
| **LongT5-large** | 📋 Planned | ~2.5GB | ~4GB | 8,192 | ⚡ Moderate | Long | Higher quality LongT5 |
| **PEGASUS-large** | ✅ Implemented | 2.5GB | ~3GB | 1,024 | ⚡ Moderate | Short | Shorter contexts |

**Content Size Characteristics:**

- **LED Models**: Best for **long to very long** content (4k-16k tokens). Preserve detail, designed for abstract content.
- **LongT5 Models** (Planned): Similar to LED, optimized for **medium to long** content (2k-8k tokens). Good alternative to LED with different architecture.
- **PEGASUS**: Suitable for **shorter** contexts (up to 1k tokens) in hybrid pipeline.

### Hybrid REDUCE Models (Abstraction & Structuring Phase)

REDUCE models synthesize compressed notes into structured summaries. Instruction-tuned LLMs excel at abstraction, deduplication, and structure adherence.

| Model | Status | Size | Memory | Context | Speed | Quality | Content Fit | Best For |
| :---- | :----: | :--: | :----: | :-----: | :---: | :-----: | :---------: | :------- |
| **Qwen2.5-7B-Instruct** | 📋 Planned | 7GB | ~8GB (4-bit) | 32k | ⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Any (compressed) | ⭐ **Mac default** |
| **Qwen2.5-14B-Instruct** | 📋 Planned | 14GB | ~16GB (4-bit) | 32k | 🐌 Slow | ⭐⭐⭐⭐⭐ Best | Any (compressed) | Maximum quality |
| **LLaMA-3-8B-Instruct** | 📋 Planned | 8GB | ~9GB (4-bit) | 8k | ⚡ Moderate | ⭐⭐⭐⭐ Very Good | Medium | Strong reasoning |
| **Mistral-7B-Instruct** | 📋 Planned | 7GB | ~8GB (4-bit) | 8k | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | Medium | Speed/quality balance |
| **Phi-3-Mini-Instruct** | 📋 Planned | 3.8GB | ~4GB (4-bit) | 4k | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | Short | Resource-constrained |

**Content Size Characteristics:**

- **Qwen2.5 Models**: Excellent instruction-following, **32k context** handles any compressed input. Best for **structured, abstract summaries** from any content length.
- **LLaMA-3-8B**: Strong reasoning, **8k context** suitable for **medium-length** compressed notes. Good abstraction quality.
- **Mistral-7B**: Fast and efficient, **8k context** for **medium-length** content. Good balance of speed and quality.
- **Phi-3-Mini**: Lightweight, **4k context** for **shorter** compressed notes. Best for resource-constrained environments.

**Key Advantages of Instruction-Tuned LLMs:**

- ✅ **Better abstraction** - True synthesis, not extractive stitching
- ✅ **Structure adherence** - Reliably follows output format (takeaways, outline, actions)
- ✅ **Content filtering** - Ignores ads, intros, outros, meta-text
- ✅ **Deduplication** - Merges duplicate ideas across chunks
- ✅ **No scaffold leakage** - Doesn't echo schema or instruction text

### Hybrid Architecture Benefits

**Separation of Concerns:**

- **MAP Phase**: Classic summarizers compress chunks efficiently (what they're good at)
- **REDUCE Phase**: Instruction LLMs abstract and structure (what they're good at)

**Content Size Handling:**

| Content Length | MAP Model | REDUCE Model | Why |
| :------------- | :-------- | :----------- | :-- |
| **Short** (< 5k tokens) | LED-base, PEGASUS | Phi-3-Mini, Mistral-7B | Fast processing, sufficient context |
| **Medium** (5k-15k tokens) | LED-base, LongT5-base | Qwen2.5-7B, LLaMA-3-8B | Balanced quality and speed |
| **Long** (15k-30k tokens) | LED-large, LongT5-large | Qwen2.5-7B, Qwen2.5-14B | Maximum context, best quality |
| **Very Long** (30k+ tokens) | LED-large | Qwen2.5-14B | Largest context windows |

### Hybrid Model Selection Matrix

| Hardware | MAP Model | REDUCE Model | Backend | Quantization |
| :-------- | :-------- | :----------- | :------ | :----------- |
| **CPU only** | LED-base, LongT5-base | Phi-3-Mini | llama.cpp | 4-bit |
| **Apple Silicon** | LED-base (CPU) | Qwen2.5-7B, Mistral-7B | llama.cpp + Metal | 4-bit |
| **NVIDIA 8-12GB** | LED-base, LongT5-base | Mistral-7B | transformers | 4-bit |
| **NVIDIA 16GB+** | LED-large, LongT5-large | Qwen2.5-7B/14B | transformers | 4-bit |

**Recommended Default (Mac Laptop):**

- **MAP**: `led-base` on CPU
- **REDUCE**: `qwen2.5-7b` via llama.cpp with Metal, 4-bit quantization

### Content Size Suitability for Hybrid Models

**Short Content (< 5k tokens):**

- **MAP**: PEGASUS, LED-base (fast compression)
- **REDUCE**: Phi-3-Mini, Mistral-7B (sufficient context, fast)
- **Best For**: Quick summaries, testing, resource-constrained

**Medium Content (5k-15k tokens):**

- **MAP**: LED-base, LongT5-base (balanced compression)
- **REDUCE**: Qwen2.5-7B, LLaMA-3-8B, Mistral-7B (good abstraction)
- **Best For**: Most podcast episodes, balanced quality/speed

**Long Content (15k-30k tokens):**

- **MAP**: LED-large, LongT5-large (high-quality compression)
- **REDUCE**: Qwen2.5-7B, Qwen2.5-14B (large context, best abstraction)
- **Best For**: Long-form podcasts, detailed summaries

**Very Long Content (30k+ tokens):**

- **MAP**: LED-large (maximum context compression)
- **REDUCE**: Qwen2.5-14B (32k context, best quality)
- **Best For**: Extended interviews, multi-hour episodes

---

## 🖥️ Hardware Requirements & Recommendations

### Minimum Requirements

| Component | Minimum | Recommended | High-End |
| :-------- | :------ | :---------- | :------- |
| **RAM** | 8GB | 16GB | 32GB+ |
| **GPU VRAM** | None (CPU) | 8GB | 16GB+ |
| **Storage** | 5GB | 20GB | 50GB+ |
| **CPU** | 4 cores | 8 cores | 16+ cores |

### Hardware-Specific Recommendations

**Apple Silicon (M1/M2/M3/M4):**

- **Best Models**: `base.en`, `bart-large`, `led-large-16384`
- **Advantage**: Unified memory allows running larger models
- **Recommended**: M2 Pro/Max or M3/M4 for production workloads
- **Device Setting**: `whisper_device: mps`, `summary_device: mps`

**NVIDIA GPU (CUDA):**

- **Best Models**: `small.en`, `bart-large`, `led-large-16384`
- **Advantage**: Highest throughput, can parallelize
- **Recommended**: RTX 3060 (8GB) minimum, RTX 4090 (24GB) for best performance
- **Device Setting**: `whisper_device: cuda`, `summary_device: cuda`

**CPU-Only:**

- **Best Models**: `tiny.en`, `bart-small`, `led-base-16384`
- **Advantage**: Works everywhere, no GPU needed
- **Limitation**: Slower, use smaller models
- **Device Setting**: `whisper_device: cpu`, `summary_device: cpu`

### Memory Planning

```text
Model Memory Requirements (VRAM/RAM)
═══════════════════════════════════════════════════════════════════════════

Whisper Models
├── tiny.en      │█                                                        1GB
├── base.en      │█                                                        1GB
├── small.en     │██                                                       2GB
├── medium.en    │█████                                                    5GB
└── large        │██████████                                              10GB

Transformers Models
├── DistilBART   │█                                                        500MB
├── BART-base    │█                                                        1GB
├── BART-large   │███                                                      3GB
├── LED-base     │██                                                       2GB
└── LED-large    │████                                                     4GB

spaCy Models
├── en_core_web_sm │                                                      50MB
├── en_core_web_md │█                                                      200MB
└── en_core_web_lg │█████                                                  500MB

Total (Production Stack)
├── base.en + bart-large + led-large + en_core_web_sm
│   │████████████████████████████████████████████████████████████████│ ~8GB
└── Recommended: 16GB+ for comfortable operation
```

---

## 📊 Complete Model Comparison Matrix

```text
                    ┌─────────────────────────────────────────────────────────┐
                    │           MODEL SELECTION MATRIX                        │
                    ├─────────────────────────────────────────────────────────┤
                    │  Category  │  Model        │  Size  │  Quality │ Speed │
                    ├────────────┼───────────────┼────────┼──────────┼───────┤
                    │            │ tiny.en       │  39MB  │    ⭐⭐   │  ⚡⚡⚡ │
                    │            │ base.en ⭐    │  74MB  │   ⭐⭐⭐  │  ⚡⚡  │
                    │ 🎙️ Whisper │ small.en      │ 244MB  │  ⭐⭐⭐⭐ │  ⚡   │
                    │            │ medium.en     │ 769MB  │ ⭐⭐⭐⭐⭐ │  🐌  │
                    │            │ large          │ 1.5GB  │ ⭐⭐⭐⭐⭐ │  🐌  │
                    │            │ large-v3       │ 1.5GB  │ ⭐⭐⭐⭐⭐ │  🐌  │
                    ├────────────┼───────────────┼────────┼──────────┼───────┤
                    │            │ en_core_web_sm │  12MB  │   ⭐⭐⭐  │  ⚡⚡⚡ │
                    │ 👤 spaCy   │ en_core_web_md │  40MB  │  ⭐⭐⭐⭐ │  ⚡⚡  │
                    │            │ en_core_web_lg │ 560MB  │ ⭐⭐⭐⭐⭐ │  ⚡   │
                    ├────────────┼───────────────┼────────┼──────────┼───────┤
                    │            │ fast           │ 300MB  │    ⭐⭐   │  ⚡⚡⚡ │
                    │            │ bart-small     │ 500MB  │   ⭐⭐⭐  │  ⚡⚡  │
                    │ 📝 MAP     │ bart-large ⭐  │  2GB   │ ⭐⭐⭐⭐⭐ │  ⚡   │
                    │            │ pegasus        │ 2.5GB  │  ⭐⭐⭐⭐ │  ⚡   │
                    ├────────────┼───────────────┼────────┼──────────┼───────┤
                    │            │ long-fast      │  1GB   │  ⭐⭐⭐⭐ │  ⚡⚡  │
                    │ 📝 REDUCE  │ long ⭐        │ 2.5GB  │ ⭐⭐⭐⭐⭐ │  ⚡   │
                    └─────────────────────────────────────────────────────────┘

    ⭐ = Production default / Recommended
```

---

## 🎨 Recommended Model Combinations

### Configuration 1: 💰 Minimal Resource ($0, Low Memory)

```yaml
# Best for: Resource-constrained environments, testing
whisper_model: tiny.en
summary_model: fast              # DistilBART
summary_reduce_model: long-fast  # LED-base
# Uses: ~2GB total memory
```

**Performance:**

- Transcription: Fast (~10x real-time on MPS)
- Summarization: Fast (~30s for 1-hour episode)
- Quality: Good for testing, acceptable for production

---

### Configuration 2: ⚖️ Balanced Production (Recommended)

```yaml
# Best for: Most production use cases
whisper_model: base.en           # ⭐ Default
summary_model: bart-large        # ⭐ Default
summary_reduce_model: long       # ⭐ Default
# Uses: ~8GB total memory
```

**Performance:**

- Transcription: Good speed (~8x real-time on MPS)
- Summarization: High quality (~2-3 min for 1-hour episode)
- Quality: Excellent balance of speed and quality

---

### Configuration 3: 🏆 Maximum Quality

```yaml
# Best for: Quality-critical production
whisper_model: small.en          # Better than base
summary_model: bart-large
summary_reduce_model: long
# Uses: ~10GB total memory
```

**Performance:**

- Transcription: High quality (~5x real-time on MPS)
- Summarization: Best quality (~3-4 min for 1-hour episode)
- Quality: Maximum accuracy

---

### Configuration 4: ⚡ Speed-Optimized

```yaml
# Best for: Batch processing, time-sensitive
whisper_model: tiny.en           # Fastest
summary_model: fast              # DistilBART
summary_reduce_model: long-fast  # LED-base
# Uses: ~3GB total memory
```

**Performance:**

- Transcription: Fastest (~12x real-time on MPS)
- Summarization: Fast (~20s for 1-hour episode)
- Quality: Good (acceptable trade-off for speed)

---

### Configuration 5: 🎯 Long-Content Optimized

```yaml
# Best for: Very long episodes (2+ hours)
whisper_model: base.en
summary_model: bart-large
summary_reduce_model: long       # 16k context handles long content
# Uses: ~8GB total memory
```

**Performance:**

- Transcription: Standard
- Summarization: Optimized for long content (LED handles 16k tokens)
- Quality: Excellent for long transcripts

---

## 🔬 Model Selection Deep Dives

### Whisper Model Selection

**Factors to Consider:**

1. **Language**: Use `.en` variants for English (faster, more accurate)
2. **Hardware**: Larger models require more VRAM
3. **Quality Needs**: `base.en` is usually sufficient, `small.en` for high quality
4. **Speed Requirements**: `tiny.en` for fastest, `base.en` for balanced

**Fallback Behavior:**

- System automatically falls back to smaller models if requested model unavailable
- Fallback chain: `large` → `medium.en` → `small.en` → `base.en` → `tiny.en`

### spaCy Model Selection

**Factors to Consider:**

1. **Name Complexity**: Common names work with `sm`, rare names need `lg`
2. **Metadata Quality**: Noisy metadata benefits from `lg` model
3. **Memory Constraints**: `sm` is sufficient for most cases
4. **Accuracy Requirements**: Upgrade to `md` or `lg` if missing names

**Default Choice:**

- `en_core_web_sm` is the default and works well for 80%+ of cases
- Only upgrade if you're consistently missing speaker names

### Transformers Model Selection

**MAP Model Selection:**

- **`bart-large`**: Best quality, production default
- **`bart-small`**: Faster, good for testing
- **`fast`**: Fastest, acceptable quality
- **`pegasus`**: Alternative style, similar quality to BART

**REDUCE Model Selection:**

- **`long` (LED-large)**: Best quality, handles up to 16k tokens
- **`long-fast` (LED-base)**: Faster, still handles 16k tokens
- **Key Insight**: LED models preserve more detail (65-75% compression is normal)

**Model-Specific Thresholds (from Issue #283):**

- **BART Models**: 4k token ceiling, 60% validation threshold
- **LED Models**: 6k token ceiling, 75% validation threshold
- These thresholds prevent false warnings and optimize quality

---

## 📈 Performance Benchmarks

### Transcription Speed (Real-time Factor)

Real-time factor = audio_duration / processing_time (higher is better)

| Model | Apple M2 (MPS) | NVIDIA RTX 4090 | CPU (8-core) |
| :---- | :-------------: | :--------------: | :-----------: |
| tiny.en | 12x | 15x | 1.2x |
| base.en | 8x | 10x | 0.8x |
| small.en | 5x | 6x | 0.5x |
| medium.en | 2x | 3x | 0.2x |
| large | 1.5x | 2x | 0.15x |

### Summarization Speed (1-hour episode)

| Configuration | Time | Quality |
| :------------ | :--: | :-----: |
| fast + long-fast | ~20s | Good |
| bart-small + long-fast | ~45s | Very Good |
| bart-large + long-fast | ~90s | Excellent |
| bart-large + long | ~120s | Best |

---

## 📏 Content Size Evaluation Guide

Different models excel at different content lengths and styles. Use this guide to select models based on your transcript characteristics.

### Content Length Categories

| Category | Token Range | Typical Duration | Characteristics |
| :------- | :---------: | :--------------: | :-------------- |
| **Short** | < 1,000 | < 5 minutes | News-style, concise, structured |
| **Medium** | 1,000-4,000 | 5-20 minutes | Conversational, moderate detail |
| **Long** | 4,000-15,000 | 20-60 minutes | Detailed, abstract, multi-topic |
| **Very Long** | 15,000-30,000 | 60-120 minutes | Extended discussions, deep dives |
| **Extreme** | 30,000+ | 120+ minutes | Multi-hour interviews, series |

### Model Recommendations by Content Size

**Short Content (< 1k tokens):**

- **Best MAP**: BART-large, DistilBART, PEGASUS
- **Best REDUCE**: LED-base (fast), or direct summarization
- **Why**: Classic models excel at concise, news-style content. No need for large context windows.
- **Example**: News podcasts, brief interviews, announcements

**Medium Content (1k-4k tokens):**

- **Best MAP**: BART-large, PEGASUS-large
- **Best REDUCE**: LED-base, LED-large
- **Why**: BART handles this range well. LED provides good synthesis without excessive detail preservation.
- **Example**: Standard podcast episodes, interviews, discussions

**Long Content (4k-15k tokens):**

- **Best MAP**: LED-base, LED-large, LongT5 (planned)
- **Best REDUCE**: LED-large, Qwen2.5-7B (planned)
- **Why**: Large context windows needed. LED preserves important detail. Hybrid LLMs provide better abstraction.
- **Example**: Long-form interviews, detailed discussions, multi-topic episodes

**Very Long Content (15k-30k tokens):**

- **Best MAP**: LED-large, LongT5-large (planned)
- **Best REDUCE**: LED-large, Qwen2.5-7B/14B (planned)
- **Why**: Maximum context required. LED-large handles compression well. Qwen2.5 provides best abstraction.
- **Example**: Extended interviews, deep dives, multi-hour episodes

**Extreme Content (30k+ tokens):**

- **Best MAP**: LED-large (maximum context)
- **Best REDUCE**: Qwen2.5-14B (planned, 32k context)
- **Why**: Only models with largest context windows can handle this. Hybrid architecture essential.
- **Example**: Multi-hour interviews, full series discussions, extended panels

### Content Style Considerations

**News-Style / Structured Prose:**

- **Best Models**: BART, PEGASUS
- **Why**: Trained on CNN/DailyMail, excel at extracting facts from structured text
- **Content Fit**: Short to medium, well-structured transcripts

**Conversational / Dialogue:**

- **Best Models**: LED, Hybrid LLMs (planned)
- **Why**: Better at handling natural speech patterns, fillers, repetition
- **Content Fit**: Medium to long, conversational transcripts

**Abstract / Conceptual:**

- **Best Models**: LED-large, Qwen2.5 (planned)
- **Why**: Preserve detail, better abstraction, handle complex ideas
- **Content Fit**: Long to very long, abstract discussions

**Noisy / Unstructured:**

- **Best Models**: Hybrid LLMs (planned), LED-large
- **Why**: Instruction-tuned LLMs filter noise better, LED handles unstructured content
- **Content Fit**: Any length, but hybrid models excel at filtering

---

## 🔄 Model Evolution & Roadmap

### Current State (Stable Baseline) ✅

As documented in [ML Provider Reference](ML_PROVIDER_REFERENCE.md), the current BART/LED implementation is **frozen and stable** as of Issue #83.

**Current Stack:**

- **Whisper**: `base.en` (production), `tiny.en` (testing)
- **spaCy**: `en_core_web_sm` (default)
- **Transformers (Classic)**: `bart-large` + `led-large-16384` (production)

**Content Size Coverage:**

- ✅ **Short to Medium** (1k-4k tokens): Excellent with BART
- ✅ **Long** (4k-15k tokens): Good with LED
- ⚠️ **Very Long** (15k+ tokens): Works but quality limitations observed

### Future Evolution

**Phase 1: Hybrid MAP-REDUCE (RFC-042, v2.5) 📋 Planned:**

- **MAP Models**: LED, LongT5 (new), PEGASUS
- **REDUCE Models**: Qwen2.5, LLaMA-3, Mistral, Phi-3 (instruction-tuned LLMs)
- **Benefits**: Better abstraction, structure adherence, content filtering
- **Content Size**: Better handling of **long to very long** content (15k-30k+ tokens)
- **Status**: Design complete, implementation planned for v2.5

**Phase 2: Intelligent Synthesis (Future):**

- **Models**: Mixture-of-Experts (MoE) models
- **Approach**: Semantic chunking instead of fixed token chunks
- **Benefits**: Better handling of very long content, more intelligent synthesis
- **Content Size**: Optimized for **extreme** content (30k+ tokens)

### Model Status Summary

| Category | Implemented | Planned | Total |
| :------- | :---------: | :-----: | :---: |
| **Whisper** | 6 | 0 | 6 |
| **spaCy** | 3 | 0 | 3 |
| **Classic MAP** | 5 | 0 | 5 |
| **Classic REDUCE** | 2 | 0 | 2 |
| **Hybrid MAP** | 2 | 2 | 4 |
| **Hybrid REDUCE** | 0 | 5 | 5 |
| **Total** | **18** | **7** | **25** |

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
   │  SPEED  │         │ QUALITY │         │ MEMORY  │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ tiny.en │         │ small.en│         │ tiny.en │
   │  fast   │         │ large   │         │ bart-   │
   │         │         │ bart-   │         │ small   │
   │         │         │ large   │         │         │
   └─────────┘         └─────────┘         └─────────┘

        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ BALANCE │         │  LONG   │         │  BEST   │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ base.en │         │ base.en │         │ small.en│
   │ bart-   │         │ bart-   │         │ bart-   │
   │ large   │         │ large   │         │ large   │
   │ long    │         │ long    │         │ long    │
   └─────────┘         └─────────┘         └─────────┘
```

---

## 📚 Related Documentation

- [ML Provider Reference](ML_PROVIDER_REFERENCE.md) - Technical deep dive into ML provider architecture
- [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) - Comparison of all providers (cloud + local)
- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) - Quick config examples
- [RFC-042: Hybrid Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md) - Hybrid MAP-REDUCE architecture design
- [ADR-036: Hybrid MAP-REDUCE Summarization](../adr/ADR-036-hybrid-map-reduce-summarization.md) - Architecture decision for hybrid approach
- [Issue #283](https://github.com/chipi/podcast_scraper/issues/283) - LED model compression characteristics and thresholds
- [Issue #83](https://github.com/chipi/podcast_scraper/issues/83) - Preprocessing pipeline importance and model freezing

---

## 🎯 Summary & Key Takeaways

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KEY TAKEAWAYS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🥇 PRODUCTION DEFAULTS (Current):                                          │
│     Whisper:      base.en        (best quality/speed balance)               │
│     spaCy:        en_core_web_sm (good accuracy, fast)                     │
│     MAP:          bart-large     (best quality)                             │
│     REDUCE:       led-large-16384 (16k context, best quality)               │
│                                                                             │
│  🥇 PLANNED DEFAULTS (v2.5 Hybrid):                                        │
│     MAP:          led-base        (fast compression)                         │
│     REDUCE:       qwen2.5-7b     (best abstraction, Mac default)            │
│                                                                             │
│  🥇 FASTEST:                                                                │
│     Whisper:      tiny.en        (12x real-time on MPS)                    │
│     MAP:          fast            (DistilBART, fastest)                     │
│     REDUCE:       long-fast       (LED-base, faster)                        │
│     Hybrid REDUCE: phi3-mini     (planned, fastest LLM)                    │
│                                                                             │
│  🥇 HIGHEST QUALITY:                                                        │
│     Whisper:      large-v3       (best accuracy)                           │
│     spaCy:        en_core_web_lg (best name detection)                     │
│     MAP:          bart-large     (best summarization)                      │
│     REDUCE:       led-large-16384 (best synthesis, current)                 │
│     Hybrid REDUCE: qwen2.5-14b   (planned, best abstraction)               │
│                                                                             │
│  📏 CONTENT SIZE INSIGHT:                                                   │
│     Short (<1k):     BART, PEGASUS excel (news-style)                      │
│     Medium (1k-4k):  BART-large, LED-base (balanced)                       │
│     Long (4k-15k):   LED-large, LongT5 (planned)                          │
│     Very Long (15k+): LED-large + Qwen2.5 (planned, hybrid)                │
│                                                                             │
│  📈 MEMORY INSIGHT:                                                         │
│     Production stack (base.en + bart-large + led-large) = ~8GB             │
│     Hybrid stack (led-base + qwen2.5-7b 4-bit) = ~10GB                     │
│     → Recommended: 16GB+ RAM for comfortable operation                      │
│                                                                             │
│  ⚡ SPEED INSIGHT:                                                           │
│     Apple Silicon (MPS) provides excellent performance                      │
│     → 8x real-time transcription with base.en                              │
│     → Unified memory allows larger models                                  │
│     → llama.cpp + Metal for hybrid LLMs (planned)                          │
│                                                                             │
│  🎯 QUALITY INSIGHT:                                                        │
│     LED models preserve more detail (65-75% compression is normal)         │
│     → Higher token ceilings (6k vs 4k for BART)                            │
│     → Model-specific thresholds prevent false warnings                     │
│     → Hybrid LLMs provide better abstraction (planned)                      │
│                                                                             │
│  🔄 HYBRID ARCHITECTURE (Planned):                                         │
│     MAP (compression): Classic models (LED, LongT5)                        │
│     REDUCE (abstraction): Instruction LLMs (Qwen, LLaMA, Mistral)          │
│     → Better structure adherence, content filtering, deduplication          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
