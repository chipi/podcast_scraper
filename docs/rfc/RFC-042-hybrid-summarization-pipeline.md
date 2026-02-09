# RFC-042: Hybrid Podcast Summarization Pipeline

## Status

🟡 **Proposed** - For v2.5 implementation

**Execution Timing:** **Phase 2 of 3** — Implement after
RFC-044 (Model Registry), before RFC-049 (GIL). This RFC
populates the registry with all model families and builds
the hybrid ML platform that RFC-049 consumes.

## RFC Number

042

## Authors

Podcast Scraper Team

## Date

2026-01-10

## Related ADRs

- [ADR-036: Hybrid MAP-REDUCE Summarization](../adr/ADR-036-hybrid-map-reduce-summarization.md)
- [ADR-037: Local LLM Backend Abstraction](../adr/ADR-037-local-llm-backend-abstraction.md)
- [ADR-038: Strict REDUCE Prompt Contract](../adr/ADR-038-strict-reduce-prompt-contract.md)

## Related RFCs

- [RFC-044: Model Registry](RFC-044-model-registry.md)
  (**prerequisite** — provides model metadata infrastructure)
- [RFC-045: ML Model Optimization Guide](RFC-045-ml-model-optimization-guide.md)
  (preprocessing + parameter tuning)
- [RFC-049: Grounded Insight Layer – Core](RFC-049-grounded-insight-layer-core.md)
  (downstream consumer of structured extraction)
- [RFC-050: GIL Use Cases](RFC-050-grounded-insight-layer-use-cases.md)
- [RFC-051: GIL Database Projection](RFC-051-grounded-insight-layer-database-projection.md)
- [RFC-052: Locally Hosted LLM Models](RFC-052-locally-hosted-llm-models-with-prompts.md)
  (model-specific prompt engineering for Ollama LLMs)
- [RFC-053: Adaptive Summarization Routing](RFC-053-adaptive-summarization-routing.md)
  (downstream — routes episodes to optimal strategies)

**Execution Order:**

```text
Phase 1: RFC-044 (Model Registry)           ~2-3 wk
    │     ModelCapabilities, ModelRegistry
    ▼
Phase 2: RFC-042 (this RFC — Hybrid ML)     ~10 wk
    │     Populate registry, hybrid provider,
    │     extended models (Embedding, QA, NLI)
    │     + RFC-052 (model-specific prompts, parallel)
    ▼
Phase 3: RFC-049 (Grounded Insight Layer)   ~6-8 wk
    │     Consume registry + hybrid platform for GIL
    ▼
Phase 4: RFC-053 (Adaptive Routing)         ~4-6 wk
          Route episodes to optimal strategies
          based on profiling + model diversity
```

## Related Issues

- TBD (to be created)

---

## Abstract

This RFC proposes a new hybrid MAP-REDUCE provider for podcast
transcripts that addresses persistent quality issues in the current
classic summarization approach. The hybrid provider uses classic
summarizers (LED, LongT5, PEGASUS) for efficient compression in the
MAP phase and instruction-tuned models (FLAN-T5, Qwen, LLaMA,
Mistral) for abstraction and structuring in the REDUCE phase. This
separation of concerns leverages each model class for what it does
best, resulting in higher-quality summaries with better structure
adherence, deduplication, and content filtering.

**Expanded scope (v2.5+):** Beyond summarization, this RFC
establishes the **local ML platform** for structured extraction
tasks including:

- **FLAN-T5** as a lightweight instruction-tuned model family
  (runs locally without llama.cpp)
- **Sentence-transformers** for semantic similarity, grounding
  validation, and topic clustering
- **Structured extraction** capability (JSON output from
  transcripts) — the foundation for
  RFC-049 Grounded Insight Layer

These additions generalize the hybrid architecture from
"summarization-only" to "any MAP → instruction-following REDUCE
pipeline," enabling reuse across summarization, GIL extraction,
and future structured tasks.

**Target deployment:** Local execution on Mac laptops
(Apple Silicon) and later expansion to other platforms.

**Provider status:** New optional provider alongside existing ML
provider, not a replacement.

---

## 1. Problem Statement

We are building a local, PyTorch-based podcast summarization system capable of handling long-form audio (30–90 minutes). The current ML provider relies exclusively on classic abstractive summarization models (BART, LED, PEGASUS) in both the MAP (chunk-level) and REDUCE (final summary) phases.

Despite multiple iterations (chunking strategies, bulletization, structural joiners, deduplication hints), the final summaries consistently exhibit the following failure modes:

### Observed Failure Modes

1. **Extractive or semi-extractive output** - Stitched sentences from the source rather than true abstraction
2. **Echoing of scaffolding text** - Copying schema headers, "chunk" labels, and structural hints
3. **Repetition loops** - "Chunks of chunks" recursion and duplicate ideas
4. **Poor content filtering** - Inability to reliably ignore intros, outros, sponsorships, or meta-text
5. **Structure non-compliance** - Poor adherence to desired output format (takeaways, outline, actions)

These issues persist even after significant prompt-like engineering at the reducer input level.

---

## 2. Root Cause Analysis

The root cause is **model-pipeline mismatch**.

### 2.1 Model Training Mismatch

Classic summarization models (BART, LED, PEGASUS):

- ❌ Are **not instruction-following models**
- ❌ Were trained primarily on **clean news-style prose** (CNN/DailyMail, XSum)
- ❌ Treat **all input text as equally salient content**
- ❌ Do not reliably distinguish between:
  - content vs instructions
  - metadata vs source text
  - schema vs information

**Result:** Any reducer input containing schema descriptions, structural headers, or deduplication instructions is often copied verbatim or recursively summarized, leading to degraded outputs.

### 2.2 Transcript Domain Mismatch

Podcast transcripts are:

- 🎙️ **Conversational** - Natural speech patterns, fillers, false starts
- 🔁 **Repetitive** - Ideas restated multiple times for emphasis
- 📝 **Noisy** - Timestamps, speaker labels, formatting artifacts
- 🗣️ **Structurally unlike news articles** - No headlines, no inverted pyramid

Classic summarizers default to safe extraction when faced with such input.

---

## 3. Goals

We want a local ML platform that:

1. ✅ Produces **genuinely abstractive summaries**
   (not extractive stitching)
2. ✅ Reliably **ignores non-content**
   (ads, promos, intros, outros)
3. ✅ **Deduplicates repeated ideas** across chunks
4. ✅ Outputs a **clean, predictable structure**:
   - Key takeaways (6–10 bullets)
   - Topic outline (6–12 bullets)
   - Action items (if any)
5. ✅ Runs **locally** with reasonable performance and cost
6. ✅ Works well on **Mac laptops** (Apple Silicon) initially
7. ✅ Exists as an **optional provider** alongside current ML
   provider
8. ✅ Provides **FLAN-T5** as a lightweight instruction-tuned
   model that runs via PyTorch (no llama.cpp required)
9. ✅ Adds **sentence-transformer embeddings** for semantic
   similarity, topic clustering, and grounding validation
10. ✅ Supports **structured extraction** (JSON output) beyond
    prose summarization — foundation for RFC-049 GIL

---

## 4. Non-Goals

- ❌ End-to-end training or fine-tuning of large models
- ❌ Fully extractive summarization
- ❌ Reliance on proprietary cloud-only APIs
- ❌ Replacing the existing ML provider (this is additive)
- ❌ Supporting all hardware configurations in v1 (focus on Mac)

---

## 5. Proposed Solution: Hybrid MAP-REDUCE Architecture

We propose **splitting responsibilities across model classes**, using each for what it does best.

### High-Level Architecture

```javascript
  ↓
Chunking (existing logic)
  ↓
MAP Phase (Classic Summarizer)
  ├─ LED / LongT5 / PEGASUS
  ├─ Compression: chunks → bullet notes
  └─ Output: Factual notes, no structure enforcement
  ↓
REDUCE Phase (Instruction-Tuned LLM)
  ├─ Qwen2.5 / LLaMA / Mistral (local)
  ├─ Abstraction, deduplication, filtering
  └─ Output: Structured summary (takeaways, outline, actions)
  ↓
Post-processing (minimal)
  └─ Whitespace cleanup, bullet normalization
```yaml

### Key Principle

**Separation of Concerns:**

- **MAP** = Compression engine (use classic summarizers)
- **REDUCE** = Reasoning + structuring engine (use instruction-tuned LLMs)

---

## 6. MAP Phase (Classic Summarization)

### Purpose

Use classic summarizers as **compression engines**, not final writers.

### Models

- **LED** (e.g., `allenai/led-base-16384`)
- **LongT5** (e.g., `google/long-t5-tglobal-base`)
- **PEGASUS** (optional, for shorter contexts)

### Characteristics

- **Input:** Cleaned transcript chunks (1000-4000 tokens)
- **Output:** Short bullet-style notes per chunk (100-200 tokens)
- **No instructions or schema text in input**
- **No expectations of structure or formatting compliance**

### Rationale

Classic summarizers:

- ✅ Are **efficient at compressing long inputs**
- ✅ Are **widely available in PyTorch** (`transformers`)
- ✅ Perform **adequately as "note generators"**
- ✅ Can run on CPU if needed (though GPU is faster)

---

## 7. REDUCE Phase (Instruction-Tuned LLM)

### Purpose

Perform **true abstraction, deduplication, filtering, and structuring**.

### Model Class

Instruction-tuned models, run locally:

**Tier 1 — Instruction-Tuned Seq2Seq (PyTorch-native):**

- **FLAN-T5-base** (250M, ~1GB) — Lightweight, runs on CPU,
  adequate for structured extraction and simple REDUCE tasks
- **FLAN-T5-large** (780M, ~3GB) — Better quality, still
  fits in modest hardware, good for GIL insight extraction
- **FLAN-T5-xl** (3B, ~12GB) — High quality, GPU recommended

**Tier 2 — Instruction-Tuned LLMs (llama.cpp / Ollama):**

- **Qwen2.5 Instruct** (7B / 14B) — Excellent
  instruction-following
- **LLaMA 3.x Instruct** (8B) — Strong reasoning
- **Mistral 7B Instruct** — Good balance of speed/quality
- **Phi-3 Mini Instruct** — Lighter-weight option

**Why Two Tiers?**

FLAN-T5 runs natively via `transformers` (PyTorch) with no
external dependencies (no llama.cpp, no Ollama). This makes
it the **lowest-friction entry point** for instruction-following
ML. Tier 2 models offer higher quality but require additional
infrastructure (llama.cpp compilation or Ollama installation).

For GIL extraction (RFC-049), FLAN-T5 provides a viable
pure-ML path where Tier 2 LLMs offer the premium path.

### Responsibilities

1. **Filter non-content** - Ignore ads, intros, outros, and meta-text
2. **Merge duplicate ideas** - Deduplicate across chunks
3. **Identify key insights** - Decisions, lessons, takeaways
4. **Produce structured output** - Predefined format with sections

### Why Instruction Models

These models:

- ✅ Are **trained to distinguish instructions from content**
- ✅ **Follow formatting constraints reliably**
- ✅ **Do not echo scaffolding text**
- ✅ Are **far more robust on conversational transcripts**
- ✅ Support **zero-shot task specification** via prompts

### Minimal REDUCE Prompt Contract

The REDUCE stage receives factual notes from MAP and outputs:

```markdown
Key takeaways:

- ...

Topic outline:
- ...

Action items (if any):
- ...
```

**No other text is permitted.**

**Instruction template:**

```javascript

Your task:
1. Ignore all ads, promotions, intros, outros, and meta-text
2. Merge duplicate ideas
3. Identify the most important insights, decisions, and lessons
4. Output ONLY the following sections (no additional text):

Key takeaways:
- [6-10 bullet points]

Topic outline:
- [6-12 bullet points covering main topics]

Action items (if any):
- [List any actionable items mentioned, or state "None"]

Input notes:
{map_outputs}
```yaml

---

## 8. Post-Processing

Post-processing becomes **simpler and more reliable**:

- ✅ Minimal cleanup (whitespace, bullet normalization)
- ✅ Optional section validation (bullet counts)
- ✅ Optional lightweight QA checks

Unlike the current system, **structure enforcement does not rely on fragile input hacks**.

---

## 9. Local Deployment Considerations

### MAP Phase

- **Hardware:** Can run on CPU or GPU
- **Execution:** Batched, parallelizable across chunks
- **Memory:** Memory-heavy but predictable (~2-4 GB per model)

### REDUCE Phase

- **Hardware:** GPU preferred, but CPU/Apple Silicon viable
- **Quantization:** 4-bit quantization recommended (reduces memory to ~4-8 GB)
- **Inference backend:**
  - `transformers` (PyTorch) for GPU
  - `llama.cpp` for CPU/Apple Silicon
  - `ollama` as alternative local inference server (optional)
- **Input size:** Small (already compressed by MAP)

### Apple Silicon Support

For Mac laptops (primary target):

- **MAP:** Run LED/LongT5 via `transformers` on CPU (acceptable speed)
- **REDUCE:** Run Qwen2.5/Mistral via `llama.cpp` with Metal acceleration
- **Expected performance:** ~5-10 tokens/sec on M1/M2/M3 with 4-bit quantization

---

## 10. Expected Outcomes

By adopting this hybrid approach, we expect:

1. ✅ **Significant improvement in summary quality**
2. ✅ **Elimination of schema/scaffold leakage**
3. ✅ **Better abstraction and deduplication**
4. ✅ **Clear separation of responsibilities** in the codebase
5. ✅ **Easier future upgrades** (swap MAP or REDUCE models independently)
6. ✅ **Better user experience** (higher-quality, more actionable summaries)

---

## 11. Architecture & Implementation

### 11.1 New Module: `hybrid_ml_provider.py`

Create a new provider that implements the hybrid architecture:

```python

# src/podcast_scraper/providers/ml/hybrid_ml_provider.py

from typing import Protocol
from podcast_scraper.summarization.base import SummarizationProvider

class HybridMLProvider:
    """Hybrid MAP-REDUCE summarization provider.

    Uses classic summarizers for MAP (compression) and instruction-tuned
    LLMs for REDUCE (abstraction + structuring).
    """

    def __init__(self, config: Config):
        self.config = config
        self.map_model = self._load_map_model()  # LED/LongT5
        self.reduce_model = self._load_reduce_model()  # Qwen/LLaMA/Mistral

    def _load_map_model(self):

```python

        """Load classic summarizer for MAP phase."""
        # Load LED or LongT5 from transformers
        pass

```python

    def _load_reduce_model(self):
        """Load instruction-tuned LLM for REDUCE phase."""
        # Load via transformers (GPU) or llama.cpp (CPU/Apple Silicon)
        pass

```python

    def map_phase(self, chunks: list[str]) -> list[str]:
        """Compress chunks into bullet notes."""
        # Run classic summarizer on each chunk
        pass

```python

    def reduce_phase(self, map_outputs: list[str]) -> str:
        """Produce final structured summary."""
        # Run instruction-tuned LLM with prompt template
        pass

```python

    def summarize(self, transcript: str) -> str:
        """Full pipeline: chunk → MAP → REDUCE."""
        chunks = self._chunk_transcript(transcript)
        map_outputs = self.map_phase(chunks)
        final_summary = self.reduce_phase(map_outputs)
        return self._post_process(final_summary)

```

## 11.2 Inference Backend Abstraction

Create a backend abstraction for REDUCE model loading:

```python

# src/podcast_scraper/providers/ml/inference_backends.py

class InferenceBackend(Protocol):
    """Protocol for local LLM inference."""

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text from prompt."""
        ...

class TransformersBackend:
    """PyTorch transformers backend (GPU)."""
    pass

class LlamaCppBackend:
    """llama.cpp backend (CPU/Apple Silicon)."""
    pass

class OllamaBackend:

```text
    """Ollama backend (optional)."""
    pass
```

## 11.3 Model Registry

> **Note:** The model registry infrastructure
> (`ModelCapabilities`, `ModelRegistry` class,
> lookup/fallback logic) is defined in **RFC-044**. This
> section documents the **model entries** that RFC-042
> populates into that registry.

Maintain a registry of all supported models across all
capabilities:

```python
# src/podcast_scraper/providers/ml/model_registry.py

# ── MAP Phase (Classic Summarizers) ──────────────────
MAP_MODELS = {
    "led-base": "allenai/led-base-16384",
    "led-large": "allenai/led-large-16384",
    "longt5-base": "google/long-t5-tglobal-base",
    "longt5-large": "google/long-t5-tglobal-large",
}

# ── REDUCE Phase (Instruction-Tuned) ────────────────
# Tier 1: Seq2Seq (PyTorch-native, no llama.cpp)
REDUCE_MODELS_SEQ2SEQ = {
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
}

# Tier 2: LLMs (llama.cpp / Ollama)
REDUCE_MODELS_LLM = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3-8b": "meta-llama/Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

# Combined for backwards compatibility
REDUCE_MODELS = {
    **REDUCE_MODELS_SEQ2SEQ,
    **REDUCE_MODELS_LLM,
}

# ── Embedding Models (Sentence-Transformers) ────────
EMBEDDING_MODELS = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
}

# ── Extractive QA Models ────────────────────────────
EXTRACTIVE_QA_MODELS = {
    "roberta-squad2": "deepset/roberta-base-squad2",
    "deberta-squad2": "deepset/deberta-v3-base-squad2",
}

# ── NLI / Cross-Encoder Models ──────────────────────
NLI_MODELS = {
    "nli-deberta-base": "cross-encoder/nli-deberta-v3-base",
    "nli-deberta-small": (
        "cross-encoder/nli-deberta-v3-small"
    ),
}
```

## 11.4 Extended Model Ecosystem

Beyond MAP and REDUCE models for summarization, the hybrid
ML platform provides additional model families that enable
downstream features (notably RFC-049 Grounded Insight Layer).

### 11.4.1 FLAN-T5: Lightweight Instruction-Tuned Models

**Why FLAN-T5 specifically?**

FLAN-T5 occupies a unique position in the model landscape:

| Property | FLAN-T5 | Classic (BART/LED) | LLMs (Qwen/LLaMA) |
| --- | --- | --- | --- |
| Instruction-following | ✅ Yes | ❌ No | ✅ Yes |
| Runs via `transformers` | ✅ Native | ✅ Native | ❌ Needs llama.cpp |
| Structured output (JSON) | ⚠️ Decent | ❌ No | ✅ Good |
| Memory (base) | ~1 GB | ~0.5-2 GB | ~4-8 GB (4-bit) |
| External dependencies | None | None | llama.cpp or Ollama |
| Zero-shot capability | ✅ Good | ❌ No | ✅ Excellent |

FLAN-T5 is the **lowest-friction instruction-following model**:
no llama.cpp compilation, no Ollama daemon, runs on CPU or MPS
via standard `transformers`. This makes it the default
"just works" option for structured extraction.

**Use Cases for FLAN-T5:**

1. **REDUCE phase** for hybrid summarization (alternative
   to Tier 2 LLMs)
2. **Insight extraction** for RFC-049 GIL (structured output
   from transcript chunks)
3. **Topic labeling** — zero-shot classification of transcript
   segments
4. **General instruction-following** — any task requiring
   "do X with this text" locally

**Loading Pattern:**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large"
)
tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-large"
)

# Instruction-style prompt
prompt = (
    "Extract the 5 key insights from this text. "
    "Output as a JSON list.\n\n"
    f"Text: {transcript_chunk}"
)
inputs = tokenizer(
    prompt, return_tensors="pt", max_length=512,
    truncation=True
)
outputs = model.generate(**inputs, max_new_tokens=256)
result = tokenizer.decode(
    outputs[0], skip_special_tokens=True
)
```

### 11.4.2 Sentence-Transformers: Embedding Models

Sentence-transformers produce dense vector embeddings for
text, enabling semantic similarity, clustering, and retrieval.

**Primary Model:** `all-MiniLM-L6-v2` (~90 MB, very fast)

**Use Cases:**

1. **Grounding validation** (RFC-049): Score semantic
   similarity between an Insight and a candidate Quote
   to validate SUPPORTED_BY edges
2. **Topic clustering**: Group related insights or quotes
   by semantic similarity
3. **Deduplication**: Detect near-duplicate insights across
   episodes (for global graph assembly)
4. **Semantic search**: Find relevant quotes given a
   natural-language query (RFC-050 use cases)

**Loading Pattern:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Encode texts
insight_emb = model.encode(
    "AI regulation will lag innovation"
)
quote_emb = model.encode(
    "Regulation will lag innovation by 3-5 years"
)

# Cosine similarity
from sentence_transformers.util import cos_sim
score = cos_sim(insight_emb, quote_emb)
# score ≈ 0.92 → strong support
```

**Memory Footprint:**

| Model | Size | Speed | Quality |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 90 MB | Very fast | Good |
| `all-MiniLM-L12-v2` | 120 MB | Fast | Better |
| `all-mpnet-base-v2` | 420 MB | Moderate | Best |

Default: `all-MiniLM-L6-v2` (best size/quality tradeoff).

### 11.4.3 Extractive QA Models

Extractive QA models find **verbatim text spans** in a
source document that answer a given question. This is
critical for RFC-049 Quote extraction.

**Why extractive QA is better than LLMs for quotes:**

- Returns **exact character offsets** (`start`, `end`)
- **Guarantees** the answer IS in the source text
  (extractive by design — cannot hallucinate)
- Fast, deterministic, small (~500 MB)
- Directly satisfies GIL grounding contract:
  `Quote.text == transcript[char_start:char_end]`

**Primary Model:** `deepset/roberta-base-squad2` (~500 MB)

**Loading Pattern:**

```python
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
)

result = qa(
    question="What evidence supports: "
             "AI regulation will lag innovation?",
    context=transcript_text,
)
# result = {
#     "answer": "Regulation will lag innovation "
#               "by 3-5 years",
#     "start": 4521,
#     "end": 4567,
#     "score": 0.89,
# }
```

**Multi-Quote Extraction Pattern:**

For each insight, extract multiple supporting quotes by
running QA with different question formulations or by using
top-k answers:

```python
def extract_supporting_quotes(
    insight: str,
    transcript: str,
    qa_pipeline,
    top_k: int = 3,
) -> list[dict]:
    """Find top-k verbatim spans supporting insight."""
    questions = [
        f"What evidence supports: {insight}?",
        f"What was said about: {insight}?",
        f"Quote supporting: {insight}",
    ]
    candidates = []
    for q in questions:
        result = qa_pipeline(
            question=q,
            context=transcript,
            top_k=top_k,
        )
        if isinstance(result, list):
            candidates.extend(result)
        else:
            candidates.append(result)

    # Deduplicate by span overlap, sort by score
    return _deduplicate_spans(candidates)[:top_k]
```

### 11.4.4 NLI Cross-Encoder Models

Natural Language Inference (NLI) models classify whether
a premise (Quote) **supports**, **contradicts**, or is
**neutral** to a hypothesis (Insight). Used for rigorous
grounding validation in RFC-049.

**Primary Model:** `cross-encoder/nli-deberta-v3-base`
(~400 MB)

**Use Case:** Validate SUPPORTED_BY edges — given an
Insight and a candidate Quote, determine if the quote
genuinely supports the insight.

**Loading Pattern:**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch

model_name = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

premise = "Regulation will lag innovation by 3-5 years"
hypothesis = "AI regulation will lag innovation"

inputs = tokenizer(
    premise, hypothesis,
    return_tensors="pt", truncation=True,
)
with torch.no_grad():
    logits = model(**inputs).logits
    # Labels: 0=contradiction, 1=neutral, 2=entailment
    probs = torch.softmax(logits, dim=-1)
    entailment_score = probs[0][2].item()
# entailment_score ≈ 0.91 → strong support
```

### 11.4.5 Model Memory Budget (All Models Combined)

Total memory footprint for all model families at
recommended defaults:

| Model Family | Model | Size | Purpose |
| --- | --- | --- | --- |
| MAP | LED-base | ~1 GB | Compression |
| REDUCE | FLAN-T5-large | ~3 GB | Structured extraction |
| Embedding | MiniLM-L6 | ~90 MB | Similarity |
| Extractive QA | RoBERTa-SQuAD2 | ~500 MB | Quote spans |
| NLI | DeBERTa-NLI | ~400 MB | Grounding |
| **Total** | | **~5 GB** | **Full GIL-capable** |

With Tier 2 LLM (Qwen2.5-7B, 4-bit) instead of FLAN-T5:

| Model Family | Model | Size | Purpose |
| --- | --- | --- | --- |
| MAP | LED-base | ~1 GB | Compression |
| REDUCE | Qwen2.5-7B (4-bit) | ~4 GB | Premium extraction |
| Embedding | MiniLM-L6 | ~90 MB | Similarity |
| Extractive QA | RoBERTa-SQuAD2 | ~500 MB | Quote spans |
| NLI | DeBERTa-NLI | ~400 MB | Grounding |
| **Total** | | **~6 GB** | **Premium GIL** |

Both configurations fit comfortably on a 16 GB Mac laptop
with room for the OS and other processes.

---

## 11.5 Structured Extraction Capability

Beyond summarization, the hybrid architecture generalizes
to **any task** that follows the pattern:

```text
Input text → MAP (compress/chunk) → REDUCE (reason/structure)
```

**Structured extraction** is a key application where the
REDUCE model outputs JSON instead of prose. This capability
is the foundation for RFC-049 Grounded Insight Layer.

### 11.5.1 Extraction Protocol

Define a generic extraction protocol that any model can
implement:

```python
# src/podcast_scraper/providers/ml/extraction.py

from typing import Protocol

class StructuredExtractor(Protocol):
    """Protocol for structured extraction from text."""

    def extract(
        self,
        text: str,
        schema: dict,
        prompt_template: str | None = None,
    ) -> dict:
        """Extract structured data from text.

        Args:
            text: Input text (transcript chunk or full)
            schema: Expected output JSON schema
            prompt_template: Jinja2 template name

        Returns:
            Extracted data conforming to schema
        """
        ...
```

### 11.5.2 Extraction Tasks (v1)

The following extraction tasks are enabled by the
hybrid platform:

| Task | MAP Model | REDUCE Model | Output |
| --- | --- | --- | --- |
| Summarization | LED/LongT5 | FLAN-T5/Qwen | Prose summary |
| Insight extraction | LED/LongT5 | FLAN-T5/Qwen | JSON: insights[] |
| Quote extraction | — (direct) | Extractive QA | JSON: quotes[] |
| Topic extraction | — (direct) | FLAN-T5/KeyBERT | JSON: topics[] |
| Grounding validation | — (direct) | NLI model | JSON: scores[] |

**Key Observation:** Quote extraction uses extractive QA
(not the MAP-REDUCE pattern) because verbatim span
extraction is fundamentally different from abstraction.
The extractive QA model finds the exact span in the
source text — no compression or rephrasing needed.

### 11.5.3 GIL Extraction Pipeline

When all models are available, the GIL extraction pipeline
combines them:

```text
Transcript
  │
  ├─► MAP (LED/LongT5) → compressed notes
  │     └─► REDUCE (FLAN-T5/LLM) → insights[]
  │
  ├─► For each insight:
  │     └─► Extractive QA → supporting quotes[]
  │           └─► Each quote has char_start, char_end
  │
  ├─► NLI Validation:
  │     └─► For each (insight, quote) pair:
  │           └─► entailment score → SUPPORTED_BY edge
  │
  ├─► Topic Extraction:
  │     └─► spaCy NER + FLAN-T5 labeling → topics[]
  │
  └─► Assembly → kg.json (RFC-049 schema)
```

### 11.5.4 Prompt Templates for Extraction

Extraction prompts follow the existing `prompt_store`
pattern (RFC-017):

```text
src/podcast_scraper/prompts/
  hybrid_ml/
    summarization/
      reduce_v1.j2          # Existing: prose summary
    extraction/
      insight_v1.j2         # NEW: extract insights
      topic_v1.j2           # NEW: extract topics
```

**Example: `insight_v1.j2`:**

```text
Extract the key insights from these notes.
Each insight should be a clear, standalone takeaway.

Output ONLY valid JSON in this format:
{
  "insights": [
    {
      "text": "...",
      "confidence": 0.0-1.0
    }
  ]
}

Notes:
{{ map_outputs }}
```

---

## 12. Configuration Schema

### 12.1 New Config Fields

Add to `Config` model in `src/podcast_scraper/config.py`:

```python
class Config(BaseModel):
    # ... existing fields ...

    # ── Summarization Provider ──────────────────
    summarization_provider: Literal[
        "transformers",  # Classic-only
        "hybrid_ml",     # Hybrid MAP-REDUCE
        "openai",
        "anthropic",
        # ... other providers
    ] = "transformers"

    # ── Hybrid Provider Config ──────────────────
    hybrid_map_model: str = "led-base"
    hybrid_reduce_model: str = "flan-t5-large"
    hybrid_reduce_backend: Literal[
        "transformers", "llama_cpp", "ollama"
    ] = "transformers"  # Default to PyTorch
    hybrid_map_device: Literal[
        "cpu", "cuda", "mps"
    ] = "cpu"
    hybrid_reduce_device: Literal[
        "cpu", "cuda", "mps"
    ] = "mps"
    hybrid_quantization: Literal[
        "none", "4bit", "8bit"
    ] = "none"  # FLAN-T5 doesn't need quantization

    # ── Embedding Model Config ──────────────────
    embedding_model: str = "minilm-l6"
    embedding_device: Literal[
        "cpu", "cuda", "mps"
    ] = "cpu"

    # ── Extractive QA Config ────────────────────
    extractive_qa_model: str = "roberta-squad2"
    extractive_qa_device: Literal[
        "cpu", "cuda", "mps"
    ] = "cpu"

    # ── NLI Grounding Config ────────────────────
    nli_model: str = "nli-deberta-base"
    nli_device: Literal[
        "cpu", "cuda", "mps"
    ] = "cpu"
```

### 12.2 Example: FLAN-T5 Config (Mac, Pure PyTorch)

```yaml
# config.hybrid.flant5.yaml
# Lowest-friction option: no llama.cpp, no Ollama

summarization_provider: hybrid_ml

hybrid_map_model: led-base
hybrid_reduce_model: flan-t5-large
hybrid_reduce_backend: transformers  # PyTorch native
hybrid_map_device: cpu
hybrid_reduce_device: mps  # Metal for Apple Silicon
hybrid_quantization: none  # FLAN-T5 fits without it

# Embedding + QA for GIL support
embedding_model: minilm-l6
extractive_qa_model: roberta-squad2
nli_model: nli-deberta-base
```

### 12.3 Example: LLM Config (Mac, llama.cpp)

```yaml
# config.hybrid.llm.yaml
# Premium quality, requires llama.cpp

summarization_provider: hybrid_ml

hybrid_map_model: led-base
hybrid_reduce_model: qwen2.5-7b
hybrid_reduce_backend: llama_cpp
hybrid_map_device: cpu
hybrid_reduce_device: mps
hybrid_quantization: 4bit

embedding_model: minilm-l6
extractive_qa_model: roberta-squad2
nli_model: nli-deberta-base
```

### 12.4 Example: NVIDIA GPU Config

```yaml
# config.hybrid.gpu.yaml

summarization_provider: hybrid_ml

hybrid_map_model: led-large
hybrid_reduce_model: qwen2.5-14b
hybrid_reduce_backend: transformers
hybrid_map_device: cuda
hybrid_reduce_device: cuda
hybrid_quantization: 4bit

embedding_model: mpnet-base
extractive_qa_model: deberta-squad2
nli_model: nli-deberta-base
```

---

## 13. Provider Integration

### 13.1 Factory Function

Update `src/podcast_scraper/summarization/factory.py`:

```python
def create_summarization_provider(config: Config) -> SummarizationProvider:
    """Factory for summarization providers."""

    provider_type = config.summarization_provider

    if provider_type == "transformers":
        from podcast_scraper.providers.ml.summarizer import TransformersSummarizer
        return TransformersSummarizer(config)

    elif provider_type == "hybrid_ml":  # NEW
        from podcast_scraper.providers.ml.hybrid_ml_provider import HybridMLProvider
        return HybridMLProvider(config)

    elif provider_type == "openai":
        from podcast_scraper.providers.openai.openai_provider import OpenAIProvider
        return OpenAIProvider(config)

    # ... other providers

```text

    else:
        raise ValueError(f"Unknown summarization provider: {provider_type}")

```yaml

### 13.2 Backwards Compatibility

- ✅ Existing configs with `summarization_provider: transformers`
  continue to work
- ✅ No breaking changes to existing code
- ✅ New provider is **opt-in via configuration**
- ✅ Extended models (embedding, QA, NLI) are loaded lazily —
  only when a feature requires them

---

## 13A. Foundation for RFC-049 (Grounded Insight Layer)

This section explicitly documents how RFC-042 enables
RFC-049, establishing a clear dependency chain.

### What RFC-042 Provides to RFC-049

| RFC-042 Deliverable | RFC-049 Consumer |
| --- | --- |
| FLAN-T5 model loading + inference | Insight extraction (structured JSON) |
| Sentence-transformers loading | Grounding validation (similarity scoring) |
| Extractive QA pipeline | Quote extraction (verbatim spans) |
| NLI cross-encoder | SUPPORTED_BY edge validation |
| Model registry | Central model management |
| Inference backend abstraction | Unified model loading across tasks |
| Prompt template integration | GIL extraction prompts |
| MAP-REDUCE pattern | Insight extraction on long transcripts |

### What RFC-049 Handles Independently

These capabilities are GIL-specific and do NOT belong
in RFC-042:

- **GIL extraction orchestration** — coordinating insight,
  quote, and topic extraction into a single pipeline
- **`kg.json` assembly** — combining extracted data into
  the GIL schema
- **Grounding contract enforcement** — ensuring every
  insight has explicit grounding status
- **Schema validation** — validating outputs against
  `kg.schema.json`
- **Workflow integration** — adding GIL as a pipeline
  stage in `orchestration.py`

### Dependency Chain

```text
RFC-042 (ML Platform)
  ├─ FLAN-T5 models loaded ──────────────► RFC-049 uses for insight extraction
  ├─ Sentence-transformers loaded ───────► RFC-049 uses for grounding
  ├─ Extractive QA pipeline ready ───────► RFC-049 uses for quote extraction
  ├─ NLI model loaded ──────────────────► RFC-049 uses for validation
  ├─ Model registry populated ──────────► RFC-049 references models by key
  └─ Structured extraction protocol ────► RFC-049 implements for GIL
```

### Provider-Tier Mapping for GIL

RFC-049 defines three extraction tiers. RFC-042 enables
Tiers 1 and 2:

| Tier | Insight Source | Quote Source | Grounding |
| --- | --- | --- | --- |
| **Tier 1: ML** | FLAN-T5 (RFC-042) | Extractive QA (RFC-042) | Sentence similarity (RFC-042) |
| **Tier 2: Hybrid** | MAP+LLM (RFC-042) | Extractive QA (RFC-042) | NLI + LLM verify (RFC-042) |
| **Tier 3: Cloud LLM** | API provider | Extractive QA (RFC-042) | NLI validation (RFC-042) |

**Critical note:** Extractive QA for quotes is used in
**all tiers** — even Tier 3 (cloud LLM). This is because
extractive QA guarantees verbatim spans, which LLMs cannot.
The grounding contract demands it.

---

## 14. Migration Plan

### Prerequisite: RFC-044 (Model Registry)

RFC-044 must be implemented first (~2-3 weeks). It provides
the `ModelCapabilities`, `ModelRegistry`, and lookup/fallback
infrastructure that this RFC depends on.

### Phase 1: Infrastructure (Week 1-2)

1. Create `hybrid_ml_provider.py` module
2. Implement inference backend abstraction
3. Populate model registry (all model families via RFC-044)
4. Update configuration schema
5. Add factory integration

### Phase 2: MAP Implementation (Week 3)

1. Implement MAP phase with LED/LongT5
2. Preserve existing chunking logic
3. Add unit tests for MAP compression

### Phase 3: REDUCE Implementation (Week 4-5)

1. Implement REDUCE with FLAN-T5 (Tier 1, PyTorch)
2. Implement REDUCE with llama.cpp (Tier 2, LLMs)
3. Add prompt templates (summarization + extraction)
4. Add unit tests for REDUCE abstraction

### Phase 4: Extended Models (Week 6-7)

1. Implement sentence-transformer embedding loader
2. Implement extractive QA pipeline
3. Implement NLI cross-encoder loader
4. Add structured extraction protocol
5. Add unit tests for each model family

### Phase 5: Integration & Testing (Week 8)

1. End-to-end integration tests (summarization)
2. End-to-end integration tests (structured extraction)
3. Manual testing on Mac laptop
4. Documentation updates

### Phase 6: Validation (Week 9-10)

1. Run summarization on representative episodes
2. Run structured extraction on representative episodes
3. Compare quality across Tier 1 vs Tier 2
4. Iterate on prompt templates if needed

---

## 15. Alternatives Considered

### 15.1 Pure Classic Summarization (BART / LED / PEGASUS only)

**Description:** Use classic summarization models for both MAP and REDUCE stages (current approach).

**Pros:**

- ✅ Simple architecture
- ✅ Fully local, PyTorch-native
- ✅ Lower operational complexity

**Cons (Observed):**

- ❌ Strong extractive bias on transcripts
- ❌ Echoing of schema and scaffolding text
- ❌ Poor adherence to output structure
- ❌ Inability to reliably ignore ads, intros, or meta-text

**Decision:** ❌ **Rejected** due to persistent quality failures despite extensive tuning.

---

### 15.2 Pure Instruction-Tuned LLM Summarization

**Description:** Skip MAP stage and summarize full transcript directly with an instruction-tuned LLM.

**Pros:**

- ✅ Best abstraction and instruction-following
- ✅ Clean structured output

**Cons:**

- ❌ Long-context requirements are expensive (30-90 min = 40k+ tokens)
- ❌ Higher memory and latency costs
- ❌ Less efficient for iterative experimentation

**Decision:** ❌ **Rejected** for cost/performance reasons on long (30–90 min) transcripts.

---

### 15.3 Fine-Tuning a Classic Summarizer

**Description:** Fine-tune BART/LED/PEGASUS on podcast transcripts with custom targets.

**Pros:**

- ✅ Potentially better domain adaptation

**Cons:**

- ❌ Requires large curated dataset (hundreds of transcript-summary pairs)
- ❌ High training cost (GPU time, expertise)
- ❌ Still limited instruction-following ability

**Decision:** ❌ **Out of scope** for now (may revisit in future).

---

### 15.4 Cloud API for REDUCE (OpenAI/Anthropic)

**Description:** Use OpenAI GPT-4 or Anthropic Claude for REDUCE phase.

**Pros:**

- ✅ Best instruction-following quality
- ✅ No local infrastructure needed

**Cons:**

- ❌ Not fully local (requires API calls)
- ❌ Ongoing costs per episode
- ❌ Privacy concerns for sensitive podcasts

**Decision:** ❌ **Rejected** for v2.5 (goal is fully local). May offer as alternative provider later.

---

## 16. Model Selection Matrix (Local Deployment)

### 16.1 Summarization Models

| Hardware | MAP | REDUCE (Tier 1) | REDUCE (Tier 2) |
| --- | --- | --- | --- |
| **CPU only** | LED-base | FLAN-T5-base | Phi-3 Mini (llama.cpp) |
| **Apple Silicon** | LED-base | FLAN-T5-large (MPS) | Qwen2.5 7B (llama.cpp) |
| **GPU 8-12 GB** | LED-base | FLAN-T5-large | Mistral 7B (4-bit) |
| **GPU 16 GB+** | LED-large | FLAN-T5-xl | Qwen2.5 14B (4-bit) |

### 16.2 Extended Models (GIL Support)

| Hardware | Embedding | Extractive QA | NLI |
| --- | --- | --- | --- |
| **CPU only** | MiniLM-L6 | RoBERTa-SQuAD2 | DeBERTa-NLI-small |
| **Apple Silicon** | MiniLM-L6 | RoBERTa-SQuAD2 | DeBERTa-NLI-base |
| **GPU 8-12 GB** | MiniLM-L12 | DeBERTa-SQuAD2 | DeBERTa-NLI-base |
| **GPU 16 GB+** | MPNet-base | DeBERTa-SQuAD2 | DeBERTa-NLI-base |

### Recommended Defaults

- **Mac Laptop (v2.5 focus):**
  - MAP: `led-base` on CPU
  - REDUCE: `flan-t5-large` on MPS (zero-friction)
  - Embedding: `minilm-l6` on CPU
  - Extractive QA: `roberta-squad2` on CPU
  - NLI: `nli-deberta-base` on CPU
- **Mac Laptop (premium):**
  - REDUCE: `qwen2.5-7b` via llama.cpp + Metal
  - All other models same as above

---

## 17. Benchmarking & Validation

### 17.1 Qualitative Validation

Run both providers (classic vs hybrid) on **3 representative podcast episodes** and compare:

1. **Abstraction quality** - Is the summary genuinely abstractive or extractive?
2. **Duplication rate** - Are ideas repeated across chunks?
3. **Structural correctness** - Does output follow the schema?
4. **Content filtering** - Are ads/intros/outros properly ignored?
5. **Readability** - Is the summary coherent and useful?

### 17.2 Success Criteria

The hybrid provider is considered successful if:

- ✅ Summaries are **more abstractive** than classic-only approach
- ✅ **No scaffold/schema leakage** in output
- ✅ **Consistent structure** (3 sections: takeaways, outline, actions)
- ✅ **Better content filtering** (no ads/intros in summary)
- ✅ **Runs locally** on Mac laptop with acceptable performance (< 10 min per episode)

### 17.3 Out of Scope (Phase 1)

- ❌ Quantitative metrics (ROUGE, BLEU, BERTScore) - defer to RFC-041
- ❌ Cost/performance benchmarks - defer to RFC-041
- ❌ User studies - defer to future work

**Rationale:** Keep Phase 1 focused on qualitative validation. Quantitative benchmarking can leverage RFC-041 infrastructure once implemented.

---

## 18. Resolved Questions

All design questions have been resolved. Decisions are
recorded here for traceability.

1. **Which instruction model provides the best
   quality/performance tradeoff on Mac laptops?**
   **FLAN-T5-large for zero-friction; Qwen2.5-7B for
   premium quality.** Start with FLAN-T5-large (Tier 1)
   as the default — it runs natively via PyTorch with no
   external dependencies. Upgrade to Qwen2.5-7B (Tier 2)
   when Ollama is available and the user wants higher
   quality. Empirical A/B testing during Phase 6
   (Validation) will confirm the tradeoff.

2. **Should REDUCE support streaming output?**
   **No (v1).** Streaming adds complexity for marginal
   UX benefit in a batch pipeline. Summaries and
   extractions are written to files, not displayed
   live. Revisit if an interactive UI is built.

3. **Should MAP outputs be cached for iterative
   tuning?**
   **Yes.** Cache MAP outputs per (episode, model,
   chunk_params) tuple to avoid redundant compression
   during REDUCE prompt iteration. Store in episode
   output directory as `map_cache.json`. Cache is
   invalidated when MAP model or chunk params change.

4. **How to handle llama.cpp installation/
   dependencies?**
   **Optional dependency.** Package `llama-cpp-python`
   as an optional install group:
   `pip install podcast-scraper[llama]`. If not
   installed, Tier 2 LLMs fall back to Ollama backend
   (RFC-052). Users with Ollama installed need no
   llama.cpp at all.

5. **Should we support Ollama as alternative backend?**
   **Yes.** RFC-052 defines the Ollama prompt layer.
   The `OllamaBackend` in §11.2 is a first-class
   inference backend alongside `TransformersBackend`
   and `LlamaCppBackend`. Config field
   `hybrid_reduce_backend: "ollama"` selects it.

6. **FLAN-T5 structured output quality: Is JSON output
   from FLAN-T5-large reliable enough for GIL?**
   **Reliable enough with validation + fallback.**
   FLAN-T5-large produces valid JSON ≥80% of the time.
   Strategy: (1) attempt JSON parse, (2) if invalid,
   apply heuristic repair (fix trailing commas, add
   brackets), (3) if still invalid, parse as
   key-value text (like existing `SummarySchema`
   parsing). Log parse success rate for monitoring.

7. **Should extractive QA run on the full transcript
   or on pre-segmented chunks?**
   **Pre-segmented chunks (sliding window).** RoBERTa
   has a 512-token context window. Strategy: split
   transcript into 448-token windows with 64-token
   overlap. Run QA on each window, merge results by
   deduplicating overlapping spans. This is the
   standard approach and aligns with the registry's
   `max_input_tokens=512` for QA models (RFC-044).

8. **Lazy loading vs eager loading for extended
   models?**
   **Lazy loading.** Load embedding/QA/NLI models only
   when a feature requests them (e.g., `generate_kg:
   true` triggers QA + NLI loading). Keep memory low
   for summarization-only runs. Implementation: use
   `@cached_property` or explicit `_load_if_needed()`
   pattern. Each model loader checks
   `ModelRegistry.get_capabilities()` before loading.

9. **Should `sentence-transformers` be a required or
   optional dependency?**
   **Optional dependency.** Package in a separate
   install group: `pip install podcast-scraper[gil]`
   (includes sentence-transformers, extractive QA, NLI
   models). Basic summarization works without it.
   Import guarded with `try/except ImportError` and
   helpful message pointing to the install group.

---

## 19. Timeline

### Target: v2.5 Release

| Phase | Duration | Deliverables |
| --- | --- | --- |
| **Phase 1: Infrastructure** | 2 weeks | Module structure, config, backends, registry |
| **Phase 2: MAP** | 1 week | LED/LongT5 compression |
| **Phase 3: REDUCE** | 2 weeks | FLAN-T5 + llama.cpp integration |
| **Phase 4: Extended Models** | 2 weeks | Embedding, QA, NLI pipelines |
| **Phase 5: Integration** | 1 week | E2E tests, Mac testing |
| **Phase 6: Validation** | 2 weeks | Episode comparisons, iteration |
| **Total** | **10 weeks** | Full ML platform for v2.5 |

---

## 20. Success Criteria

The hybrid ML platform is ready for v2.5 when:

### Functional Requirements

- ✅ Hybrid provider available via
  `summarization_provider: hybrid_ml`
- ✅ FLAN-T5 works as REDUCE model (Tier 1, PyTorch)
- ✅ Tier 2 LLMs work via llama.cpp backend
- ✅ Produces structured summaries (takeaways, outline,
  actions)
- ✅ Existing classic provider continues to work

### Extended Model Requirements

- ✅ Sentence-transformer embeddings load and produce
  vectors
- ✅ Extractive QA pipeline returns verbatim spans with
  character offsets
- ✅ NLI cross-encoder classifies entailment/neutral/
  contradiction
- ✅ Structured extraction protocol implemented and tested
- ✅ Model registry contains all model families

### Quality Requirements

- ✅ Summaries more abstractive than classic-only approach
- ✅ No scaffold/schema text leaks into output
- ✅ FLAN-T5 produces valid structured JSON output
  ≥80% of the time
- ✅ Extractive QA returns correct spans (validates
  against transcript source)

### Performance Requirements

- ✅ Full summarization pipeline < 10 min per episode
- ✅ Full model suite memory < 16 GB on Mac laptop
- ✅ Extended models (embedding, QA, NLI) load in < 30s
  each

### Documentation Requirements

- ✅ Configuration guide for hybrid provider
- ✅ Model selection recommendations (Tier 1 vs Tier 2)
- ✅ Hardware requirements documented
- ✅ Extended model usage guide
- ✅ GIL readiness checklist (what RFC-049 needs)

---

## Conclusion

The observed summarization failures stem from a **mismatch
between task requirements and model capabilities**.

By separating **compression (MAP)** from **reasoned
abstraction and structuring (REDUCE)**, and assigning each
to the appropriate model class, this RFC proposes a robust,
scalable, and locally runnable solution for high-quality
podcast processing.

**Beyond summarization**, this RFC now establishes the
**local ML platform** for structured extraction tasks.
The addition of FLAN-T5, sentence-transformers, extractive
QA, and NLI models creates the foundation for RFC-049
(Grounded Insight Layer), where:

- **FLAN-T5** enables structured insight extraction
  with zero external dependencies
- **Extractive QA** provides verbatim quote extraction
  that satisfies the GIL grounding contract better
  than any LLM can
- **Sentence-transformers** enable semantic grounding
  validation and topic clustering
- **NLI cross-encoders** provide rigorous evidence
  verification

The hybrid provider will exist **alongside** the current
ML provider, giving users choice and flexibility. Mac
laptop support (Apple Silicon) makes this accessible to
developers and power users.

**This is a strategic enhancement that builds a local ML
platform — improving summarization today, enabling
evidence-backed insight extraction (RFC-049) tomorrow,
and powering adaptive routing (RFC-053) across diverse
content types in the future.**
