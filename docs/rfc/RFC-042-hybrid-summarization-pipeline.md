# RFC-042: Hybrid Podcast Summarization Pipeline

## Status

🟡 **Proposed** - For v2.5 implementation

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

- None (independent new provider)

## Related Issues

- TBD (to be created)

---

## Abstract

This RFC proposes a new hybrid MAP-REDUCE summarization provider for podcast transcripts that addresses persistent quality issues in the current classic summarization approach. The hybrid provider uses classic summarizers (LED, LongT5, PEGASUS) for efficient compression in the MAP phase and instruction-tuned LLMs (Qwen, LLaMA, Mistral) for abstraction and structuring in the REDUCE phase. This separation of concerns leverages each model class for what it does best, resulting in higher-quality summaries with better structure adherence, deduplication, and content filtering.

**Target deployment:** Local execution on Mac laptops (Apple Silicon) and later expansion to other platforms.

**Provider status:** New optional provider alongside existing ML provider, not a replacement.

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

We want a summarization system that:

1. ✅ Produces **genuinely abstractive summaries** (not extractive stitching)
2. ✅ Reliably **ignores non-content** (ads, promos, intros, outros)
3. ✅ **Deduplicates repeated ideas** across chunks
4. ✅ Outputs a **clean, predictable structure**:
   - Key takeaways (6–10 bullets)
   - Topic outline (6–12 bullets)
   - Action items (if any)
5. ✅ Runs **locally** with reasonable performance and cost
6. ✅ Works well on **Mac laptops** (Apple Silicon) initially
7. ✅ Exists as an **optional provider** alongside current ML provider

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

Instruction-tuned LLMs, run locally:

- **Qwen2.5 Instruct** (7B / 14B) - Excellent instruction-following
- **LLaMA 3.x Instruct** (8B) - Strong reasoning capabilities
- **Mistral 7B Instruct** - Good balance of speed/quality
- **Phi-3 Mini Instruct** - Lighter-weight option for constrained hardware

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

# src/podcast_scraper/ml/hybrid_ml_provider.py

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

# src/podcast_scraper/ml/inference_backends.py

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

Maintain a registry of supported models:

```python

# src/podcast_scraper/ml/model_registry.py

MAP_MODELS = {
    "led-base": "allenai/led-base-16384",
    "led-large": "allenai/led-large-16384",
    "longt5-base": "google/long-t5-tglobal-base",
    "longt5-large": "google/long-t5-tglobal-large",
}

REDUCE_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3-8b": "meta-llama/Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
}
```yaml

---

## 12. Configuration Schema

### 12.1 New Config Fields

Add to `Config` model in `src/podcast_scraper/config.py`:

```python
class Config(BaseModel):
    # ... existing fields ...

    # Summarization provider selection
    summarization_provider: Literal[
        "transformers",  # Current ML provider (classic-only)
        "hybrid_ml",     # NEW: Hybrid MAP-REDUCE provider
        "openai",
        "anthropic",
        # ... other providers
    ] = "transformers"

    # Hybrid provider configuration (only used if summarization_provider == "hybrid_ml")
    hybrid_map_model: str = "led-base"  # MAP phase model
    hybrid_reduce_model: str = "qwen2.5-7b"  # REDUCE phase model
    hybrid_reduce_backend: Literal["transformers", "llama_cpp", "ollama"] = "llama_cpp"
    hybrid_map_device: Literal["cpu", "cuda", "mps"] = "cpu"
    hybrid_reduce_device: Literal["cpu", "cuda", "mps"] = "mps"  # Metal for Apple Silicon
    hybrid_quantization: Literal["none", "4bit", "8bit"] = "4bit"
```

### 12.2 Example Configuration (Mac Laptop)

```yaml

# config.hybrid.yaml

rss_url: "https://feeds.npr.org/510289/podcast.xml"
output_dir: "./output"

# Use new hybrid provider

summarization_provider: hybrid_ml

# Hybrid provider settings (optimized for Apple Silicon)

hybrid_map_model: led-base
hybrid_reduce_model: qwen2.5-7b
hybrid_reduce_backend: llama_cpp
hybrid_map_device: cpu
hybrid_reduce_device: mps  # Use Metal Performance Shaders
hybrid_quantization: 4bit

# Other settings

transcription_provider: transformers
whisper_model: large-v3
max_episodes: 3
```

## 12.3 Example Configuration (NVIDIA GPU)

```yaml

# config.hybrid.gpu.yaml

summarization_provider: hybrid_ml

hybrid_map_model: led-large
hybrid_reduce_model: qwen2.5-14b
hybrid_reduce_backend: transformers  # Use PyTorch for GPU
hybrid_map_device: cuda
hybrid_reduce_device: cuda
hybrid_quantization: 4bit
```yaml

---

## 13. Provider Integration

### 13.1 Factory Function

Update `src/podcast_scraper/summarization/factory.py`:

```python
def create_summarization_provider(config: Config) -> SummarizationProvider:
    """Factory for summarization providers."""

    provider_type = config.summarization_provider

    if provider_type == "transformers":
        from podcast_scraper.ml.summarizer import TransformersSummarizer
        return TransformersSummarizer(config)

    elif provider_type == "hybrid_ml":  # NEW
        from podcast_scraper.ml.hybrid_ml_provider import HybridMLProvider
        return HybridMLProvider(config)

    elif provider_type == "openai":
        from podcast_scraper.openai.openai_provider import OpenAIProvider
        return OpenAIProvider(config)

    # ... other providers

```text

    else:
        raise ValueError(f"Unknown summarization provider: {provider_type}")

```yaml

### 13.2 Backwards Compatibility

- ✅ Existing configs with `summarization_provider: transformers` continue to work
- ✅ No breaking changes to existing code
- ✅ New provider is **opt-in via configuration**

---

## 14. Migration Plan

### Phase 1: Infrastructure (Week 1-2)

1. ✅ Create `hybrid_ml_provider.py` module
2. ✅ Implement inference backend abstraction
3. ✅ Add model registry
4. ✅ Update configuration schema
5. ✅ Add factory integration

### Phase 2: MAP Implementation (Week 3)

1. ✅ Implement MAP phase with LED/LongT5
2. ✅ Preserve existing chunking logic
3. ✅ Add unit tests for MAP compression

### Phase 3: REDUCE Implementation (Week 4-5)

1. ✅ Implement REDUCE phase with instruction-tuned LLMs
2. ✅ Add llama.cpp backend for Apple Silicon
3. ✅ Add prompt template system
4. ✅ Add unit tests for REDUCE abstraction

### Phase 4: Integration & Testing (Week 6)

1. ✅ End-to-end integration tests
2. ✅ Manual testing on Mac laptop
3. ✅ Documentation updates

### Phase 5: Validation (Week 7-8)

1. ✅ Run on representative podcast episodes
2. ✅ Compare quality against current provider
3. ✅ Iterate on prompt templates if needed

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

| Hardware | MAP (Chunk Compression) | REDUCE (Final Summary) |
| ---------- | ------------------------- | ------------------------ |
| **CPU only** | LED-base, LongT5-base | Phi-3 Mini Instruct (llama.cpp) |
| **Apple Silicon (M1/M2/M3)** | LED-base (CPU) | Qwen2.5 7B / Mistral 7B (llama.cpp + Metal) |
| **NVIDIA GPU 8–12GB** | LED-base / LongT5 | Mistral 7B Instruct (4-bit quantization) |
| **NVIDIA GPU 16GB+** | LED-large / LongT5-large | Qwen2.5 7B–14B Instruct (4-bit quantization) |

### Recommended Defaults

- **Mac Laptop (v2.5 focus):**
  - MAP: `led-base` on CPU
  - REDUCE: `qwen2.5-7b` via llama.cpp with Metal, 4-bit quantization

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

## 18. Open Questions

1. **Which instruction model provides the best quality/performance tradeoff on Mac laptops?**
   - Qwen2.5 vs LLaMA 3 vs Mistral vs Phi-3
   - May require empirical testing

2. **Should REDUCE support streaming output for UI responsiveness?**
   - Nice-to-have for future, not critical for v2.5

3. **Should MAP outputs be cached for iterative tuning?**
   - Yes, likely beneficial for experimentation

4. **How to handle llama.cpp installation/dependencies?**
   - Package as optional dependency? Pre-built binaries? Documentation-only?

5. **Should we support Ollama as alternative backend?**
   - Many users already have Ollama installed
   - Could simplify deployment

---

## 19. Timeline

### Target: v2.5 Release

| Phase | Duration | Deliverables |
| ------- | ---------- | ------------- |
| **Phase 1: Infrastructure** | 2 weeks | Module structure, config schema, backends |
| **Phase 2: MAP Implementation** | 1 week | LED/LongT5 compression working |
| **Phase 3: REDUCE Implementation** | 2 weeks | Instruction LLM integration, llama.cpp |
| **Phase 4: Integration & Testing** | 1 week | End-to-end tests, Mac testing |
| **Phase 5: Validation** | 2 weeks | Episode comparisons, iteration |
| **Total** | **8 weeks** | Hybrid provider ready for v2.5 |

---

## 20. Success Criteria

The hybrid summarization provider is considered ready for v2.5 release when:

### Functional Requirements

- ✅ Hybrid provider is available via `summarization_provider: hybrid_ml`
- ✅ Works on Mac laptops (Apple Silicon) with llama.cpp backend
- ✅ Produces structured summaries with 3 sections (takeaways, outline, actions)
- ✅ Existing classic provider continues to work (backwards compatibility)

### Quality Requirements

- ✅ Summaries are more abstractive than classic-only approach
- ✅ No scaffold/schema text leaks into output
- ✅ Ads/intros/outros are filtered out
- ✅ Duplicate ideas are merged across chunks

### Performance Requirements

- ✅ Full pipeline completes in < 10 minutes per episode on Mac laptop
- ✅ Memory usage < 16 GB (reasonable for modern laptops)

### Documentation Requirements

- ✅ Configuration guide for hybrid provider
- ✅ Model selection recommendations
- ✅ Hardware requirements documented
- ✅ Troubleshooting guide for common issues

---

## Conclusion

The observed summarization failures stem from a **mismatch between task requirements and model capabilities**.

By separating **compression (MAP)** from **reasoned abstraction and structuring (REDUCE)**, and assigning each to the appropriate model class, this RFC proposes a robust, scalable, and locally runnable solution for high-quality podcast summarization.

The hybrid provider will exist **alongside** the current ML provider, giving users choice and flexibility. Mac laptop support (Apple Silicon + llama.cpp) makes this accessible to developers and power users, with future expansion to other platforms.

**This is a strategic enhancement that leverages modern instruction-tuned LLMs while maintaining local-first principles.**
