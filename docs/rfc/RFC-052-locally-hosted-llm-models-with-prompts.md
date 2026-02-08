# RFC-052: Locally Hosted LLM Models with Prompts

- **Status**: Draft
- **Date**: 2026-02-05
- **Authors**:
- **Stakeholders**: Maintainers, privacy-conscious users,
  cost-conscious users, offline users
- **Execution Timing**: **Phase 2b (parallel with RFC-042)**
  — Prompt engineering layer that runs alongside the Hybrid
  ML Platform build. No hard dependency on RFC-044, but
  models should be registered in the registry once available.
- **Related PRDs**:
  - `docs/prd/PRD-014-ollama-provider-integration.md`
    (Ollama provider foundation)
- **Related RFCs**:
  - `docs/rfc/RFC-037-ollama-provider-implementation.md`
    (Ollama provider implementation)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
    (Hybrid MAP-REDUCE architecture — Tier 2 backend)
  - `docs/rfc/RFC-044-model-registry.md`
    (Model Registry — model capability lookup)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
    (GIL — downstream consumer of local LLM extraction)
  - `docs/rfc/RFC-013-openai-provider-implementation.md`
    (Provider pattern reference)
- **Related Issues**:
  - #196 - Implement Ollama data providers (parent issue)

**Execution Order:**

```text
Phase 1:  RFC-044 (Model Registry)          ~2-3 weeks
    │
    ▼
Phase 2:  RFC-042 (Hybrid ML Platform)      ~10 weeks
    │
    ├──► Phase 2b: RFC-052 (this RFC)       parallel
    │    Model-specific prompt engineering
    │    for Ollama-hosted LLMs
    │
    ▼
Phase 3:  RFC-049 (GIL)                     ~6-8 weeks
          Uses RFC-052 prompts for local
          LLM-based GIL extraction
```

## Abstract

This RFC establishes a new area of focus for **locally
hosted LLM models with optimized prompts** to solve cost
and latency challenges while maintaining privacy. Building
on the Ollama provider foundation (RFC-037), this RFC
defines the architecture for implementing specific sub-24GB
RAM models (Qwen2.5 7B, Llama 3.1 8B, Mistral 7B,
Phi-3 Mini, Gemma 2 9B) as targeted use cases. The key innovation is
**prompt engineering** to maximize quality from smaller,
locally runnable models, making local LLM inference
practical for podcast processing workflows.

**Key Advantages:**

- **Zero API costs** — No per-token pricing
- **Reduced latency** — No network round-trips
- **Complete privacy** — Data never leaves the machine
- **Offline capable** — No internet required
- **Prompt optimization** — Custom prompts tuned for each
  model's strengths

**Expanded Scope (v2.5+):** Beyond speaker detection and
summarization, this RFC extends prompt engineering to
**GIL extraction tasks** (RFC-049):

- **Insight extraction prompts** — model-specific prompts
  for extracting structured insights from transcripts
- **Topic labeling prompts** — zero-shot topic classification
  tuned per model
- **Evidence grounding prompts** — prompts that help local
  LLMs identify supporting quotes for insights

This makes RFC-052 the **prompt quality layer** for all
local LLM tasks, while RFC-042 provides the infrastructure
and RFC-049 defines the extraction orchestration.

**Architecture Alignment:** This RFC extends the provider
system (RFC-013, RFC-029) with locally hosted LLM
capabilities, leveraging the unified provider pattern while
introducing model-specific prompt optimization as a
first-class concern.

## Problem Statement

Current cloud-based LLM providers (OpenAI, Anthropic, Gemini) solve quality and capability
challenges but introduce:

1. **Cost concerns** - Per-token pricing accumulates quickly for large-scale processing
2. **Latency issues** - Network round-trips add significant delay
3. **Privacy risks** - Data must be sent to external services
4. **Rate limits** - API throttling limits throughput
5. **Internet dependency** - Requires stable network connection

**Local LLM solutions** (via Ollama) address these concerns but face different challenges:

1. **Quality gap** - Smaller models may produce lower quality outputs
2. **Hardware constraints** - Models must fit in available RAM (< 24GB for most users)
3. **Prompt sensitivity** - Smaller models require more careful prompt engineering
4. **Model selection** - Many models available, unclear which work best for podcast tasks

**This RFC addresses:** How to leverage locally hosted LLMs effectively through:
- **Model selection** - Identify best sub-24GB RAM models for podcast tasks
- **Prompt engineering** - Optimize prompts for each model's architecture and capabilities
- **Use case implementation** - Specific implementations for speaker detection and summarization

## Goals

1. **Cost Reduction**: Eliminate API costs for users with capable hardware
2. **Latency Reduction**: Local inference removes network latency
3. **Privacy Enhancement**: Keep all data processing local
4. **Quality Optimization**: Use prompt engineering to maximize quality from smaller models
5. **Model Diversity**: Support multiple models to match different hardware capabilities
6. **Practical Implementation**: Focus on models that run on common hardware (< 24GB RAM)

## Constraints & Assumptions

**Constraints:**

- Models must run on hardware with < 24GB RAM (target: 8-16GB typical)
- Ollama must be installed and running locally
- Models must be pulled before use (`ollama pull <model>`)
- Performance depends on local hardware (CPU/GPU)
- Prompt engineering is critical for quality (smaller models are more prompt-sensitive)
- No audio transcription support (Ollama hosts LLMs only)

**Assumptions:**

- Users have sufficient hardware for chosen model (8GB+ RAM minimum)
- Ollama server is running on default port (11434)
- Users have already pulled desired models
- Prompt optimization can compensate for model size limitations
- Different models excel at different tasks (speaker detection vs summarization)

## Design & Implementation

### 1. Model Selection Strategy

**Target Models (Sub-24GB RAM):**

| # | Model | Size | RAM | Best For |
| --- | --- | --- | --- | --- |
| 1 | **Qwen2.5 7B** | ~4.4GB | 8GB+ | GIL extraction, summarization — best structured JSON in class |
| 2 | **Llama 3.1 8B** | 4.7GB | 8GB+ | General purpose, speaker detection — strong all-rounder |
| 3 | **Mistral 7B** | 4.1GB | 8GB+ | Fast inference, speaker detection — fastest in class |
| 4 | **Gemma 2 9B** | 5.5GB | 12GB+ | Summarization quality — strong for longer outputs |
| 5 | **Phi-3 Mini** | 2.3GB | 4GB+ | Dev/test, low-resource — lightest, good for CI |

**Why Qwen2.5 7B at priority 1:** GIL extraction
(RFC-049) is the biggest new use case and requires
reliable structured JSON output (insights, quotes,
topics). Qwen2.5 7B produces the most consistent
well-formed JSON among sub-10B models, has excellent
instruction-following for complex multi-step prompts,
and is already referenced in RFC-042 as a Tier 2
REDUCE model. Llama 3.1 8B is a better general
all-rounder, but for the structured extraction that
GIL demands, Qwen edges ahead.

**Selection Criteria:**

- **RAM footprint** — Must fit in < 24GB
  (target: 8-16GB)
- **Quality** — Must produce acceptable results
  for podcast tasks
- **Structured output** — Must reliably produce
  well-formed JSON for GIL extraction
- **Speed** — Must complete inference in reasonable
  time (< 5 min per episode)
- **Availability** — Must be available via Ollama
  (`ollama pull`)

### 2. Prompt Engineering Architecture

**Key Principle:** Each model requires **optimized prompts** tailored to its architecture,
training, and capabilities.

**Prompt Structure:**

```text
src/podcast_scraper/prompts/ollama/
├── qwen2.5_7b/                    # Priority 1
│   ├── ner/
│   │   ├── system_ner_v1.j2
│   │   └── guest_host_v1.j2
│   ├── summarization/
│   │   ├── system_v1.j2
│   │   └── long_v1.j2
│   └── extraction/                # GIL prompts
│       ├── insight_v1.j2
│       └── topic_v1.j2
├── llama3.1_8b/                   # Priority 2
│   ├── ner/
│   ├── summarization/
│   └── extraction/
├── mistral_7b/                    # Priority 3
│   ├── ner/
│   ├── summarization/
│   └── extraction/
├── gemma2_9b/                     # Priority 4
│   ├── ner/
│   ├── summarization/
│   └── extraction/
└── phi3_mini/                     # Priority 5
    ├── ner/
    ├── summarization/
    └── extraction/
```

**Prompt Optimization Strategy:**

1. **Model-specific tuning** - Each model has different instruction-following capabilities
2. **Task-specific prompts** - Speaker detection vs summarization require different approaches
3. **Context window awareness** - Smaller models may need chunked inputs
4. **Output format constraints** - Structured outputs (JSON) may need explicit formatting
   instructions
5. **Iterative refinement** - Prompts evolve based on quality testing

### 3. Implementation Pattern

**Per-Model Implementation:**

Each model gets its own implementation issue tracking:
- Model-specific prompt development
- Quality validation
- Performance benchmarking
- Integration testing
- Documentation

**Unified Provider Integration:**

All models integrate via the existing Ollama provider (RFC-037):
- Single `OllamaProvider` class
- Model selection via config (`ollama_speaker_model`, `ollama_summary_model`)
- Prompt selection based on model name
- Backward compatible with existing Ollama implementation

### 4. Configuration Schema

**Model Selection:**

```python
# Environment-based defaults (like OpenAI)
ollama_speaker_model: str = Field(
    default="llama3.1:8b",
    description="Ollama model for speaker detection"
)

ollama_summary_model: str = Field(
    default="qwen2.5:7b",
    description="Ollama model for summarization"
)

ollama_extraction_model: str = Field(
    default="qwen2.5:7b",
    description="Ollama model for GIL extraction"
)
```

**Default Rationale:** Qwen2.5 7B is the default for
summarization and GIL extraction due to superior
structured JSON output. Llama 3.1 8B remains the
default for speaker detection due to strong NER
performance across diverse podcast formats.

**Prompt Selection (Automatic):**

```python
def _get_prompt_path(model_name: str, task: str, prompt_version: str) -> str:
    """Get model-specific prompt path.

    Examples:
        llama3.1:8b + ner + v1 -> prompts/ollama/llama3.1_8b/ner/system_ner_v1.j2
        mistral:7b + summarization + v1 -> prompts/ollama/mistral_7b/summarization/system_v1.j2
    """
    model_dir = model_name.replace(":", "_").replace("-", "_")
    return f"prompts/ollama/{model_dir}/{task}/{prompt_version}.j2"
```

## Key Decisions

1. **Model-Specific Prompts**
   - **Decision**: Each model gets its own prompt directory with optimized prompts
   - **Rationale**: Different models have different instruction-following capabilities and
     training. Generic prompts produce suboptimal results.

2. **Sub-24GB RAM Focus**
   - **Decision**: Focus on models that run on < 24GB RAM (target: 8-16GB)
   - **Rationale**: Makes local LLM inference accessible to most users. Larger models (70B+)
     require specialized hardware and are out of scope.

3. **Separate Issues Per Model**
   - **Decision**: Create individual GitHub issues for each model implementation
   - **Rationale**: Allows parallel work, clear ownership, and focused implementation. Each
     model has unique characteristics requiring dedicated attention.

4. **Prompt Engineering as First-Class Concern**
   - **Decision**: Prompt optimization is a core implementation task, not an afterthought
   - **Rationale**: Smaller models are more prompt-sensitive. Quality depends heavily on
     prompt design.

5. **Unified Provider Integration**
   - **Decision**: All models integrate via existing Ollama provider, not separate providers
   - **Rationale**: Maintains consistency, reduces code duplication, leverages existing
     infrastructure.

## Alternatives Considered

1. **Generic Prompts for All Models**
   - **Description**: Use the same prompts for all models
   - **Pros**: Simpler implementation, less maintenance
   - **Cons**: Suboptimal quality, doesn't leverage model-specific strengths
   - **Why Rejected**: Quality is critical. Model-specific prompts are necessary for
     acceptable results.

2. **Separate Provider Per Model**
   - **Description**: Create `LlamaProvider`, `MistralProvider`, etc.
   - **Pros**: Clear separation, model-specific optimizations
   - **Cons**: Code duplication, maintenance burden, inconsistent patterns
   - **Why Rejected**: Unified provider pattern (RFC-013) is established and works well.
     Model selection via config is cleaner.

3. **Focus on Single Model Only**
   - **Description**: Implement only one model (e.g., Llama 3.1 8B)
   - **Pros**: Faster implementation, less complexity
   - **Cons**: Doesn't address diverse hardware capabilities, limits user choice
   - **Why Rejected**: Different users have different hardware. Multiple models provide flexibility.

## Testing Strategy

**Test Coverage:**

- **Unit tests**: Prompt loading, model name parsing, prompt path resolution
- **Integration tests**: End-to-end workflows with each model (speaker detection, summarization)
- **Quality validation**: Manual review of outputs from each model
- **Performance benchmarking**: Inference time, memory usage per model

**Test Organization:**

- `tests/unit/providers/ollama/test_prompt_loading.py`
  — Prompt loading logic
- `tests/integration/providers/ollama/test_qwen2.5_7b.py`
  — Qwen2.5 7B integration
- `tests/integration/providers/ollama/test_llama3.1_8b.py`
  — Llama 3.1 8B integration
- `tests/integration/providers/ollama/test_mistral_7b.py`
  — Mistral 7B integration
- `tests/integration/providers/ollama/test_gemma2_9b.py`
  — Gemma 2 9B integration
- `tests/integration/providers/ollama/test_phi3_mini.py`
  — Phi-3 Mini integration

**Test Execution:**

- Unit tests run in CI (fast, no Ollama dependency)
- Integration tests require Ollama + models (manual/local testing)
- Quality validation: Manual review of 3-5 representative episodes per model
- Performance benchmarking: Measure on representative hardware

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: RFC approval and issue creation
- **Phase 2**: Implement Qwen2.5 7B (priority 1 — best
  for GIL extraction + summarization) — NER +
  summarization + extraction prompts
- **Phase 3**: Implement Llama 3.1 8B (priority 2 —
  best all-rounder, NER default) — NER +
  summarization + extraction prompts
- **Phase 4**: Implement Mistral 7B (priority 3 —
  fastest inference) — NER + summarization prompts
- **Phase 5**: Implement Gemma 2 9B (priority 4 —
  strong summarization) — NER + summarization prompts
- **Phase 6**: Implement Phi-3 Mini (priority 5 —
  lightweight dev/test)
- **Phase 7**: GIL extraction prompts for remaining
  models (Mistral, Gemma, Phi-3) — aligned with
  RFC-049 Phase 3 kickoff

**Monitoring:**

- **Quality metrics**: Manual review scores per model
- **Performance metrics**: Inference time, memory usage
- **Usage tracking**: Which models are used most (via config analysis)
- **Error rates**: Model-specific failures or timeouts

**Success Criteria:**

1. ✅ All 5 models implemented with model-specific
   prompts
2. ✅ Qwen2.5 7B + Llama 3.1 8B have full prompt
   coverage (NER + summarization + GIL extraction)
3. ✅ Quality acceptable for production use (validated
   via manual review)
4. ✅ Performance acceptable (< 5 min per episode on
   target hardware)
5. ✅ Documentation complete (model selection guide,
   prompt development guide)
6. ✅ Integration tests passing for all models

## Integration with RFC-042 and RFC-049

### Relationship to RFC-042 (Hybrid ML Platform)

RFC-042 defines the **infrastructure** for local ML/LLM
execution. RFC-052 provides the **prompt optimization**
that makes that infrastructure effective:

| RFC-042 Provides | RFC-052 Adds |
| --- | --- |
| OllamaBackend (inference) | Model-specific prompts |
| Tier 2 REDUCE models | Prompt tuning per model |
| Structured extraction protocol | GIL extraction prompts |
| Inference backend abstraction | Prompt path resolution |

**Key Point:** RFC-042's Tier 2 models (Qwen2.5, LLaMA,
Mistral, Phi-3) overlap with RFC-052's target models.
RFC-042 focuses on the **infrastructure** (loading,
inference, backends); RFC-052 focuses on **prompt quality**
(model-specific instructions, output format tuning).

### Relationship to RFC-049 (GIL Extraction)

When RFC-049 uses local LLMs via Ollama for GIL extraction,
RFC-052 provides the optimized prompts:

| GIL Task | RFC-052 Prompt | Notes |
| --- | --- | --- |
| Insight extraction | `extraction/insight_v1.j2` | Per-model tuned |
| Topic labeling | `extraction/topic_v1.j2` | Zero-shot classification |
| Evidence grounding | Verification prompts | Complement to NLI |

**Extraction tier mapping:**

- **Tier 1 (ML):** FLAN-T5 prompts (in RFC-042, not RFC-052)
- **Tier 2 (Hybrid):** RFC-052 prompts via Ollama
- **Tier 3 (Cloud LLM):** Cloud provider prompts (existing)

RFC-052 prompts are the **Tier 2 prompt library** for
GIL extraction.

### Relationship to RFC-044 (Model Registry)

RFC-052 models should be registered in the RFC-044 registry
so that RFC-049 can query model capabilities:

```python
# RFC-044 registry entries for RFC-052 models
"ollama/qwen2.5:7b": ModelCapabilities(
    max_input_tokens=32768,
    model_type="qwen",
    model_family="reduce",
    supports_json_output=True,
    supports_extraction=True,
    memory_mb=8000,
    default_device="cpu",
)
"ollama/llama3.1:8b": ModelCapabilities(
    max_input_tokens=8192,
    model_type="llama",
    model_family="reduce",
    supports_json_output=True,
    supports_extraction=True,
    memory_mb=8000,
    default_device="cpu",
)
```

This enables RFC-049 to check `supports_extraction` before
attempting GIL extraction via a local LLM.

## Relationship to Other RFCs

This RFC (RFC-052) is part of a layered architecture:

```text
RFC-044 (Model Registry) ── model capabilities
    │
    ▼
RFC-042 (Hybrid ML Platform) ── infrastructure
    │
    ├──► RFC-052 (this RFC) ── prompt optimization
    │    Parallel: model-specific prompts
    │    for Ollama-hosted LLMs
    │
    ▼
RFC-049 (GIL) ── domain extraction
    Uses RFC-052 prompts for Tier 2 extraction
```

**Locally Hosted LLM Initiative:**

1. **RFC-037: Ollama Provider Implementation** — Foundation
   provider infrastructure (API, unified pattern)
2. **RFC-052: Locally Hosted LLM Models with Prompts** —
   This RFC (prompt engineering + model strategy)
3. **Individual Model Issues** — Specific implementations
   per model

**Key Distinction:**

- **RFC-037**: Ollama provider infrastructure (API calls,
  unified pattern)
- **RFC-042**: Hybrid ML platform (inference backends,
  model loading, structured extraction)
- **RFC-052**: Model-specific prompt engineering (the
  "quality knob" for local LLMs)
- **RFC-049**: GIL extraction orchestration (consumes
  prompts from RFC-052 for Tier 2)

Together, these provide:

- Complete local LLM solution (no API costs, privacy,
  offline)
- Multiple model options for different hardware
- Optimized prompts for summarization AND GIL extraction
- Seamless integration with the multi-provider architecture

## Benefits

1. **Cost Elimination**: Zero API costs for users with capable hardware
2. **Latency Reduction**: Local inference removes network round-trips
3. **Privacy Enhancement**: All processing stays local
4. **Offline Capability**: No internet required after model download
5. **Hardware Flexibility**: Multiple models support different RAM constraints
6. **Quality Optimization**: Model-specific prompts maximize quality from smaller models

## Migration Path

**For Users:**

1. Install Ollama: `brew install ollama` (or equivalent)
2. Pull desired models:
   `ollama pull qwen2.5:7b` and/or
   `ollama pull llama3.1:8b`
3. Configure provider:
   `speaker_detector_provider: "ollama"`,
   `summary_provider: "ollama"`
4. Select models:
   `ollama_speaker_model: "llama3.1:8b"`,
   `ollama_summary_model: "qwen2.5:7b"`
5. Run pipeline: Models and prompts are automatically
   selected

**For Developers:**

1. Review RFC-052 (this document)
2. Review RFC-037 (Ollama provider implementation)
3. Pick a model issue to implement
4. Develop model-specific prompts
5. Validate quality and performance
6. Submit PR with prompts and tests

## Resolved Questions

All design questions have been resolved. Decisions are
recorded here for traceability.

1. **Prompt Versioning**: How should we version
   prompts?
   **Use `v1`, `v2`, etc. in prompt filenames.** This
   aligns with the existing `prompt_store` (RFC-017)
   pattern (e.g., `summarization/long_v1.j2`). New
   versions are created when the prompt structure
   changes meaningfully; minor tweaks (wording,
   examples) can be edited in place. The `v1` suffix
   is part of the template name, not a separate
   metadata field. Old versions are kept in the repo
   for A/B comparison but are not loaded by default.

2. **Quality Thresholds**: What quality level is
   acceptable for production use?
   **Manual review on 5+ representative episodes for
   v1, with quantitative metrics deferred.** Each
   model×task prompt is tested against a diverse set
   of episodes (short/long, single/multi-speaker,
   interview/monologue). Acceptance criteria:
   (a) structured output parses correctly ≥95% of
   runs, (b) content quality is "acceptable" per
   human review (no hallucinated quotes, reasonable
   insight extraction), (c) no regressions vs cloud
   LLM baseline on the same episodes. Automated
   quality metrics (ROUGE, BERTScore) are deferred
   to v1.1 when a gold-standard evaluation set is
   available.

3. **Performance Targets**: What inference time is
   acceptable per episode?
   **< 5 minutes per episode on target hardware
   (M1/M2 Mac, 16GB RAM).** Breakdown: summarization
   < 2 min, speaker detection < 1 min, GIL extraction
   < 2 min. These targets assume Ollama with default
   quantization (Q4_0). If a model exceeds targets,
   try smaller quantization or a lighter model (e.g.,
   Phi-3 Mini as fallback). Performance is logged per
   run for monitoring regressions.

4. **Model Updates**: How do we handle model updates
   (e.g., Llama 3.2 released)?
   **Treat major versions as new models.** Llama 3.2
   gets its own prompt directory
   (`ollama/llama3.2_8b/`) and RFC-044 registry entry.
   Existing Llama 3.1 prompts remain unchanged. This
   allows independent prompt tuning and avoids
   breaking existing setups. Minor updates (e.g.,
   3.1.1 → 3.1.2) use the same prompts unless
   behavior changes significantly. The Ollama tag
   in config is explicit (e.g., `llama3.1:8b` vs
   `llama3.2:8b`).

5. **Prompt Testing**: How do we systematically test
   prompt quality?
   **Manual review for v1, with automated format
   tests.** Two layers: (a) Automated integration
   tests verify output format (JSON parses, required
   fields present, correct types) — these run in CI
   with `@pytest.mark.ollama`. (b) Manual quality
   review on representative episodes — performed when
   adding a new model or changing a prompt. A
   `scripts/eval_prompts.py` utility can batch-run
   all prompts against a fixed episode set and
   produce a comparison report. Full automated
   quality testing (human eval correlation, etc.) is
   deferred to post-v1.

---

## Conclusion

RFC-052 provides the **prompt engineering layer** that
enables high-quality local LLM inference through
Ollama. By creating model-specific prompt templates for
each model×task combination, it ensures that each local
LLM produces output comparable to cloud providers.

**Key design choices:**

- **Model-specific prompts** — each model gets tuned
  templates that respect its strengths and
  architectural quirks
- **Qwen2.5 7B as default** — best-in-class
  structured JSON output for GIL extraction
- **Aligned with `prompt_store`** — follows existing
  RFC-017 infrastructure for template management
- **GIL extraction included** — insight, quote, and
  topic extraction prompts are first-class citizens
  alongside summarization and speaker detection

**RFC-052 runs in parallel with RFC-042 (Phase 2b).
Together they provide: ML platform (042) + prompt
quality (052) → ready for GIL extraction (049).**

## References

- **Related PRD**: `docs/prd/PRD-014-ollama-provider-integration.md`
- **Related RFC**: `docs/rfc/RFC-037-ollama-provider-implementation.md`
- **Related RFC**: `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-044-model-registry.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related Issue**: #196 - Implement Ollama data providers
- **Source Code**: `podcast_scraper/providers/ollama/`
- **External Documentation**:
  - [Ollama Documentation](https://ollama.ai/docs)
  - [Qwen2.5 Docs](https://qwen.readthedocs.io/)
  - [Llama 3.1 Docs](https://llama.meta.com/llama3.1/)
  - [Mistral AI Docs](https://docs.mistral.ai/)
  - [Phi-3 Docs](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)
  - [Gemma 2 Docs](https://ai.google.dev/gemma/docs)
