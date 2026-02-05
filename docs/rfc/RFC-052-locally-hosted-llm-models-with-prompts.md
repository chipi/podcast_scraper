# RFC-052: Locally Hosted LLM Models with Prompts

- **Status**: Draft
- **Date**: 2026-02-05
- **Authors**:
- **Stakeholders**: Maintainers, privacy-conscious users, cost-conscious users, offline users
- **Related PRDs**:
  - `docs/prd/PRD-014-ollama-provider-integration.md` (Ollama provider foundation)
- **Related RFCs**:
  - `docs/rfc/RFC-037-ollama-provider-implementation.md` (Ollama provider implementation)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md` (Hybrid MAP-REDUCE architecture)
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (Provider pattern reference)
- **Related Issues**:
  - #196 - Implement Ollama data providers (parent issue)

## Abstract

This RFC establishes a new area of focus for **locally hosted LLM models with optimized
prompts** to solve cost and latency challenges while maintaining privacy. Building on the
Ollama provider foundation (RFC-037), this RFC defines the architecture for implementing
specific sub-24GB RAM models (Llama 3.1 8B, Mistral 7B, Phi-3 Mini, Gemma 2 9B) as targeted
use cases. The key innovation is **prompt engineering** to maximize quality from smaller,
locally runnable models, making local LLM inference practical for podcast processing
workflows.

**Key Advantages:**

- **Zero API costs** - No per-token pricing
- **Reduced latency** - No network round-trips, local inference
- **Complete privacy** - Data never leaves the machine
- **Offline capable** - No internet required
- **Prompt optimization** - Custom prompts tuned for each model's strengths

**Architecture Alignment:** This RFC extends the provider system (RFC-013, RFC-029) with
locally hosted LLM capabilities, leveraging the unified provider pattern while introducing
model-specific prompt optimization as a first-class concern.

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

| Model | Size | RAM Required | Best For | Use Case |
|-------|------|--------------|----------|----------|
| **Llama 3.1 8B** | 4.7GB | 8GB+ | General use, good quality | Speaker detection, summarization |
| **Mistral 7B** | 4.1GB | 8GB+ | Fast, good for speaker detection | Speaker detection (primary), summarization (secondary) |
| **Phi-3 Mini** | 2.3GB | 4GB+ | Lightweight, dev/test | Development, testing, resource-constrained |
| **Gemma 2 9B** | 5.5GB | 12GB+ | Balanced quality/speed | Summarization (primary), speaker detection (secondary) |

**Selection Criteria:**

- **RAM footprint** - Must fit in < 24GB (target: 8-16GB)
- **Quality** - Must produce acceptable results for podcast tasks
- **Speed** - Must complete inference in reasonable time (< 5 min per episode)
- **Availability** - Must be available via Ollama (`ollama pull`)

### 2. Prompt Engineering Architecture

**Key Principle:** Each model requires **optimized prompts** tailored to its architecture,
training, and capabilities.

**Prompt Structure:**

```
src/podcast_scraper/prompts/ollama/
├── llama3.1_8b/                    # Model-specific prompts
│   ├── ner/
│   │   ├── system_ner_v1.j2        # System prompt for NER
│   │   └── guest_host_v1.j2         # User prompt for speaker detection
│   └── summarization/
│       ├── system_v1.j2            # System prompt for summarization
│       └── long_v1.j2               # User prompt for long-form summaries
├── mistral_7b/                      # Mistral-specific prompts
│   ├── ner/
│   └── summarization/
├── phi3_mini/                       # Phi-3-specific prompts
│   ├── ner/
│   └── summarization/
└── gemma2_9b/                       # Gemma-specific prompts
    ├── ner/
    └── summarization/
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
    default="llama3.1:8b",  # Default for speaker detection
    description="Ollama model for speaker detection"
)

ollama_summary_model: str = Field(
    default="llama3.1:8b",  # Default for summarization
    description="Ollama model for summarization"
)
```

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

- `tests/unit/providers/ollama/test_prompt_loading.py` - Prompt loading logic
- `tests/integration/providers/ollama/test_llama3.1_8b.py` - Llama 3.1 8B integration
- `tests/integration/providers/ollama/test_mistral_7b.py` - Mistral 7B integration
- `tests/integration/providers/ollama/test_phi3_mini.py` - Phi-3 Mini integration
- `tests/integration/providers/ollama/test_gemma2_9b.py` - Gemma 2 9B integration

**Test Execution:**

- Unit tests run in CI (fast, no Ollama dependency)
- Integration tests require Ollama + models (manual/local testing)
- Quality validation: Manual review of 3-5 representative episodes per model
- Performance benchmarking: Measure on representative hardware

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: RFC approval and issue creation (this RFC)
- **Phase 2**: Implement Llama 3.1 8B (highest priority, best general-purpose model)
- **Phase 3**: Implement Mistral 7B (fast, good for speaker detection)
- **Phase 4**: Implement Phi-3 Mini (lightweight, dev/test)
- **Phase 5**: Implement Gemma 2 9B (balanced quality/speed)

**Monitoring:**

- **Quality metrics**: Manual review scores per model
- **Performance metrics**: Inference time, memory usage
- **Usage tracking**: Which models are used most (via config analysis)
- **Error rates**: Model-specific failures or timeouts

**Success Criteria:**

1. ✅ All 4 models implemented with model-specific prompts
2. ✅ Quality acceptable for production use (validated via manual review)
3. ✅ Performance acceptable (< 5 min per episode on target hardware)
4. ✅ Documentation complete (model selection guide, prompt development guide)
5. ✅ Integration tests passing for all models

## Relationship to Other RFCs

This RFC (RFC-052) is part of the **locally hosted LLM initiative** that includes:

1. **RFC-037: Ollama Provider Implementation** - Foundation provider implementation
2. **RFC-052: Locally Hosted LLM Models with Prompts** - This RFC (architecture and strategy)
3. **Individual Model Issues** - Specific implementations per model

**Key Distinction:**

- **RFC-037**: Implements the Ollama provider infrastructure (unified provider pattern, API
  integration)
- **RFC-052**: Defines the model-specific prompt engineering strategy and implementation approach
- **Model Issues**: Track specific model implementations (prompts, validation, integration)

Together, these provide:
- Complete local LLM solution (no API costs, privacy, offline)
- Multiple model options for different hardware capabilities
- Optimized prompts for maximum quality from smaller models

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
2. Pull desired model: `ollama pull llama3.1:8b`
3. Configure provider: `speaker_detector_provider: "ollama"`, `summary_provider: "ollama"`
4. Select model: `ollama_speaker_model: "llama3.1:8b"`, `ollama_summary_model: "llama3.1:8b"`
5. Run pipeline: Models and prompts are automatically selected

**For Developers:**

1. Review RFC-052 (this document)
2. Review RFC-037 (Ollama provider implementation)
3. Pick a model issue to implement
4. Develop model-specific prompts
5. Validate quality and performance
6. Submit PR with prompts and tests

## Open Questions

1. **Prompt Versioning**: How should we version prompts? (v1, v2, etc.)
2. **Quality Thresholds**: What quality level is acceptable for production use?
3. **Performance Targets**: What inference time is acceptable per episode?
4. **Model Updates**: How do we handle model updates (e.g., Llama 3.2 released)?
5. **Prompt Testing**: How do we systematically test prompt quality?

## References

- **Related PRD**: `docs/prd/PRD-014-ollama-provider-integration.md`
- **Related RFC**: `docs/rfc/RFC-037-ollama-provider-implementation.md`
- **Related RFC**: `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
- **Related Issue**: #196 - Implement Ollama data providers
- **Source Code**: `podcast_scraper/providers/ollama/`
- **External Documentation**:
  - [Ollama Documentation](https://ollama.ai/docs)
  - [Llama 3.1 Documentation](https://llama.meta.com/llama3.1/)
  - [Mistral AI Documentation](https://docs.mistral.ai/)
  - [Phi-3 Documentation](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)
  - [Gemma 2 Documentation](https://ai.google.dev/gemma/docs)
