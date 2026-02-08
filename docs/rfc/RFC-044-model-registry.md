# RFC-044: Model Registry for Architecture Limits

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: ML Provider maintainers, Core developers
- **Execution Timing**: **Phase 1 of 3** — Build first, before
  RFC-042 and RFC-049. This RFC provides the infrastructure that
  RFC-042 populates and RFC-049 consumes.
- **Related RFCs**:
  - `docs/rfc/RFC-012-episode-summarization.md`
    (Summarization architecture)
  - `docs/rfc/RFC-013-openai-provider-implementation.md`
    (Provider system)
  - `docs/rfc/RFC-029-provider-refactoring-consolidation.md`
    (Provider architecture)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
    (Hybrid ML platform — populates registry with all model
    families)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
    (GIL — consumes registry for extraction model selection)
- **Related Documents**:
  - Phase 1 & 2 of hardcoded values tightening completed
    (duplicate constants resolved, validation ranges
    extracted, default extensions centralized)

## Abstract

This RFC proposes a centralized **Model Registry** to eliminate
hardcoded model architecture limits throughout the codebase.
Currently, model limits (e.g., `1024` for BART, `16384` for LED)
are scattered across multiple files, making it difficult to add
new models and maintain consistency. The registry provides a
single source of truth for model capabilities, supports dynamic
detection with intelligent fallbacks, and enables extensibility
for future model types.

**Expanded Scope (v2.5+):** With the introduction of RFC-042
(Hybrid ML Platform) and RFC-049 (Grounded Insight Layer), the
model landscape has grown from 2 model families (MAP + REDUCE
summarizers) to **6 model families**:

| Family | Purpose | Example Models |
| --- | --- | --- |
| MAP Summarizers | Chunk compression | LED, LongT5 |
| REDUCE (Seq2Seq) | Instruction-following | FLAN-T5 |
| REDUCE (LLMs) | Premium extraction | Qwen, LLaMA, Mistral |
| Embedding | Semantic similarity | MiniLM, MPNet |
| Extractive QA | Verbatim span extraction | RoBERTa-SQuAD2 |
| NLI Cross-Encoder | Grounding validation | DeBERTa-NLI |

This RFC provides the **infrastructure** that RFC-042 populates
with model entries and that RFC-049 queries for extraction model
selection.

**Architecture Alignment:** This RFC aligns with RFC-029 provider
architecture by centralizing model metadata and making the
codebase model-agnostic. It supports the provider system's goal
of pluggable model implementations.

**Execution Order:**

```text
Phase 1: RFC-044 (this RFC) — Registry infrastructure
    │     Build ModelCapabilities, ModelRegistry, lookup/fallback
    │     Duration: ~2-3 weeks
    ▼
Phase 2: RFC-042 — Hybrid ML Platform
    │     Populate registry with all model families
    │     Implement hybrid provider + extended models
    │     Duration: ~10 weeks
    ▼
Phase 3: RFC-049 — Grounded Insight Layer
          Query registry for model selection per tier
          Run extraction pipeline using registered models
          Duration: ~6-8 weeks
```

## Problem Statement

Model architecture limits are currently hardcoded in multiple locations:

- `BART_MAX_POSITION_EMBEDDINGS = 1024` in `summarizer.py:50`
- `LED_MAX_CONTEXT_WINDOW = 16384` in `summarizer.py:51`
- `DEFAULT_MAP_MAX_INPUT_TOKENS = 1024` in `config.py:166`
- `DEFAULT_REDUCE_MAX_INPUT_TOKENS = 4096` in `config.py:167`
- Dynamic detection fallbacks to hardcoded values in multiple places (e.g., `summarizer.py:1248`, `summarizer.py:1779`, `summarizer.py:2441`)

**Context:** This RFC addresses Phase 3 of the hardcoded values tightening initiative. Phase 1 (duplicate constants resolution) and Phase 2 (validation ranges and default extensions) have been completed. This RFC focuses specifically on model architecture limits, which are the highest-impact remaining hardcoded values.

**Issues:**

1. **Maintenance Burden**: Adding new models requires updating hardcoded values in multiple files
2. **Inconsistency Risk**: Limits can drift out of sync between files
3. **No Single Source of Truth**: Model capabilities are scattered and undocumented
4. **Error-Prone**: Easy to use wrong limits for new or unknown models
5. **Limited Extensibility**: Hard to add new model types (T5, LongT5, etc.) without code changes

**Impact of Not Solving This:**

- Adding new models becomes increasingly difficult and error-prone
- Risk of using incorrect limits leading to model failures or poor performance
- Code becomes harder to maintain as model support grows
- No centralized place to document model capabilities and recommendations

**Use Cases:**

1. **Adding New Model**: Developer wants to add LongT5 support - needs to update 5+ hardcoded values
2. **Dynamic Model Detection**: Code needs to determine model limits at runtime for unknown models
3. **Model Capability Lookup**: Code needs to check if a model supports long context (>4096 tokens)
4. **Extensibility**: Third-party code wants to register custom models with their limits

## Goals

1. **Centralize Model Metadata**: Single source of truth for
   all model architecture limits
2. **Eliminate Hardcoded Limits**: Remove scattered hardcoded
   values throughout codebase
3. **Support Dynamic Detection**: Maintain ability to detect
   limits from loaded model instances
4. **Enable Extensibility**: Allow registration of new models
   without code changes
5. **Type Safety**: Provide structured, type-safe model
   capability information
6. **Intelligent Fallbacks**: Pattern-based guessing for
   unknown models with safe defaults
7. **Support All Model Families**: Handle MAP, REDUCE,
   Embedding, QA, and NLI models (not just summarizers)
8. **Enable RFC-042 and RFC-049**: Provide the model
   metadata infrastructure that downstream RFCs depend on

## Constraints & Assumptions

**Constraints:**

- Must maintain backward compatibility with existing model identifiers (aliases and full IDs)
- Must not break existing dynamic detection logic that reads from `model.config`
- Must support both local transformers models and API-based models
- Must not require model loading to determine capabilities (registry should work without model instance)
- Performance: Registry lookups must be O(1) and not add noticeable overhead

**Assumptions:**

- Model identifiers (aliases or full HuggingFace IDs) are stable and won't change
- Model architecture limits don't change for a given model version
- Pattern-based fallbacks are acceptable for unknown models (with conservative defaults)
- Registry can be extended at runtime for custom models

**Test vs Production Models:**

The registry is **model-agnostic** and handles both test and production models identically:
- **Test models** (e.g., `bart-small`, `long-fast`) use smaller, faster models for CI/local dev
- **Production models** (e.g., `bart-large`, `long`) use higher-quality models for production
- **Architecture limits are the same** for models of the same type (e.g., both BART models have 1024 token limit)
- The registry provides capabilities based on model ID/type, not environment
- Model selection (test vs prod) is handled by `config_constants.py` (`TEST_DEFAULT_*` vs `PROD_DEFAULT_*`), not by the registry

## Design & Implementation

### 1. Model Registry Structure

Create a new module `src/podcast_scraper/providers/ml/model_registry.py` with:

**ModelCapabilities Dataclass:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelCapabilities:
    """Model architecture capabilities and limits.

    Generalized to support all model families:
    summarizers, instruction-tuned models, embedding
    models, extractive QA, and NLI cross-encoders.
    """

    # ── Core (all models) ──────────────────────
    max_input_tokens: int  # Maximum input tokens
    model_type: str  # "bart", "led", "flan-t5", etc.
    model_family: str  # "map", "reduce", "embedding",
    #                     "extractive_qa", "nli"
    supports_long_context: bool  # >=4096 tokens

    # ── Summarizer-specific ────────────────────
    default_chunk_size: Optional[int] = None
    default_overlap: Optional[int] = None

    # ── Instruction-tuned model fields ─────────
    supports_json_output: bool = False
    supports_extraction: bool = False

    # ── Embedding model fields ─────────────────
    embedding_dim: Optional[int] = None

    # ── Resource estimates ─────────────────────
    memory_mb: Optional[int] = None
    default_device: str = "cpu"

    # Note: Word-based defaults (900 words, 150
    # overlap) remain in config_constants.py as
    # they're global, not model-specific
```

**ModelRegistry Class:**
```python
class ModelRegistry:
    """Centralized registry of model capabilities and metadata."""

    _registry: Dict[str, ModelCapabilities] = {
        # ── BART models (1024 token limit) ─────
        "bart-large": ModelCapabilities(
            max_input_tokens=1024,
            model_type="bart",
            model_family="map",
            supports_long_context=False,
            default_chunk_size=600,
            default_overlap=60,
            memory_mb=1600,
        ),
        "bart-small": ModelCapabilities(
            max_input_tokens=1024,
            model_type="bart",
            model_family="map",
            supports_long_context=False,
            default_chunk_size=600,
            default_overlap=60,
            memory_mb=500,
        ),
        "facebook/bart-large-cnn": ModelCapabilities(
            max_input_tokens=1024,
            model_type="bart",
            model_family="map",
            supports_long_context=False,
            memory_mb=1600,
        ),
        "facebook/bart-base": ModelCapabilities(
            max_input_tokens=1024,
            model_type="bart",
            model_family="map",
            supports_long_context=False,
            memory_mb=500,
        ),
        "sshleifer/distilbart-cnn-12-6": ModelCapabilities(
            max_input_tokens=1024,
            model_type="bart",
            model_family="map",
            supports_long_context=False,
            memory_mb=1200,
        ),

        # ── PEGASUS models (1024 token limit) ──
        "pegasus": ModelCapabilities(
            max_input_tokens=1024,
            model_type="pegasus",
            model_family="map",
            supports_long_context=False,
            memory_mb=2000,
        ),
        "google/pegasus-large": ModelCapabilities(
            max_input_tokens=1024,
            model_type="pegasus",
            model_family="map",
            supports_long_context=False,
            memory_mb=2000,
        ),
        "google/pegasus-xsum": ModelCapabilities(
            max_input_tokens=1024,
            model_type="pegasus",
            model_family="map",
            supports_long_context=False,
            memory_mb=2000,
        ),

        # LED models (16384 token limit)
        "long": ModelCapabilities(
            max_input_tokens=16384,
            model_type="led",
            model_family="map",
            supports_long_context=True,
            default_chunk_size=16384,
            default_overlap=1638,
            memory_mb=2000,
        ),
        "long-fast": ModelCapabilities(
            max_input_tokens=16384,
            model_type="led",
            model_family="map",
            supports_long_context=True,
            default_chunk_size=16384,
            default_overlap=1638,
            memory_mb=1000,
        ),
        "allenai/led-large-16384": ModelCapabilities(
            max_input_tokens=16384,
            model_type="led",
            model_family="map",
            supports_long_context=True,
            memory_mb=2000,
        ),
        "allenai/led-base-16384": ModelCapabilities(
            max_input_tokens=16384,
            model_type="led",
            model_family="map",
            supports_long_context=True,
            memory_mb=1000,
        ),

        # ── FLAN-T5 (Tier 1 REDUCE — PyTorch) ──
        "google/flan-t5-base": ModelCapabilities(
            max_input_tokens=512,
            model_type="flan-t5",
            model_family="reduce",
            supports_long_context=False,
            supports_json_output=True,
            supports_extraction=True,
            memory_mb=1000,
        ),
        "google/flan-t5-large": ModelCapabilities(
            max_input_tokens=512,
            model_type="flan-t5",
            model_family="reduce",
            supports_long_context=False,
            supports_json_output=True,
            supports_extraction=True,
            memory_mb=3000,
            default_device="mps",
        ),
        "google/flan-t5-xl": ModelCapabilities(
            max_input_tokens=512,
            model_type="flan-t5",
            model_family="reduce",
            supports_long_context=False,
            supports_json_output=True,
            supports_extraction=True,
            memory_mb=12000,
            default_device="cuda",
        ),

        # ── Embedding Models ────────────────────
        "sentence-transformers/all-MiniLM-L6-v2":
            ModelCapabilities(
                max_input_tokens=256,
                model_type="sentence-transformer",
                model_family="embedding",
                supports_long_context=False,
                embedding_dim=384,
                memory_mb=90,
            ),
        "sentence-transformers/all-MiniLM-L12-v2":
            ModelCapabilities(
                max_input_tokens=256,
                model_type="sentence-transformer",
                model_family="embedding",
                supports_long_context=False,
                embedding_dim=384,
                memory_mb=120,
            ),
        "sentence-transformers/all-mpnet-base-v2":
            ModelCapabilities(
                max_input_tokens=384,
                model_type="sentence-transformer",
                model_family="embedding",
                supports_long_context=False,
                embedding_dim=768,
                memory_mb=420,
            ),

        # ── Extractive QA Models ────────────────
        "deepset/roberta-base-squad2":
            ModelCapabilities(
                max_input_tokens=512,
                model_type="roberta",
                model_family="extractive_qa",
                supports_long_context=False,
                memory_mb=500,
            ),
        "deepset/deberta-v3-base-squad2":
            ModelCapabilities(
                max_input_tokens=512,
                model_type="deberta",
                model_family="extractive_qa",
                supports_long_context=False,
                memory_mb=700,
            ),

        # ── NLI Cross-Encoder Models ────────────
        "cross-encoder/nli-deberta-v3-base":
            ModelCapabilities(
                max_input_tokens=512,
                model_type="deberta-nli",
                model_family="nli",
                supports_long_context=False,
                memory_mb=400,
            ),
        "cross-encoder/nli-deberta-v3-small":
            ModelCapabilities(
                max_input_tokens=512,
                model_type="deberta-nli",
                model_family="nli",
                supports_long_context=False,
                memory_mb=200,
            ),
    }

    @classmethod
    def get_capabilities(
        cls,
        model_id: str,
        model_instance: Optional[Any] = None,
    ) -> ModelCapabilities:
        """Get model capabilities with dynamic fallback.

        Priority:
        1. Registry lookup (by alias or full ID)
        2. Dynamic detection from model_instance.config
        3. Intelligent fallback based on model name patterns
        4. Safe default (1024 for unknown models)

        Args:
            model_id: Model identifier (alias or full HuggingFace ID)
            model_instance: Optional loaded model instance for dynamic detection

        Returns:
            ModelCapabilities with architecture limits
        """
        # 1. Check registry first
        if model_id in cls._registry:
            return cls._registry[model_id]

        # 2. Try dynamic detection from model instance
        if model_instance is not None:
            try:
                config = (
                    model_instance.model.config
                    if hasattr(model_instance, 'model')
                    else model_instance.config
                )
                max_pos = getattr(
                    config, 'max_position_embeddings',
                    None,
                )
                if max_pos:
                    model_type = cls._infer_model_type(
                        config
                    )
                    family = cls._infer_model_family(
                        model_id, model_type
                    )
                    supports_long = max_pos >= 4096
                    return ModelCapabilities(
                        max_input_tokens=max_pos,
                        model_type=model_type,
                        model_family=family,
                        supports_long_context=supports_long,
                    )
            except (AttributeError, TypeError):
                pass

        # 3. Pattern-based fallback
        lower_id = model_id.lower()
        if "led" in lower_id or "longformer" in lower_id:
            return ModelCapabilities(
                max_input_tokens=16384,
                model_type="led",
                model_family="map",
                supports_long_context=True,
            )
        if "bart" in lower_id:
            return ModelCapabilities(
                max_input_tokens=1024,
                model_type="bart",
                model_family="map",
                supports_long_context=False,
            )
        if "pegasus" in lower_id:
            return ModelCapabilities(
                max_input_tokens=1024,
                model_type="pegasus",
                model_family="map",
                supports_long_context=False,
            )
        if "flan-t5" in lower_id or "flan_t5" in lower_id:
            return ModelCapabilities(
                max_input_tokens=512,
                model_type="flan-t5",
                model_family="reduce",
                supports_long_context=False,
                supports_json_output=True,
                supports_extraction=True,
            )
        if "sentence-transformer" in lower_id:
            return ModelCapabilities(
                max_input_tokens=256,
                model_type="sentence-transformer",
                model_family="embedding",
                supports_long_context=False,
                embedding_dim=384,
            )
        if "squad" in lower_id:
            return ModelCapabilities(
                max_input_tokens=512,
                model_type="qa",
                model_family="extractive_qa",
                supports_long_context=False,
            )
        if "nli" in lower_id:
            return ModelCapabilities(
                max_input_tokens=512,
                model_type="nli",
                model_family="nli",
                supports_long_context=False,
            )

        # 4. Safe default (conservative)
        return ModelCapabilities(
            max_input_tokens=1024,
            model_type="unknown",
            model_family="unknown",
            supports_long_context=False,
        )

    @classmethod
    def _infer_model_type(cls, config: Any) -> str:
        """Infer model type from config."""
        model_type = getattr(
            config, 'model_type', ''
        ).lower()
        if 'bart' in model_type:
            return 'bart'
        if 'led' in model_type:
            return 'led'
        if 'longformer' in model_type:
            return 'led'
        if 'pegasus' in model_type:
            return 'pegasus'
        if 't5' in model_type:
            return 'flan-t5'
        return 'unknown'

    @classmethod
    def _infer_model_family(
        cls, model_id: str, model_type: str,
    ) -> str:
        """Infer model family from ID and type."""
        lower_id = model_id.lower()
        if 'nli' in lower_id:
            return 'nli'
        if 'squad' in lower_id:
            return 'extractive_qa'
        if 'sentence-transformer' in lower_id:
            return 'embedding'
        if model_type in ('bart', 'led', 'pegasus'):
            return 'map'
        if model_type in ('flan-t5',):
            return 'reduce'
        return 'unknown'

    @classmethod
    def register_model(
        cls,
        model_id: str,
        capabilities: ModelCapabilities,
    ) -> None:
        """Register a new model (for extensibility).

        Args:
            model_id: Model identifier
            capabilities: Model capabilities
        """
        cls._registry[model_id] = capabilities
```

### 2. Integration Points

**Replace hardcoded limits in `summarizer.py`:**

```python
# Before:
model_max_tokens = (
    getattr(
        model.model.config,
        "max_position_embeddings",
        BART_MAX_POSITION_EMBEDDINGS,
    )
    if model.model and hasattr(model.model, "config")
    else BART_MAX_POSITION_EMBEDDINGS
)

# After:
from .model_registry import ModelRegistry
caps = ModelRegistry.get_capabilities(
    model_name, model_instance
)
model_max_tokens = caps.max_input_tokens
```

**Replace hardcoded limits in `config.py`:**

```python
# Before:
DEFAULT_MAP_MAX_INPUT_TOKENS = 1024  # BART limit
DEFAULT_REDUCE_MAX_INPUT_TOKENS = 4096  # LED limit

# After:
# Keep as defaults, but document they're model-specific
# When model is known, use ModelRegistry
```

**Update all dynamic detection sites:**

Replace all instances of:

```python
getattr(
    model.model.config,
    "max_position_embeddings",
    BART_MAX_POSITION_EMBEDDINGS,
)
```

With:

```python
ModelRegistry.get_capabilities(
    model_name, model_instance
).max_input_tokens
```

**RFC-042 Integration (Hybrid ML Platform):**

RFC-042 registers models into the registry and queries
capabilities during model loading:

```python
# In hybrid_ml_provider.py
caps = ModelRegistry.get_capabilities(
    cfg.hybrid_reduce_model
)
if caps.supports_json_output:
    # Use structured extraction mode
    ...
if caps.model_family == "extractive_qa":
    # Use QA pipeline mode
    ...
```

**RFC-049 Integration (GIL Extraction):**

RFC-049 queries the registry to determine which models
are available per extraction tier:

```python
# In GIL extraction orchestrator
qa_caps = ModelRegistry.get_capabilities(
    cfg.extractive_qa_model
)
# QA models have max_input_tokens = 512
# → need chunking strategy for long transcripts
chunk_size = qa_caps.max_input_tokens - 64  # margin
```

### 4. Relationship to Config Files

**User Config Files** (`examples/config.example.yaml`):
- Users specify model names: `summary_model: "bart-large"` or `summary_model: "facebook/bart-large-cnn"`
- Users can optionally specify token limits: `summary_tokenize.map_max_input_tokens: 1024`
- **Registry Usage**: When processing, registry looks up capabilities based on model name from config
- **Validation Opportunity**: Registry could validate that user-specified token limits match model capabilities (warn if mismatch)

**Experiment Config Files** (`data/eval/configs/*.yaml`):
- Specify models: `backend.map_model: "bart-small"`, `backend.reduce_model: "long-fast"`
- Hardcode token limits: `tokenize.map_max_input_tokens: 1024`, `tokenize.reduce_max_input_tokens: 4096`
- **Registry Usage**: Registry could auto-populate token limits based on selected models
- **Future Enhancement**: Could derive `tokenize` settings from registry instead of hardcoding

**Current State:**
- Config files specify model names (strings) - this doesn't change
- Config files can specify token limits - these are currently hardcoded defaults
- Registry would be used **internally** during processing, not exposed in config file format

**Future Possibilities:**
1. **Auto-population**: When loading config, if model is specified but token limits aren't, derive from registry
2. **Validation**: Warn if user-specified token limits exceed model capabilities
3. **Documentation**: Config file examples could reference registry for valid model names

### 3. Default Management Strategy

**Current Default Hierarchy:**
1. **Global defaults** (in `config_constants.py`):
   - `DEFAULT_SUMMARY_WORD_CHUNK_SIZE = 900` (words)
   - `DEFAULT_SUMMARY_WORD_OVERLAP = 150` (words)
   - These are **global recommendations**, not model-specific

2. **Model-specific defaults** (currently hardcoded):
   - BART/PEGASUS: `ENCODER_DECODER_TOKEN_CHUNK_SIZE = 600` (tokens) in `summarizer.py`
   - LED: Uses full context (16384 tokens) - no chunking needed
   - These are **model-specific** and should move to registry

3. **Config defaults** (in `config.py`):
   - `DEFAULT_MAP_MAX_INPUT_TOKENS = 1024` (BART limit)
   - `DEFAULT_REDUCE_MAX_INPUT_TOKENS = 4096` (LED limit)
   - These are **fallback defaults** when model is unknown

**Registry Default Strategy:**
- **Registry provides model-specific token chunk defaults** (`default_chunk_size`, `default_overlap`)
- **Global word-based defaults remain in `config_constants.py`** (not model-specific)
- **Config defaults remain as fallbacks** when model is unknown or not in registry
- **Priority**: User config → Registry defaults → Config defaults → Hardcoded fallback

**Example Usage:**
```python
# Get model capabilities
capabilities = ModelRegistry.get_capabilities(cfg.summary_model)

# Use registry defaults if user didn't specify
chunk_size = (
    cfg.summary_chunk_size
    or capabilities.default_chunk_size
    or config.DEFAULT_SUMMARY_CHUNK_SIZE
)
```

### 3. Promotion Script Implementation

**Script Location:** `scripts/registry/promote_baseline.py`

**Responsibilities:**
1. Read baseline config from `data/eval/baselines/{baseline_id}/config.yaml`
2. Read baseline metrics from `data/eval/baselines/{baseline_id}/metrics.json` (optional, for summary)
3. Validate config completeness and correctness
4. Generate `ModeConfiguration` dataclass instance
5. Update `model_registry.py` by adding to `_mode_registry` dict
6. Preserve existing registry entries (append-only)
7. Log promotion details

**Validation Checks:**
- Config file exists and is valid YAML
- Required fields present (map_model, reduce_model, map_params, reduce_params)
- Models exist in model registry (or can be resolved)
- Preprocessing profile exists
- Metrics meet acceptance criteria (if provided)

**Example Usage:**
```bash
# Promote baseline to registry
make registry-promote \
  BASELINE_ID=baseline_ml_dev_authority_smoke_v1 \
  MODE_ID=ml_small_authority

# Verify promotion
python -c "from podcast_scraper.providers.ml.model_registry import ModelRegistry; print(ModelRegistry.get_mode_configuration('ml_small_authority'))"
```

**Make Task:**
```makefile
registry-promote:
	@echo "Promoting baseline $(BASELINE_ID) to mode $(MODE_ID)..."
	@$(PYTHON) scripts/registry/promote_baseline.py \
		--baseline-id $(BASELINE_ID) \
		--mode-id $(MODE_ID) \
		--baseline-dir data/eval/baselines/$(BASELINE_ID)
	@echo "✓ Promotion complete. Review changes to model_registry.py before committing."
```

### 4. Migration Strategy

**Phase 1: Create Registry (Model Capabilities)**
- Create `model_registry.py` with all current models
- Populate `default_chunk_size` and `default_overlap` from current hardcoded values
- Add comprehensive tests

**Phase 2: Replace in `summarizer.py`**
- Replace all hardcoded `BART_MAX_POSITION_EMBEDDINGS` and `LED_MAX_CONTEXT_WINDOW` references
- Replace `ENCODER_DECODER_TOKEN_CHUNK_SIZE` with registry lookup
- Update dynamic detection logic
- Run tests to verify behavior unchanged

**Phase 3: Add Mode Configuration Support**
- Extend registry with `ModeConfiguration` dataclass
- Add `_mode_registry` dict to `ModelRegistry`
- Implement `get_mode_configuration()` method
- Create promotion script `scripts/registry/promote_baseline.py`
- Add Make task `registry-promote`

**Phase 4: Promote First Baseline**
- Run promotion for proven baseline (e.g., `baseline_ml_dev_authority_smoke_v1`)
- Verify registry update
- Test app code can use mode configuration

**Phase 5: Update App Code to Use Modes**
- Replace `PROD_DEFAULT_*` constants with mode lookups
- Update `Config` class to support mode-based initialization
- Add runtime fingerprint logging
- Update factory methods to use registry modes

**Phase 6: Update `config.py`**
- Document that defaults come from registry modes
- Add comments referencing ModelRegistry for model-specific limits
- Keep Config defaults as fallbacks for unknown models/modes

**Phase 7: Remove Old Constants**
- Remove `BART_MAX_POSITION_EMBEDDINGS` and `LED_MAX_CONTEXT_WINDOW` from `summarizer.py`
- Remove `ENCODER_DECODER_TOKEN_CHUNK_SIZE` (replaced by registry)
- Remove `PROD_DEFAULT_*` constants (replaced by mode registry)
- Update any remaining references

**Phase 8: Add Tests**
- Test registry completeness (all models in `DEFAULT_SUMMARY_MODELS` are registered)
- Test default chunk size/overlap values match current behavior
- Test dynamic detection fallback
- Test pattern-based fallbacks
- Test extensibility (register_model)
- Test promotion script (read baseline, update registry)
- Test mode configuration lookup and usage

## Key Decisions

1. **Registry as Class with Class Variable**
   - **Decision**: Use class variable `_registry` instead of module-level dict
   - **Rationale**: Allows runtime registration via `register_model()`, easier to test, more extensible

2. **Priority Order for Capability Resolution**
   - **Decision**: Registry → Dynamic Detection → Pattern Fallback → Safe Default
   - **Rationale**: Registry is fastest and most accurate, dynamic detection is reliable when model is loaded, pattern fallback handles edge cases, safe default prevents failures

3. **Frozen Dataclass for ModelCapabilities**
   - **Decision**: Use `@dataclass(frozen=True)` for immutability
   - **Rationale**: Prevents accidental modification, ensures consistency, thread-safe

4. **Conservative Default (1024)**
   - **Decision**: Unknown models default to 1024 token limit
   - **Rationale**: Safer to underestimate than overestimate (prevents model failures from exceeding limits)

## Alternatives Considered

1. **Configuration File Approach**
   - **Description**: Store model capabilities in YAML/JSON config file
   - **Pros**: Easy to edit, no code changes needed
   - **Cons**: Runtime file I/O, harder to validate, no type safety
   - **Why Rejected**: Performance overhead, complexity of file management, less type-safe

2. **Pure Dynamic Detection**
   - **Description**: Always detect from `model.config` at runtime
   - **Pros**: Always accurate, no hardcoded values
   - **Cons**: Requires model to be loaded, slower, fails for unknown models
   - **Why Rejected**: Can't work without model instance, adds latency, no fallback for API models

3. **Separate Registry Per Model Type**
   - **Description**: Different registries for BART, LED, etc.
   - **Pros**: More organized, type-specific logic
   - **Cons**: More complex, harder to extend, duplication
   - **Why Rejected**: Over-engineering, single registry is simpler and sufficient

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: Registry lookup, dynamic detection, pattern fallbacks, safe defaults
- **Integration Tests**: Verify registry works with actual model instances
- **Completeness Tests**: Ensure all models in `DEFAULT_SUMMARY_MODELS` are registered

**Test Organization:**

- Location: `tests/unit/podcast_scraper/providers/ml/test_model_registry.py`
- Markers: `@pytest.mark.unit`, `@pytest.mark.ml_models` for integration tests
- Fixtures: Mock model instances with configs

**Test Execution:**

- Unit tests run in fast CI suite
- Integration tests require model loading (slower, marked appropriately)
- Test data: All current model aliases and full IDs

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1** (Week 1): Create registry and tests
- **Phase 2** (Week 2): Replace in `summarizer.py`, verify tests pass
- **Phase 3** (Week 3): Update `config.py`, remove old constants
- **Phase 4** (Week 4): Final verification, documentation updates

**Monitoring:**

- Track registry lookup success rate (should be 100% for known models)
- Monitor dynamic detection fallback usage (should be rare)
- Verify no regressions in model behavior

**Success Criteria:**

1. ✅ All hardcoded model limits removed from `summarizer.py`
2. ✅ Registry contains all models in `DEFAULT_SUMMARY_MODELS`
3. ✅ All tests pass (unit + integration)
4. ✅ No performance regression
5. ✅ Documentation updated

## Relationship to Other RFCs

This RFC (RFC-044) is the **foundation layer** in a three-RFC
dependency chain:

```text
RFC-044 (this RFC) → RFC-042 → RFC-049
   Registry infra       Populate      Consume
   (Phase 1)            (Phase 2)     (Phase 3)
```

**Dependency chain:**

1. **RFC-044 (Phase 1 — this RFC)**: Model registry
   infrastructure. Build `ModelCapabilities`,
   `ModelRegistry`, lookup/fallback/registration logic.
   All other RFCs depend on this.

2. **RFC-042 (Phase 2)**: Hybrid ML Platform. Populates
   the registry with all model families (MAP, REDUCE,
   Embedding, QA, NLI). Implements the hybrid provider
   and structured extraction protocol.

3. **RFC-049 (Phase 3)**: Grounded Insight Layer. Queries
   the registry for model availability per extraction
   tier. Runs the GIL extraction pipeline using
   registered models.

**Other related RFCs:**

- **RFC-012**: Summarization algorithms (uses registry
  for model limits)
- **RFC-013**: OpenAI provider (can be extended to
  include API model capabilities)
- **RFC-029**: Provider architecture (registry supports
  pluggable models)
- **RFC-045**: ML model optimization (preprocessing +
  parameter tuning, informed by registry capabilities)

**Key Distinction:**

- **RFC-044**: Model metadata, capabilities, and lookup
  infrastructure
- **RFC-042**: Model catalog, loading, and hybrid
  pipeline implementation
- **RFC-049**: Domain-specific extraction using models
  from the registry

## Benefits

1. **Single Source of Truth**: All model limits centralized in one place
2. **Eliminates Hardcoded Values**: Removes scattered magic numbers
3. **Extensibility**: Easy to add new models without code changes
4. **Type Safety**: Structured dataclass prevents errors
5. **Future-Proof**: Supports new model types (T5, LongT5, etc.)
6. **Maintainability**: Easier to update and document model capabilities

## Migration Path

1. **Phase 1**: Create `ModelRegistry` with current
   summarization models + new model families, add tests
2. **Phase 2**: Replace hardcoded limits in `summarizer.py`
   (backward compatible)
3. **Phase 3**: Update `config.py` documentation, remove
   old constants
4. **Phase 4**: Verify all tests pass, update docs
5. **Phase 5**: RFC-042 populates registry with all model
   families (FLAN-T5, Embedding, QA, NLI)
6. **Phase 6**: RFC-049 consumes registry for GIL
   extraction model selection

## Promotion Mechanism: Baseline → Registry

**Key Design Decision:** Code must never depend on `data/eval/` directly. Instead, proven baseline configurations are "promoted" into the registry via Make tasks, making them available as app defaults.

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    data/eval/ (experimentation)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  configs/    │  │  baselines/  │  │    runs/     │     │
│  │  *.yaml      │  │  */config.yaml│  │  (temporary) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                               │
│         └──────────────────┘                               │
│                  │                                          │
│         [Promotion via Make]                                │
│                  │                                          │
└──────────────────┼──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Registry (code)                         │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  ModelCapabilities (architecture limits)              │ │
│  │  ModeConfigurations (runtime params from baselines)   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  App code uses registry → never touches data/eval/         │
└─────────────────────────────────────────────────────────────┘
```

### Mode Configuration Structure

Extend the registry to include **Mode Configurations** that store complete runtime parameters:

```python
@dataclass(frozen=True)
class ModeConfiguration:
    """Complete runtime configuration for a summarization mode.

    Promoted from proven baseline configurations. These become
    the app defaults, ensuring baseline == app behavior.
    """
    mode_id: str  # e.g., "ml_small_authority", "ml_large_authority"
    map_model: str  # Model identifier (resolved alias or full ID)
    reduce_model: str  # Model identifier
    preprocessing_profile: str  # e.g., "cleaning_v4"
    map_params: Dict[str, Any]  # max_new_tokens, repetition_penalty, etc.
    reduce_params: Dict[str, Any]
    tokenize: Dict[str, Any]  # map_max_input_tokens, reduce_max_input_tokens
    chunking: Optional[Dict[str, Any]]  # strategy, word_chunk_size, etc.
    promoted_from: str  # Baseline ID that this was promoted from
    promoted_at: str  # ISO timestamp
    metrics_summary: Optional[Dict[str, Any]]  # Key metrics from baseline
```

### Promotion Workflow

**1. Make Task for Promotion:**

```makefile
# Promote baseline config to registry
registry-promote:
	@echo "Promoting baseline $(BASELINE_ID) to mode $(MODE_ID)..."
	@$(PYTHON) scripts/registry/promote_baseline.py \
		--baseline-id $(BASELINE_ID) \
		--mode-id $(MODE_ID) \
		--baseline-dir data/eval/baselines/$(BASELINE_ID)

# Example usage:
# make registry-promote BASELINE_ID=baseline_ml_dev_authority_smoke_v1 MODE_ID=ml_small_authority
```

**2. Promotion Script (`scripts/registry/promote_baseline.py`):**

```python
"""Promote baseline configuration to Model Registry.

This script reads a baseline's config.yaml and metrics.json,
extracts the proven configuration, and updates the registry
with a new mode configuration.

The registry becomes the single source of truth for app defaults,
completely decoupled from data/eval/ experimentation space.
"""
```

**3. Registry Update:**

The promotion script:
1. Reads `data/eval/baselines/{baseline_id}/config.yaml`
2. Reads `data/eval/baselines/{baseline_id}/metrics.json` (for metrics summary)
3. Validates the config (models exist, params are valid)
4. Updates `src/podcast_scraper/providers/ml/model_registry.py`:
   - Adds new `ModeConfiguration` to `_mode_registry`
   - Preserves existing entries
5. Logs promotion details (baseline_id, mode_id, timestamp)

**4. App Code Usage:**

```python
from podcast_scraper.providers.ml.model_registry import ModelRegistry

# Get mode configuration (replaces hardcoded PROD_DEFAULT_* constants)
mode_config = ModelRegistry.get_mode_configuration("ml_small_authority")

# Use in provider initialization
cfg = Config(
    summary_model=mode_config.map_model,
    summary_reduce_model=mode_config.reduce_model,
    # ... other fields from mode_config
)
```

### Benefits

1. **Complete Decoupling**: Code never imports or references `data/eval/`
2. **Single Source of Truth**: Registry is the only place app code reads defaults
3. **Explicit Promotion**: Make task makes promotion intentional and traceable
4. **Version Control**: Registry changes are committed, baseline stays in `data/eval/`
5. **No Silent Drift**: App behavior is explicitly tied to promoted baselines

### Promotion Criteria

Before promoting, validate:
- Baseline metrics meet acceptance criteria (gates pass, quality thresholds)
- Config is complete and valid
- Models are available (cached or downloadable)
- Preprocessing profile exists

### Registry Structure (Extended)

```python
class ModelRegistry:
    """Centralized registry of model capabilities and mode configurations."""

    # Model capabilities (architecture limits)
    _registry: Dict[str, ModelCapabilities] = { ... }

    # Mode configurations (runtime params from baselines)
    _mode_registry: Dict[str, ModeConfiguration] = {
        "ml_small_authority": ModeConfiguration(
            mode_id="ml_small_authority",
            map_model="bart-small",
            reduce_model="long-fast",
            preprocessing_profile="cleaning_v4",
            map_params={
                "max_new_tokens": 200,
                "min_new_tokens": 80,
                "repetition_penalty": 1.3,
                # ... full params from baseline
            },
            reduce_params={ ... },
            tokenize={ ... },
            chunking={ ... },
            promoted_from="baseline_ml_dev_authority_smoke_v1",
            promoted_at="2026-02-01T15:00:00Z",
            metrics_summary={
                "speaker_label_leak_rate": 0.0,
                "avg_tokens": 470,
            }
        ),
    }

    @classmethod
    def get_mode_configuration(cls, mode_id: str) -> ModeConfiguration:
        """Get mode configuration by ID."""
        if mode_id not in cls._mode_registry:
            raise ValueError(f"Mode {mode_id} not found in registry")
        return cls._mode_registry[mode_id]
```

### Runtime Fingerprint Logging

When provider initializes with a mode, log fingerprint:

```python
logger.info("=== Runtime Configuration Fingerprint ===")
logger.info(f"Mode ID: {mode_id}")
logger.info(f"MAP Model: {mode_config.map_model}")
logger.info(f"REDUCE Model: {mode_config.reduce_model}")
logger.info(f"Preprocessing Profile: {mode_config.preprocessing_profile}")
logger.info(f"MAP Params: max_tokens={mode_config.map_params['max_new_tokens']}, ...")
logger.info(f"REDUCE Params: max_tokens={mode_config.reduce_params['max_new_tokens']}, ...")
logger.info(f"Promoted From: {mode_config.promoted_from}")
```

## Resolved Questions

All design questions have been resolved. Decisions are
recorded here for traceability.

1. **Memory requirements in registry?**
   **Yes** — `memory_mb` field added to
   `ModelCapabilities`. Enables resource budgeting
   across model families.

2. **Recommended devices in registry?**
   **Yes** — `default_device` field added (`"cpu"`,
   `"mps"`, `"cuda"`). Informs model loading strategy.

3. **Validate model IDs against HuggingFace Hub at
   runtime?**
   **No.** Adds network dependency, latency, and
   breaks offline users (key RFC-052 use case). The
   registry handles unknown models via pattern-based
   fallback → safe defaults. Development-time
   validation is available via `make registry-validate`
   (checks Hub availability in CI, not at runtime).

4. **Support model versioning (per-version limits)?**
   **No (v1).** Architecture limits rarely change
   between versions. When they do, models have different
   HuggingFace IDs (e.g., `facebook/bart-large-cnn`),
   which the registry handles as separate entries. Easy
   to add later if needed.

5. **Auto-populate token limits from registry?**
   **Yes.** When a user specifies a model but not token
   limits, derive optimal limits from the registry.
   Priority chain: User config → Registry defaults →
   Config defaults → Hardcoded fallback. Already
   described in Default Management Strategy (§3).

6. **Validate user-specified token limits?**
   **Yes, as warnings (not errors).** If a user sets
   `map_max_input_tokens: 2048` for a BART model (1024
   limit), log a warning and clamp to the model limit.
   Prevents silent truncation failures. Implementation:
   `validate_config()` method comparing user config
   against registry capabilities. Warn-and-clamp
   strategy, never crash.

7. **Mode versioning (`v1`, `v2`) or replace in place?**
   **Replace in place.** Mode IDs stay stable (e.g.,
   `ml_small_authority`). The `promoted_from` +
   `promoted_at` fields provide full traceability. Git
   history shows what changed and when. Explicit version
   suffixes add naming burden without functional benefit.

8. **Family-specific subclasses for capabilities?**
   **No, keep flat dataclass.** Current field count (11)
   is manageable. Subclasses add type-checking
   complexity, casting boilerplate, and multiple class
   definitions for little gain. `Optional` fields with
   `None` defaults clearly signal which fields apply per
   family; `model_family` identifies the type. Revisit
   if field count exceeds ~20.

9. **Register cloud API model capabilities?**
   **Yes, but v1.1+.** Cloud API models have relevant
   capabilities (context window, JSON mode, tool calls)
   that benefit from unified lookup. v1 scope: local ML
   models only. Future phase adds entries for OpenAI,
   Gemini, Anthropic, and Ollama-hosted models
   (RFC-052). The existing `ModelCapabilities` dataclass
   already supports the needed fields.

---

## Conclusion

The scattered hardcoded model limits throughout the
codebase are a **maintenance and correctness liability**
that grows with every new model family.

By centralizing model metadata into a single
**Model Registry**, this RFC provides:

- **A single source of truth** for all model
  architecture limits — no more magic numbers in
  `summarizer.py`, `config.py`, or detection fallbacks
- **Intelligent resolution** via a four-level priority
  chain: registry lookup → dynamic detection → pattern
  fallback → safe default
- **Extensibility** through `register_model()` for
  custom models and `ModeConfiguration` for promoted
  baselines
- **Cross-RFC infrastructure** that RFC-042 populates
  with 6 model families and RFC-049 queries for
  extraction model selection

The registry is deliberately **minimal and focused**:
a frozen dataclass for capabilities, a class-variable
dict for storage, and O(1) lookups. No external files,
no network calls, no runtime overhead.

**As the foundational Phase 1 of the three-phase
build-out (RFC-044 → RFC-042 → RFC-049), this registry
is the infrastructure that makes the entire local ML
platform and Grounded Insight Layer possible.**

## References

- **Related RFC**: `docs/rfc/RFC-012-episode-summarization.md`
- **Related RFC**: `docs/rfc/RFC-029-provider-refactoring-consolidation.md`
- **Related RFC**: `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Source Code**: `src/podcast_scraper/providers/ml/summarizer.py`
- **Source Code**: `src/podcast_scraper/config.py`
