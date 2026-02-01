# RFC-044: Model Registry for Architecture Limits

- **Status**: Draft
- **Authors**: [To be filled]
- **Stakeholders**: ML Provider maintainers, Core developers
- **Related RFCs**:
  - `docs/rfc/RFC-012-episode-summarization.md` (Summarization architecture)
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (Provider system)
  - `docs/rfc/RFC-029-provider-refactoring-consolidation.md` (Provider architecture)
- **Related Documents**:
  - Phase 1 & 2 of hardcoded values tightening completed (duplicate constants resolved, validation ranges extracted, default extensions centralized)

## Abstract

This RFC proposes a centralized **Model Registry** to eliminate hardcoded model architecture limits throughout the codebase. Currently, model limits (e.g., `1024` for BART, `16384` for LED) are scattered across multiple files, making it difficult to add new models and maintain consistency. The registry provides a single source of truth for model capabilities, supports dynamic detection with intelligent fallbacks, and enables extensibility for future model types.

**Architecture Alignment:** This RFC aligns with RFC-029 provider architecture by centralizing model metadata and making the codebase model-agnostic. It supports the provider system's goal of pluggable model implementations.

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

1. **Centralize Model Metadata**: Single source of truth for all model architecture limits
2. **Eliminate Hardcoded Limits**: Remove scattered hardcoded values throughout codebase
3. **Support Dynamic Detection**: Maintain ability to detect limits from loaded model instances
4. **Enable Extensibility**: Allow registration of new models without code changes
5. **Type Safety**: Provide structured, type-safe model capability information
6. **Intelligent Fallbacks**: Pattern-based guessing for unknown models with safe defaults

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
    """Model architecture capabilities and limits."""
    max_position_embeddings: int  # Maximum input tokens
    model_type: str  # "bart", "led", "pegasus", "t5", etc.
    supports_long_context: bool  # True for LED, LongT5, etc. (>=4096 tokens)
    default_chunk_size: Optional[int] = None  # Recommended token chunk size (model-specific)
    default_overlap: Optional[int] = None  # Recommended token overlap (model-specific)

    # Note: Word-based defaults (900 words, 150 overlap) remain in config_constants.py
    # as they're global recommendations, not model-specific
```

**ModelRegistry Class:**
```python
class ModelRegistry:
    """Centralized registry of model capabilities and metadata."""

    _registry: Dict[str, ModelCapabilities] = {
        # BART models (1024 token limit)
        # Note: Both test (bart-small) and production (bart-large) models have same limits
        "bart-large": ModelCapabilities(  # Production default
            max_position_embeddings=1024,
            model_type="bart",
            supports_long_context=False,
            default_chunk_size=600,  # ENCODER_DECODER_TOKEN_CHUNK_SIZE from summarizer.py
            default_overlap=60,  # 10% of 600 (CHUNK_OVERLAP_RATIO = 0.1)
        ),
        "bart-small": ModelCapabilities(  # Test default
            max_position_embeddings=1024,
            model_type="bart",
            supports_long_context=False,
            default_chunk_size=600,  # Same as bart-large (model type determines chunk size)
            default_overlap=60,
        ),
        "facebook/bart-large-cnn": ModelCapabilities(
            max_position_embeddings=1024,
            model_type="bart",
            supports_long_context=False,
        ),
        "facebook/bart-base": ModelCapabilities(
            max_position_embeddings=1024,
            model_type="bart",
            supports_long_context=False,
        ),
        "sshleifer/distilbart-cnn-12-6": ModelCapabilities(
            max_position_embeddings=1024,
            model_type="bart",
            supports_long_context=False,
        ),

        # PEGASUS models (1024 token limit)
        "pegasus": ModelCapabilities(
            max_position_embeddings=1024,
            model_type="pegasus",
            supports_long_context=False,
        ),
        "google/pegasus-large": ModelCapabilities(
            max_position_embeddings=1024,
            model_type="pegasus",
            supports_long_context=False,
        ),
        "google/pegasus-xsum": ModelCapabilities(
            max_position_embeddings=1024,
            model_type="pegasus",
            supports_long_context=False,
        ),

        # LED models (16384 token limit)
        # Note: Both test (long-fast) and production (long) models have same limits
        "long": ModelCapabilities(  # Production default
            max_position_embeddings=16384,
            model_type="led",
            supports_long_context=True,
            default_chunk_size=16384,  # LED can use full context (no chunking needed)
            default_overlap=1638,  # 10% of 16384 (CHUNK_OVERLAP_RATIO = 0.1)
        ),
        "long-fast": ModelCapabilities(  # Test default
            max_position_embeddings=16384,
            model_type="led",
            supports_long_context=True,
            default_chunk_size=16384,  # Same as long (model type determines chunk size)
            default_overlap=1638,
        ),
        "allenai/led-large-16384": ModelCapabilities(
            max_position_embeddings=16384,
            model_type="led",
            supports_long_context=True,
        ),
        "allenai/led-base-16384": ModelCapabilities(
            max_position_embeddings=16384,
            model_type="led",
            supports_long_context=True,
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
                max_pos = getattr(config, 'max_position_embeddings', None)
                if max_pos:
                    model_type = cls._infer_model_type(config)
                    supports_long = max_pos >= 4096
                    return ModelCapabilities(
                        max_position_embeddings=max_pos,
                        model_type=model_type,
                        supports_long_context=supports_long,
                    )
            except (AttributeError, TypeError):
                pass

        # 3. Pattern-based fallback
        if "led" in model_id.lower() or "longformer" in model_id.lower():
            return ModelCapabilities(
                max_position_embeddings=16384,
                model_type="led",
                supports_long_context=True,
            )
        if "bart" in model_id.lower():
            return ModelCapabilities(
                max_position_embeddings=1024,
                model_type="bart",
                supports_long_context=False,
            )
        if "pegasus" in model_id.lower():
            return ModelCapabilities(
                max_position_embeddings=1024,
                model_type="pegasus",
                supports_long_context=False,
            )

        # 4. Safe default (conservative)
        return ModelCapabilities(
            max_position_embeddings=1024,
            model_type="unknown",
            supports_long_context=False,
        )

    @classmethod
    def _infer_model_type(cls, config: Any) -> str:
        """Infer model type from config."""
        model_type = getattr(config, 'model_type', '').lower()
        if 'bart' in model_type:
            return 'bart'
        if 'led' in model_type or 'longformer' in model_type:
            return 'led'
        if 'pegasus' in model_type:
            return 'pegasus'
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
    getattr(model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS)
    if model.model and hasattr(model.model, "config")
    else BART_MAX_POSITION_EMBEDDINGS
)

# After:
from .model_registry import ModelRegistry
capabilities = ModelRegistry.get_capabilities(model_name, model_instance)
model_max_tokens = capabilities.max_position_embeddings
```

**Replace hardcoded limits in `config.py`:**

```python
# Before:
DEFAULT_MAP_MAX_INPUT_TOKENS = 1024  # BART model limit
DEFAULT_REDUCE_MAX_INPUT_TOKENS = 4096  # LED model limit

# After:
# Keep as defaults, but document they're model-specific
# When model is known, use ModelRegistry.get_capabilities()
```

**Update all dynamic detection sites:**

Replace all instances of:
```python
getattr(model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS)
```

With:
```python
ModelRegistry.get_capabilities(model_name, model_instance).max_position_embeddings
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

### 4. Migration Strategy

**Phase 1: Create Registry**
- Create `model_registry.py` with all current models
- Populate `default_chunk_size` and `default_overlap` from current hardcoded values
- Add comprehensive tests

**Phase 2: Replace in `summarizer.py`**
- Replace all hardcoded `BART_MAX_POSITION_EMBEDDINGS` and `LED_MAX_CONTEXT_WINDOW` references
- Replace `ENCODER_DECODER_TOKEN_CHUNK_SIZE` with registry lookup
- Update dynamic detection logic
- Run tests to verify behavior unchanged

**Phase 3: Update `config.py`**
- Document that defaults are model-specific
- Add comments referencing ModelRegistry for model-specific limits
- Keep Config defaults as fallbacks for unknown models

**Phase 4: Remove Old Constants**
- Remove `BART_MAX_POSITION_EMBEDDINGS` and `LED_MAX_CONTEXT_WINDOW` from `summarizer.py`
- Remove `ENCODER_DECODER_TOKEN_CHUNK_SIZE` (replaced by registry)
- Update any remaining references

**Phase 5: Add Tests**
- Test registry completeness (all models in `DEFAULT_SUMMARY_MODELS` are registered)
- Test default chunk size/overlap values match current behavior
- Test dynamic detection fallback
- Test pattern-based fallbacks
- Test extensibility (register_model)

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

This RFC (RFC-044) complements:

1. **RFC-012: Episode Summarization** - Provides infrastructure for model-agnostic summarization
2. **RFC-029: Provider Refactoring Consolidation** - Supports provider system's goal of pluggable models
3. **RFC-013: OpenAI Provider Implementation** - Can be extended to include API model capabilities

**Key Distinction:**
- **RFC-044**: Focuses on model metadata and architecture limits
- **RFC-012**: Focuses on summarization algorithms and strategies
- **RFC-029**: Focuses on provider architecture and interfaces

Together, these RFCs provide a complete, extensible model system.

## Benefits

1. **Single Source of Truth**: All model limits centralized in one place
2. **Eliminates Hardcoded Values**: Removes scattered magic numbers
3. **Extensibility**: Easy to add new models without code changes
4. **Type Safety**: Structured dataclass prevents errors
5. **Future-Proof**: Supports new model types (T5, LongT5, etc.)
6. **Maintainability**: Easier to update and document model capabilities

## Migration Path

1. **Phase 1**: Create `ModelRegistry` with current models, add tests
2. **Phase 2**: Replace hardcoded limits in `summarizer.py` (backward compatible)
3. **Phase 3**: Update `config.py` documentation, remove old constants
4. **Phase 4**: Verify all tests pass, update documentation
5. **Phase 5**: (Future) Extend registry with additional metadata (memory, device, quality ratings)

## Open Questions

1. Should registry include memory requirements for models?
2. Should registry include recommended devices (CPU/GPU/MPS)?
3. Should we validate model IDs against HuggingFace Hub at runtime?
4. Should registry support model versioning (different limits per version)?
5. **Config File Integration**: Should registry auto-populate `summary_tokenize` limits in user configs based on selected models?
6. **Config Validation**: Should registry validate that user-specified token limits in config files don't exceed model capabilities?

## References

- **Related RFC**: `docs/rfc/RFC-012-episode-summarization.md`
- **Related RFC**: `docs/rfc/RFC-029-provider-refactoring-consolidation.md`
- **Source Code**: `src/podcast_scraper/providers/ml/summarizer.py`
- **Source Code**: `src/podcast_scraper/config.py`
