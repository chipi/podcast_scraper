# Phase 7 (Prompt Management) Implementation Review

**Date**: 2025-12-22  
**Status**: Partially Complete  
**Reference**: `docs/rfc/RFC-017-prompt-management.md`, `docs/wip/INCREMENTAL_MODULARIZATION_PLAN.md` Stage 7

## Summary

Phase 7 (Prompt Management) was partially implemented during Stage 6 (OpenAI Providers). The core `prompt_store.py` module and prompt files were created, and OpenAI providers were updated to use them. However, several components from RFC-017 are still missing.

## ✅ What's Implemented (from Stage 6)

### 1. Core Prompt Store (`prompt_store.py`)

- ✅ File-based prompt loading from `prompts/` directory
- ✅ Jinja2 templating support
- ✅ LRU caching with `@lru_cache`
- ✅ SHA256 hashing (`hash_text()`)
- ✅ Functions: `render_prompt()`, `get_prompt_metadata()`, `get_prompt_source()`
- ✅ Environment variable support (`PROMPT_DIR`)
- ✅ `set_prompt_dir()` and `get_prompt_dir()` helpers
- ✅ `clear_cache()` for testing

### 2. Prompt Directory Structure

- ✅ `prompts/summarization/system_v1.j2`
- ✅ `prompts/summarization/long_v1.j2`
- ✅ `prompts/ner/system_ner_v1.j2`
- ✅ `prompts/ner/guest_host_v1.j2`

### 3. OpenAI Provider Integration

- ✅ `summarization/openai_provider.py` uses `prompt_store.render_prompt()`
- ✅ `speaker_detectors/openai_detector.py` uses `prompt_store.render_prompt()`
- ✅ Providers load prompts from versioned files

## ❌ What's Missing (from RFC-017)

### 1. Experiment Configuration Models (`experiment_config.py`)

**Status**: ❌ Not Created  
**Priority**: High (required for Stage 7)

**Required Components**:

- `PromptConfig` Pydantic model
- `HFBackendConfig` Pydantic model
- `OpenAIBackendConfig` Pydantic model
- `DataConfig` Pydantic model
- `ExperimentParams` Pydantic model
- `ExperimentConfig` Pydantic model
- `load_experiment_config()` function
- `discover_input_files()` helper
- `episode_id_from_path()` helper

**Reference**: RFC-017 Section 3 (lines 278-512)

### 2. Config Fields for Prompts (`config.py`)

**Status**: ❌ Not Added  
**Priority**: Medium (optional per RFC, but mentioned in Stage 7)

**Missing Fields**:
```python
# Summarization prompts
openai_summary_system_prompt: Optional[str] = Field(
    default=None,
    description="System prompt name for summarization (e.g. 'summarization/system_v1')",
)
openai_summary_user_prompt: str = Field(
    default="summarization/long_v1",
    description="User prompt name for summarization",
)
summary_prompt_params: Dict[str, Any] = Field(
    default_factory=dict,
    description="Template parameters for summary prompts",
)

# NER prompts
openai_speaker_system_prompt: Optional[str] = Field(
    default=None,
    description="System prompt name for NER",
)
openai_speaker_user_prompt: str = Field(
    default="ner/guest_host_v1",
    description="User prompt name for NER",
)
ner_prompt_params: Dict[str, Any] = Field(
    default_factory=dict,
    description="Template parameters for NER prompts",
)
```

**Current State**: Providers use `getattr()` with defaults:

- `openai_summary_system_prompt` → `getattr(cfg, "openai_summary_system_prompt", "summarization/system_v1")`
- `openai_summary_user_prompt` → `getattr(cfg, "openai_summary_user_prompt", "summarization/long_v1")`
- `openai_speaker_prompt` → `getattr(cfg, "openai_speaker_prompt", "ner/guest_host_v1")`

**Reference**: RFC-017 Section 6 (lines 816-852)

### 3. Unit Tests for `prompt_store`

**Status**: ❌ Not Created  
**Priority**: High (required for Stage 7)

**Missing Tests**:

- Prompt loading from files
- Template rendering with parameters
- Caching behavior (`lru_cache`)
- SHA256 hashing
- Metadata generation (`get_prompt_metadata()`)
- Error handling (`PromptNotFoundError`)
- Environment variable support (`PROMPT_DIR`)
- `set_prompt_dir()` and `clear_cache()`

**Current State**: Only mocks in `tests/test_openai_providers.py`:

- `@patch("podcast_scraper.prompt_store.render_prompt")` used to mock prompt rendering
- No direct tests of `prompt_store` functionality

**Reference**: RFC-017 Section "Testing Strategy" (lines 980-985), Stage 7 Deliverable 5

### 4. Prompt Metadata Tracking in Results

**Status**: ❌ Not Implemented  
**Priority**: Medium (mentioned in RFC but not critical for basic functionality)

**Missing**: Providers don't call `get_prompt_metadata()` to track prompt versions in results.

**Current State**: Providers return basic metadata but don't include prompt metadata:
```python
# Current (summarization/openai_provider.py)
"metadata": {
    "model": self.model,
    "provider": "openai",
    "max_length": max_length,
    "min_length": min_length,
}
```

**Expected** (per RFC-017):
```python
from ..prompt_store import get_prompt_metadata

"metadata": {
    "model": self.model,
    "provider": "openai",
    "prompts": {
        "system": get_prompt_metadata(system_prompt_name),
        "user": get_prompt_metadata(user_prompt_name, params),
    }
}
```

**Reference**: RFC-017 Section 7 (lines 854-900)

## Implementation Priority

### High Priority (Required for Stage 7 Completion)

1. **Create `experiment_config.py`** - Required for experiment pipeline (Stage 8)
2. **Add unit tests for `prompt_store`** - Required for Stage 7 success criteria

### Medium Priority (Nice to Have)

3. **Add config fields for prompts** - Improves usability but providers work with `getattr()` defaults
4. **Add prompt metadata tracking** - Improves reproducibility but not critical for basic functionality

## Recommendations

### Option 1: Complete Phase 7 Now

Implement all missing components:

- Create `experiment_config.py` with all models
- Add comprehensive unit tests for `prompt_store`
- Add config fields for prompts (optional but recommended)
- Add prompt metadata tracking (optional but recommended)

**Effort**: 1-2 days  
**Benefit**: Complete Phase 7, ready for Stage 8 (AI Experiment Pipeline)

### Option 2: Defer to Stage 8

Only implement what's needed for Stage 8:

- Create `experiment_config.py` (required for Stage 8)
- Add basic unit tests for `prompt_store` (good practice)

**Effort**: 0.5-1 day  
**Benefit**: Minimal implementation, defer optional features

### Option 3: Status Quo

Leave as-is since providers work correctly:

- Providers use `prompt_store` successfully
- Missing pieces are for experiment pipeline (Stage 8)

**Effort**: 0 days  
**Benefit**: No changes needed, but incomplete Phase 7

## Conclusion

Phase 7 is **functionally complete** for production use (providers work correctly), but **incomplete** per RFC-017 and Stage 7 deliverables. The missing pieces (`experiment_config.py` and tests) are primarily needed for Stage 8 (AI Experiment Pipeline).

**Recommendation**: Implement Option 1 (complete Phase 7) to ensure all Stage 7 deliverables are met before moving to Stage 8.
