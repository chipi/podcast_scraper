# Issue #371 / RFC-044 Model Registry — Analysis and Plan

**Issue:** [Implement RFC-044: Model Registry with Baseline Promotion Mechanism](https://github.com/chipi/podcast_scraper/issues/371)  
**RFC:** [RFC-044: Model Registry for Architecture Limits](../rfc/RFC-044-model-registry.md)  
**Date:** 2026-02-12

## Summary

Implement the centralized Model Registry (RFC-044) to remove hardcoded model limits and add a baseline→mode promotion path. The issue lists 8 phases; this doc validates the approach against the current codebase and defines a concrete execution plan.

## Current State (Validated)

### Hardcoded limits

- **summarizer.py**
  - `BART_MAX_POSITION_EMBEDDINGS = 1024` (line 189)
  - `LED_MAX_CONTEXT_WINDOW = 16384` (line 190)
  - `ENCODER_DECODER_TOKEN_CHUNK_SIZE = 600` (lines 203–205)
  - Used at: 2058 (default arg), 2190–2192, 2888–2890, 2914–2917, 3711–3713, 3853
  - Plus `LED_ATTENTION_WINDOW = 1024` (line 2883) — keep as LED-specific constant unless RFC extends
- **ml_provider.py**
  - `chunk_size=chunk_size or summarizer.BART_MAX_POSITION_EMBEDDINGS` (line 1369)
- **config.py**
  - `DEFAULT_MAP_MAX_INPUT_TOKENS = 1024`, `DEFAULT_REDUCE_MAX_INPUT_TOKENS = 4096` (lines 441–442)
- **config_constants.py**
  - `DEFAULT_SUMMARY_CHUNK_SIZE = 2048` (line 258) — docstring references BART 1024

### Model surface

- **DEFAULT_SUMMARY_MODELS** (summarizer.py): aliases → full IDs; includes `pegasus-cnn` → `google/pegasus-cnn_dailymail` (production MAP). All must be in registry (alias + full ID).
- **PROD_DEFAULT_*** (config_constants.py): `PROD_DEFAULT_SUMMARY_MODEL` = `google/pegasus-cnn_dailymail`, `PROD_DEFAULT_SUMMARY_REDUCE_MODEL` = LED-base; replaced by mode config in later phases.

### Baselines

- `data/eval/baselines/baseline_ml_dev_authority_smoke_v1/` has `config.yaml` with: `map_model: bart-small`, `reduce_model: long-fast`, `preprocessing_profile: cleaning_v4`, `map_params`, `reduce_params`, `tokenize`, `chunking`. Structure matches RFC `ModeConfiguration` (Phase 3).

## Approach Validation

- **RFC vs code:** RFC-044 design matches current usage: single registry, fallback order (registry → dynamic → pattern → 1024 default), frozen `ModelCapabilities`, optional `model_instance` in `get_capabilities`. Line numbers in RFC are outdated; replacement sites above are accurate.
- **Registry contents:** Phase 1 must include every key in `DEFAULT_SUMMARY_MODELS` and their resolved full IDs, plus `google/pegasus-cnn_dailymail` (in code but not in RFC’s example table). RFC also specifies FLAN-T5, Embedding, QA, NLI for future use; we include them so the registry is ready for RFC-042/049.
- **Backward compatibility:** Replacing constants with `ModelRegistry.get_capabilities(model_id, model_instance).max_input_tokens` (and default_chunk_size where applicable) preserves behavior if registry entries match current values.
- **Decoupling:** Promotion script reads `data/eval/baselines/` and writes into `model_registry.py`; app code only imports the registry, never `data/eval/` — as required.

## Clarifications (Optional)

1. **PR scope:** Prefer one PR per phase (e.g. Phase 1+2 first), or a single PR for Phases 1–4?
2. **First promoted mode:** Use `baseline_ml_dev_authority_smoke_v1` → `ml_small_authority` as in the RFC example, or a different baseline/mode_id?
3. **LED_ATTENTION_WINDOW (1024):** Leave in summarizer as implementation detail, or add to `ModelCapabilities` (e.g. optional `attention_window`)?

If no answer: proceeding with smaller PRs (Phase 1+2 first), first promotion = `baseline_ml_dev_authority_smoke_v1` → `ml_small_authority`, and keeping `LED_ATTENTION_WINDOW` in summarizer unless we extend the RFC.

## Execution Plan

### Phase 1: Create Registry (Model Capabilities)

1. Add `src/podcast_scraper/providers/ml/model_registry.py`:
   - `ModelCapabilities` dataclass (frozen), per RFC (all fields).
   - `ModelRegistry` with `_registry: Dict[str, ModelCapabilities]`:
     - All aliases and full IDs from `DEFAULT_SUMMARY_MODELS` (including `google/pegasus-cnn_dailymail`).
     - BART, PEGASUS, LED entries with correct `max_input_tokens`, `default_chunk_size`, `default_overlap`, `memory_mb` where known.
     - FLAN-T5, Embedding, QA, NLI entries as in RFC (for future use).
   - `get_capabilities(model_id, model_instance=None)` with order: registry → dynamic (from `model_instance`) → pattern → safe default 1024.
   - `_infer_model_type(config)`, `_infer_model_family(model_id, model_type)`.
   - `register_model(model_id, capabilities)`.
2. Export from `providers/ml/__init__.py` if desired.
3. Unit tests in `tests/unit/podcast_scraper/providers/ml/test_model_registry.py`: registry lookup for every alias/full ID in `DEFAULT_SUMMARY_MODELS`, pattern fallbacks, dynamic detection (mocked config), safe default, `register_model`, and completeness (all DEFAULT_SUMMARY_MODELS keys resolve).

### Phase 2: Replace Hardcoded Limits in summarizer.py and ml_provider.py

1. In **summarizer.py**:
   - Replace `BART_MAX_POSITION_EMBEDDINGS` / `LED_MAX_CONTEXT_WINDOW` at each use site with `ModelRegistry.get_capabilities(model_name, model_instance).max_input_tokens` (or equivalent) where `model_name`/`model_instance` are available; otherwise use capabilities for the current model in context.
   - Replace `ENCODER_DECODER_TOKEN_CHUNK_SIZE` with registry-derived default (e.g. `caps.default_chunk_size or 600`) when model is known.
   - Remove the three constants only after all references are switched (can be same or follow-up commit).
2. In **ml_provider.py**:
   - Replace `summarizer.BART_MAX_POSITION_EMBEDDINGS` with registry lookup (e.g. by summary model from config).
3. Run full test suite; fix any regressions (e.g. tests that patch or depend on old constants).

### Phase 3: Mode Configuration and Promotion (Next PR)

- Add `ModeConfiguration` dataclass and `_mode_registry` to `ModelRegistry`; implement `get_mode_configuration(mode_id)`.
- Add `scripts/registry/promote_baseline.py` (read baseline config + optional metrics, validate, append to `_mode_registry` in code).
- Add Make target `registry-promote BASELINE_ID=... MODE_ID=...`.
- Tests for promotion script and mode lookup.

### Phases 4–8

- Phase 4: Promote first baseline (`baseline_ml_dev_authority_smoke_v1` → `ml_small_authority`).
- Phase 5: App code uses modes (replace `PROD_DEFAULT_*` with mode lookups, Config support, fingerprint logging).
- Phase 6: Docs (config.py docstrings, promotion workflow, EXPERIMENT_GUIDE).
- Phase 7: Remove `PROD_DEFAULT_*` and remaining hardcoded constants.
- Phase 8: Full testing and validation (completeness, fallbacks, promotion, fingerprint).

## Acceptance (from Issue)

- [ ] All hardcoded model limits removed from codebase  
- [ ] Registry contains all models in `DEFAULT_SUMMARY_MODELS` and structure supports all 6 families  
- [ ] Promotion script runs and updates registry  
- [ ] App code can use mode configurations without importing `data/eval/`  
- [ ] Runtime fingerprint logging works  
- [ ] All tests pass; no performance regression; docs updated  

## References

- Issue: [#371](https://github.com/chipi/podcast_scraper/issues/371)
- RFC: [RFC-044-model-registry.md](../rfc/RFC-044-model-registry.md)
- Related: #369 (preprocessing profiles), #370 (cleaning_v4)
