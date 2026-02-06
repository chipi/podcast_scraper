# ADR-047: Centralized Model Registry

- **Status**: Proposed
- **Date**: 2026-02-05
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-044](../rfc/RFC-044-model-registry.md), [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md)

## Context & Problem Statement

Model architecture limits (e.g., `1024` for BART, `16384` for LED) are currently hardcoded throughout the codebase:

- `BART_MAX_POSITION_EMBEDDINGS = 1024` in `summarizer.py:50`
- `LED_MAX_CONTEXT_WINDOW = 16384` in `summarizer.py:51`
- `DEFAULT_MAP_MAX_INPUT_TOKENS = 1024` in `config.py:166`
- Dynamic detection fallbacks to hardcoded values in multiple places

This leads to:

- **Maintenance Burden**: Adding new models requires updating hardcoded values in multiple files
- **Inconsistency Risk**: Limits can drift out of sync between files
- **No Single Source of Truth**: Model capabilities are scattered and undocumented
- **Error-Prone**: Easy to use wrong limits for new or unknown models
- **Limited Extensibility**: Hard to add new model types without code changes

## Decision

We adopt a **Centralized Model Registry** to store model architecture limits and capabilities.

1. **Model Registry**: Single source of truth for all model architecture limits stored in `src/podcast_scraper/providers/ml/model_registry.py`.
2. **ModelCapabilities Dataclass**: Structured, type-safe model capability information (max context window, model type, etc.).
3. **O(1) Lookup**: Registry provides fast lookup by model ID/alias.
4. **Pattern-Based Fallbacks**: Intelligent guessing for unknown models (e.g., "bart-*" → BART limits).
5. **Safe Defaults**: Conservative defaults for unknown models.
6. **Extensibility**: Runtime registration for custom models.

## Rationale

- **Single Source of Truth**: Model capabilities are documented and maintained in one place
- **Eliminates Hardcoded Values**: Removes scattered hardcoded limits throughout codebase
- **Maintainability**: Adding new models requires updating one place, not multiple files
- **Consistency**: Limits can't drift out of sync
- **Extensibility**: New models can be registered without code changes
- **Type Safety**: Structured, type-safe model capability information

## Alternatives Considered

1. **Keep Hardcoded Values**: Rejected as it leads to maintenance burden and inconsistency.
2. **Dynamic Detection Only**: Rejected as it requires model loading and doesn't work for unknown models.
3. **Configuration Files**: Rejected as it adds complexity and doesn't provide type safety.

## Consequences

- **Positive**:
  - Single source of truth for model capabilities
  - Eliminates hardcoded values throughout codebase
  - Easy to add new models
  - Type-safe model capability information
  - Pattern-based fallbacks for unknown models
- **Negative**:
  - Initial implementation complexity
  - Requires migration of existing hardcoded values
- **Neutral**:
  - Requires implementation of RFC-044

## Implementation Notes

- **Module**: `src/podcast_scraper/providers/ml/model_registry.py` - Model registry
- **Pattern**: Registry pattern with pattern-based fallbacks
- **ModelCapabilities**: Max context window, model type (BART, LED, T5, etc.), aliases
- **Lookup**: O(1) lookup by model ID/alias
- **Fallbacks**: Pattern-based guessing (e.g., "bart-*" → BART limits, "led-*" → LED limits)
- **Defaults**: Conservative defaults for unknown models (e.g., 512 tokens)
- **Extensibility**: Runtime registration via `register_model()` function
- **Model-Agnostic**: Handles both test and production models identically
- **Status**: 🟡 Draft RFC (RFC-044) - Not yet implemented

## References

- [RFC-044: Model Registry for Architecture Limits](../rfc/RFC-044-model-registry.md)
- [RFC-029: Provider Refactoring Consolidation](../rfc/RFC-029-provider-refactoring-consolidation.md) - Related architecture
