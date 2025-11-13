# RFC-010: Automatic Speaker Name Detection

- **Status**: Draft
- **Authors**: GPT-5 Codex
- **Stakeholders**: Maintainers, Whisper integration owners, CLI users
- **Related PRDs**: `docs/prd/PRD-001-transcript-pipeline.md`, `docs/prd/PRD-002-whisper-fallback.md`

## Abstract

Introduce automatic speaker name detection so Whisper transcriptions can label voices with real host/guest names without requiring manual CLI configuration. The feature relies on spaCy-powered named entity recognition (NER) to extract PERSON entities from RSS metadata (title, description, feed details) and feeds those names into the screenplay formatter and logging pipeline.

## Problem Statement

Currently transcripts default to `Host` / `Guest` unless the user provides `--speaker-names`. That manual step:

- Requires feed-specific knowledge
- Breaks when episodes have unique guests
- Adds friction when running at scale or embedding the pipeline

Automating speaker name extraction improves UX, increases transcript quality, and aligns with the product goal of frictionless transcript generation.

## Constraints & Assumptions

- Primary language controlled by configuration (default `"en"`). spaCy model selection is driven by the language setting with `en_core_web_sm` as the default.
- spaCy must be an optional dependency that is installable via existing packaging flow.
- CLI overrides (`--speaker-names`) take precedence over auto-detection.
- Default fallback remains `["Host", "Guest"]` when extraction fails.
- Whisper transcription or screenplay formatting must still work without spaCy (e.g., when dependency missing).

## Design & Implementation

1. **Dependency & Configuration**
   - Add spaCy to runtime dependencies and expose configurable model selection driven by language:
     - `cfg.language` (default `"en"`) governing both Whisper transcription language and NER pipeline.
     - `cfg.ner_model` optional override; default model derived from language (e.g., `"en_core_web_sm"`).
   - CLI gains `--language` (reused by Whisper) and `--ner-model` flags; config supports `language`, `ner_model`, and `auto_speakers` toggle (default `true`).
   - Maintain compatibility with existing Whisper language handling; ensure screenplay formatter respects detected language.

2. **Whisper Model Selection Based on Language**
   - Per [Whisper documentation](https://github.com/openai/whisper), English-only models (`.en` variants) exist for `tiny`, `base`, `small`, and `medium` sizes and perform better for English-only applications, especially for `tiny.en` and `base.en`.
   - **Model selection logic**:
     - When `cfg.language` is `"en"` or `"English"`:
       - If user specifies `whisper_model` as `tiny`, `base`, `small`, or `medium`, automatically prefer the `.en` variant (e.g., `base` → `base.en`).
       - For `large` and `turbo` models, no `.en` variant exists; use multilingual versions as-is.
     - When `cfg.language` is any other language:
       - Always use multilingual models (no `.en` suffix), regardless of model size.
     - This selection happens transparently in `whisper.load_whisper_model()`; users can still explicitly request `.en` models via `--whisper-model base.en` if desired.
   - **Language parameter propagation**:
     - The `cfg.language` value is passed to `whisper_model.transcribe()` as the `language` parameter, enabling proper language-specific transcription and improving accuracy for non-English content.
     - Default behavior (when language not specified) remains English (`"en"`) for backwards compatibility.

3. **Extraction Pipeline**
   - **Step order**: episode title → episode description → feed title/description (for host inference).
   - Parse PERSON entities via spaCy NER using the configured language/model pair.
   - Maintain two buckets:
     - **Hosts**: recurring names detected across feed-level metadata/episodes.
     - **Guests**: per-episode PERSON names minus identified hosts.

4. **Caching Strategy**
   - Introduce a feature flag (`cfg.cache_detected_hosts`) controlling whether host detection is memoized across episodes within a run.
   - Provide both code paths (cached vs. per-episode) to allow benchmarking; default can start with caching enabled.

5. **Integration Points**
   - Extend `models.Episode` or attach metadata with detected guest list.
   - Whisper transcription: when screenplay formatting is enabled, inject `speaker_names` derived from detection unless CLI overrides exist; align transcription language with the configured feed language to preserve accent/locale expectations.
   - Logging/metadata: emit info-level summaries of detected speaker lists for visibility.

6. **Failure Modes**
   - If spaCy model missing: warn once and fall back to defaults.
   - If NER returns >N names, cap at configured limit (default 4) to avoid noise.

## Heuristics

- Treat PERSON entities found in feed-level metadata as hosts (up to a configurable max).
- For episodes, subtract the host set; remaining names become guests sorted by appearance order.
- If zero guests detected, preserve host-only labels (`Host`, `Co-Host`, etc.).
- Allow manual host list override in config (`known_hosts`) to bias classification.

## Feature Flags & Experiments

- `auto_speakers` (bool, default `true`): master switch for automatic detection.
- `cache_detected_hosts` (bool, default `true`): toggles host memoization. Document benchmarking plan in RFC to compare runtime.
- `language_override_enabled` (bool, default `false`): allows experimenting with alternative language auto-detection strategies before making them default.

## Testing Strategy

- Unit tests with synthetic RSS samples covering:
  - Title-only detection (`"Alice interviews Bob"`).
  - Description-rich detection (multiple guest names).
  - Feed-level host inference.
  - CLI override precedence.
  - spaCy missing/disabled scenarios.
- Integration smoke test ensuring Whisper screenplay uses detected names end-to-end (gated to skip when spaCy unavailable).

## Rollout Plan

1. Land RFC + implementation behind configuration defaults.
2. Update README/config docs to describe new options.
3. Release minor version noting new dependency.
4. Collect user feedback, adjust heuristics or caching default.

## Open Questions

- Should we persist detected names into output metadata (e.g., JSON sidecar) for downstream tools?
- Do we need language detection to select an alternate spaCy model automatically?
- What is the acceptable performance impact for large feeds (e.g., 500 episodes)?

## References

- spaCy documentation: <https://spacy.io/usage/models>
- Whisper GitHub repository: <https://github.com/openai/whisper> (model selection and language support)
- Whisper screenplay formatting logic: `podcast_scraper/whisper.py`
- Existing RFCs: `docs/rfc/RFC-005-whisper-integration.md`, `docs/rfc/RFC-006-screenplay-formatting.md`
- Language handling reference: `podcast_scraper/config.py` (existing Whisper language options)
