# RFC-010: Automatic Speaker Name Detection

- **Status**: Completed
- **Authors**: GPT-5 Codex
- **Stakeholders**: Maintainers, Whisper integration owners, CLI users
- **Related PRDs**: `docs/prd/PRD-008-speaker-name-detection.md` (primary), `docs/prd/PRD-001-transcript-pipeline.md`, `docs/prd/PRD-002-whisper-fallback.md`

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
- spaCy is a required dependency for speaker detection functionality.
- Manual speaker names (`--speaker-names`) are ONLY used as fallback when automatic detection fails (not as override).
- Manual names format: first item = host, second item = guest (e.g., `["Lenny", "Guest"]`).
- When guest detection fails: keep detected hosts (if any) + use manual guest name as fallback.
- Default fallback remains `["Host", "Guest"]` when extraction fails and no manual names are provided.
- Whisper transcription and screenplay formatting work independently of speaker detection; detected names enhance screenplay output when available.

## Design & Implementation

1. **Dependency & Configuration**
   - Add spaCy to runtime dependencies and expose configurable model selection driven by language:

```python
     - `cfg.language` (default `"en"`) governing both Whisper transcription language and NER pipeline.
     - `cfg.ner_model` optional override; default model derived from language (e.g., `"en_core_web_sm"`).
```

   - CLI gains `--language` (reused by Whisper) and `--ner-model` flags; config supports `language`, `ner_model`, and `auto_speakers` toggle (default `true`).
   - Maintain compatibility with existing Whisper language handling; ensure screenplay formatter respects detected language.

2. **Whisper Model Selection Based on Language**
   - Per [Whisper documentation](https://github.com/openai/whisper), English-only models (`.en` variants) exist for `tiny`, `base`, `small`, and `medium` sizes and perform better for English-only applications, especially for `tiny.en` and `base.en`.
   - **Model selection logic**:

```text
     - When `cfg.language` is `"en"` or `"English"`:
       - If user specifies `whisper_model` as `tiny`, `base`, `small`, or `medium`, automatically prefer the `.en` variant (e.g., `base` → `base.en`).
       - For `large` and `turbo` models, no `.en` variant exists; use multilingual versions as-is.
     - When `cfg.language` is any other language:
       - Always use multilingual models (no `.en` suffix), regardless of model size.
     - This selection happens transparently in `whisper.load_whisper_model()`; users can still explicitly request `.en` models via `--whisper-model base.en` if desired.
```

   - **Language parameter propagation**:

```text
     - The `cfg.language` value is passed to `whisper_model.transcribe()` as the `language` parameter, enabling proper language-specific transcription and improving accuracy for non-English content.
     - Default behavior (when language not specified) remains English (`"en"`) for backwards compatibility.
```

3. **Extraction Pipeline**
   - **Host Detection** (feed-level only):

```python
     - **Priority 1**: Extract host names from RSS author tags (channel-level only).
       - RSS 2.0 `<author>` tag (channel-level, should be single author).
       - iTunes `<itunes:author>` tag (channel-level, can help confirm host).
       - iTunes `<itunes:owner><itunes:name>` tag (channel-level, can help confirm host).
       - Author tags are the most reliable source as they explicitly specify the podcast host(s).
       - Only extract from channel-level tags, not from individual episode items.
       - Clean author names by removing email addresses if present (format: "Name <email@example.com>").
       - If author tags are found, use them directly without NER or validation.
       - Multiple author sources (author/itunes:author/itunes:owner) are collected together to confirm hosts.
     - **Priority 2**: If no author tags exist, fall back to NER extraction from feed title/description.
       - Extract PERSON entities from feed title and feed description using spaCy NER.
       - Validate detected hosts by checking if they also appear in the first episode's title/description.
       - Only hosts that appear in both feed metadata AND first episode are kept (validation step).
     - Hosts are cached and reused across all episodes in a run.
```python

   - **Guest Detection** (episode-level only):
     - Extract PERSON entities from episode title and first 20 characters of episode description.
     - Descriptions are limited to first 20 characters to focus on the most relevant part (often contains guest name).
     - Guests are episode-specific and should NEVER be extracted from feed metadata.
     - Remove any detected hosts from the episode persons list to get pure guests.
     - Each episode's guests are detected independently.
     - **Name Sanitization**: All extracted person names are sanitized to remove non-letter characters:

```python
       - Parentheses and their contents: `"John (Smith)"` → `"John"`
       - Trailing punctuation: `"John,"` → `"John"`
       - Leading punctuation
       - Non-letter characters except spaces, hyphens, and apostrophes
       - Whitespace normalization (multiple spaces → single space)
     - **Deduplication**: Names are deduplicated case-insensitively to avoid duplicates like `"John"` and `"john"`.
     - **Pattern-based Fallback**: If NER fails to detect entities, a pattern-based fallback extracts names from segments after common separators (`|`, `—`, `–`, `-`):
       - Splits title on separators and extracts the last segment
       - Matches pattern: 2-3 words, each starting with capital letter
       - Filters out common non-name phrases (e.g., "Guest", "Host", "Interview")
       - Adds candidates with lower confidence (0.7) compared to NER-based detection (1.0)
```python

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

- **Host Detection**:
  - **Preferred**: Extract from RSS author tags (channel-level only).
    - RSS 2.0 `<author>` tag (channel-level, single author).
    - iTunes `<itunes:author>` tag (channel-level, can confirm host).
    - iTunes `<itunes:owner><itunes:name>` tag (channel-level, can confirm host).
    - Author tags explicitly specify hosts and are most reliable.
    - Only extract from channel-level tags, never from episode items.
    - Clean email addresses from author tags if present.
    - Multiple sources are collected together to confirm/validate hosts.
    - No validation needed when using author tags.
  - **Fallback**: Extract PERSON entities from feed-level metadata (feed title/description) using NER.
    - Validate hosts by requiring they also appear in the first episode's metadata.
    - This ensures hosts are truly recurring speakers, not one-time mentions.
  - Hosts are cached and reused across all episodes.
- **Guest Detection**:
  - Extract PERSON entities from episode title and first 20 characters of episode description.
  - Descriptions are limited to first 20 characters to focus on the most relevant part (often contains guest name).
  - NEVER extract guests from feed metadata (they are episode-specific).
  - Remove detected hosts from episode persons to get pure guests.
  - Each episode's guests are detected independently.
  - **Name Sanitization**: All extracted person names are sanitized:

```text
    - Removes parentheses and their contents: `"John (Smith)"` → `"John"`
    - Removes trailing/leading punctuation (commas, periods, etc.)
    - Removes non-letter characters except spaces, hyphens, and apostrophes
    - Normalizes whitespace
    - Validates: must be at least 2 characters and contain at least one letter
```

  - **Deduplication**: Names are deduplicated case-insensitively to avoid duplicates.
  - **Pattern-based Fallback**: When NER fails (e.g., spaCy misclassifies names or fails on long titles), a pattern-based fallback:

```text
    - Splits title on common separators (`|`, `—`, `–`, `-`)
    - Extracts the last segment (often contains guest name)
    - Matches pattern: 2-3 words, each starting with capital letter
    - Filters out common non-name phrases
    - Adds candidates with confidence 0.7 (lower than NER-based 1.0)
```

- **Fallback Behavior**:
  - If zero guests detected, preserve host-only labels (`Host`, `Co-Host`, etc.).
  - If zero hosts detected, fall back to default `["Host", "Guest"]`.
  - **Manual Speaker Names Fallback**: Manual speaker names (configured via `--speaker-names` or `speaker_names` in config) are ONLY used as a fallback when automatic detection fails:

```text
    - Manual names format: first item = host, second item = guest (e.g., `["Lenny", "Guest"]`).
    - When guest detection fails for an episode:
      - If hosts were detected: keep detected hosts + use manual guest name (second item) as fallback.
      - If no hosts detected: use both manual names (host + guest).
    - Manual names are never used to override successful detection; they only activate when detection fails.
    - This ensures detected hosts are preserved across episodes while allowing manual guest fallback per episode when needed.
```

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

## Implementation Details

### Name Sanitization

- All extracted person names are sanitized using `_sanitize_person_name()`:
  - Removes parentheses and their contents: `"John (Smith)"` → `"John"`
  - Removes trailing punctuation: `"John,"` → `"John"`
  - Removes leading punctuation
  - Removes non-letter characters except spaces, hyphens, and apostrophes
  - Normalizes whitespace (multiple spaces → single space)
  - Validates: must be at least 2 characters and contain at least one letter

### Deduplication

- Names are deduplicated case-insensitively using `seen_sanitized_names` set
- Prevents duplicates like `"John"` and `"john"` from appearing in candidate list

### Pattern-based Fallback

- When NER fails to detect PERSON entities (e.g., spaCy misclassifies as ORG), a pattern-based fallback:
  1. Splits title on common separators (`|`, `—`, `–`, `-`)
  2. Extracts the last segment (often contains guest name)
  3. Matches pattern: `^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$` (2-3 words, each starting with capital)
  4. Filters out common non-name phrases: `{'guest', 'host', 'episode', 'title', 'interview', 'conversation'}`
  5. Adds candidates with confidence 0.7 (lower than NER-based 1.0)
  6. Logs at DEBUG level: `"Pattern-based fallback: extracted 'Name' from last segment 'Segment'"`

### Description Limiting

- Episode descriptions are limited to first 20 characters before NER processing
- Focuses on the most relevant part (often contains guest name at the start)
- Reduces noise from longer descriptions and improves processing speed

## Open Questions

- Should we persist detected names into output metadata (e.g., JSON sidecar) for downstream tools?
- Do we need language detection to select an alternate spaCy model automatically?
- What is the acceptable performance impact for large feeds (e.g., 500 episodes)?

## References

- spaCy documentation: <https://spacy.io/usage/models>
- Whisper GitHub repository: <https://github.com/openai/whisper> (model selection and language support)
- Whisper screenplay formatting logic: `podcast_scraper/whisper_integration.py`
- Existing RFCs: `docs/rfc/RFC-005-whisper-integration.md`, `docs/rfc/RFC-006-screenplay-formatting.md`
- Language handling reference: `podcast_scraper/config.py` (existing Whisper language options)
