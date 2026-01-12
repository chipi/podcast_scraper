# Release v2.4.0 - Production Readiness & Infrastructure

**Release Date:** January 2026
**Type:** Minor Release

## Summary

v2.4.0 is a **major minor release** introducing **production-ready configuration defaults**, **advanced cache management**, and **significant quality improvements** to summarization. This release focuses on stability, organization, and performance, providing a solid foundation for future expansion.

## 🚀 Key Features

### ⚙️ Production-Ready Configuration Defaults

**Improved defaults for production use:**

#### Automatic Transcription (New Default)

- **Changed**: `transcribe_missing` now defaults to `true`
- **Impact**: Episodes without transcripts are automatically transcribed
- **Rationale**: Reduces friction for podcast ingestion workflows
- **Override**: Use `--no-transcribe-missing` to restore old behavior

```bash

# v2.3.2 (old behavior)

python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing

# v2.4.0 (new default)

python3 -m podcast_scraper.cli https://example.com/feed.xml  # Automatic!

# Disable if needed

python3 -m podcast_scraper.cli https://example.com/feed.xml --no-transcribe-missing
```python

## Whisper Model Default (base.en)

- **Changed**: Default Whisper model changed from `base` to `base.en`
- **Impact**: Better accuracy for English podcasts, faster processing
- **Rationale**: Most users process English content; `.en` models are optimized
- **Override**: Use `--whisper-model base` for multilingual support

```yaml

# config.yaml

whisper_model: base.en  # New default (was: base)
```

## 🗂️ Advanced Cache Management

**Comprehensive cache management CLI:**

### Cache Status Command

```bash

# View cache status for all ML models

python3 -m podcast_scraper.cli cache --status

# Example output:
# ML Model Cache Status
# ==================================================
# Whisper models: 2.5 GB (3 models)
# Transformers: 8.2 GB (5 models)
# spaCy: 150 MB (1 model)
# Total: 10.85 GB

```

## Cache Cleaning Command

```bash

# Clean all ML caches (interactive)

python3 -m podcast_scraper.cli cache --clean

# Clean specific cache

python3 -m podcast_scraper.cli cache --clean whisper
python3 -m podcast_scraper.cli cache --clean transformers
python3 -m podcast_scraper.cli cache --clean spacy

# Non-interactive (for scripts/automation)

python3 -m podcast_scraper.cli cache --clean --yes
```

## Makefile Integration

```bash

# Show cache status

make cache-status

# Clean all caches

make cache-clean

# Clean individual caches

make cache-clean-whisper
make cache-clean-transformers
make cache-clean-spacy
```

## 🎯 Summarization Quality Improvements

**Significant quality enhancements for ML summarization:**

### Model-Specific Threshold Tuning (Issue #283)

**Problem Fixed**: Episodes at 3-4k tokens produced verbose summaries with warnings, while episodes >4k tokens produced cleaner summaries (backwards behavior).

**Solution**: Implemented model-specific thresholds for LED vs BART:

- **LED Models** (long-context):
  - Ceiling: 6,000 tokens (was 4,000)
  - Validation: 75% (was 60%)
- **BART Models** (short-context):
  - Ceiling: 4,000 tokens (unchanged)
  - Validation: 60% (unchanged)

**Impact:**

- ✅ Reduced false warning frequency by ~40%
- ✅ More consistent quality across episode lengths
- ✅ Better handling of 3-6k token episodes

#### Increased Token Limits

- **Chunk summaries**: 80-160 tokens (was 60-100)
- **Final summaries**: 200-480 tokens (was 150-300)

**Impact:** Summaries retain more detail and context while remaining concise.

### 📊 Output Directory Reorganization

**Improved output structure for better organization:**

#### New Directory Structure

```text
output/
└── rss_feeds.example.com_abc123/
    └── run_my_run_id/
        ├── transcripts/          # NEW: Separate transcripts directory
        │   ├── 001_episode_title.txt
        │   ├── 001_episode_title.cleaned.txt
        │   └── 002_episode_title.txt
        └── metadata/             # NEW: Separate metadata directory
            ├── 001_episode_title.metadata.json
            └── 002_episode_title.metadata.json
```

- **Filtering**: Easier to list only transcripts or only metadata
- **Tooling**: Simpler glob patterns for processing
- **Clarity**: Obvious which files are which

**Migration:**

- ✅ Fully backward compatible (reads both old and new structures)
- ✅ New runs automatically use new structure
- ✅ Existing output directories work unchanged

### 🔧 Run ID Suffix Enhancements

**Automatic run ID suffix generation for better tracking:**

#### Suffix Format

```text
run_{base_run_id}_{whisper_model}_{transformers}_{summarizer}_{speaker_detector}
```

Example: `run_2.4.0_w_base.en_tf_bart-large-cnn_r_led-base-16384_sp_spacy_sm`

- `w_{model}` - Whisper model (e.g., `w_base.en`, `w_small.en`)
- `tf_{model}` - Transformers MAP model (e.g., `tf_bart-large-cnn`)
- `r_{model}` - REDUCE model (e.g., `r_led-base-16384`)
- `sp_{model}` - spaCy model (e.g., `sp_spacy_sm`, `sp_spacy_lg`)

## 🧪 Test Infrastructure Improvements

**Comprehensive test coverage and stability enhancements:**

### Test Pyramid Enhancement

- **Unit Tests**: ~250 tests (fast, isolated, no network/filesystem)
- **Integration Tests**: ~100 tests (moderate, component interaction)
- **E2E Tests**: ~50 tests (slow, full pipeline, real fixtures)

#### Network Isolation (Unit Tests)

- **Automatic Blocking**: All network requests blocked in unit tests using `pytest-socket`.
- **Impact**: Prevents accidental network calls, faster execution, more reliable tests.

#### Fixture Enhancements

**New Long-Form Test Fixtures:**

- **p07** (Sustainability) - 14.5k words (tests hierarchical reduce)
- **p08** (Solar Energy) - 19k words (tests extractive fallback)
- **p09** (Biohacking) - 3 episodes (tests threshold boundaries)

## Technical Details

### Provider Matrix

| Capability | Local | OpenAI |
| ------------ | ------- | -------- |
| Transcription | ✅ | ✅ |
| Speaker Detection | ✅ | ✅ |
| Summarization | ✅ | ✅ |

### Configuration Changes

**New Configuration Fields:**

```yaml

# Provider selection

transcription_provider: whisper      # or openai
speaker_detector_provider: spacy     # or openai
summary_provider: transformers       # or openai

# Parallelism control

transcription_parallelism: 1         # Episodes to transcribe in parallel
processing_parallelism: 2            # Episodes to process in parallel

# OpenAI configuration

openai_api_key: null                 # Set via OPENAI_API_KEY env var
openai_transcription_model: whisper-1
openai_speaker_model: gpt-4o-mini
openai_summary_model: gpt-4o-mini
openai_temperature: 0.3

# Summarization Model Tuning

summary_model: facebook/bart-large-cnn
summary_reduce_model: allenai/led-large-16384

# Changed defaults

transcribe_missing: true             # Was: false (NEW DEFAULT)
whisper_model: base.en               # Was: base (NEW DEFAULT)
```

## CLI Changes

**New Commands:**

```bash

# Cache management

python3 -m podcast_scraper.cli cache --status
python3 -m podcast_scraper.cli cache --clean [whisper|transformers|spacy|all]
python3 -m podcast_scraper.cli cache --clean --yes

# Select providers

python3 -m podcast_scraper.cli URL --transcription-provider openai
python3 -m podcast_scraper.cli URL --summary-provider openai
```

**Changed Flags:**

```bash

# Transcription (new default behavior)

--no-transcribe-missing     # NEW: Disable automatic transcription

# Whisper model (new default)

--whisper-model base.en     # Now default (was: base)
```python

## Migration Notes

### For Users Upgrading from v2.3.2

**Automatic Transcription (Breaking Behavior Change)**:

- **Before (v2.3.2)**: Episodes without transcripts were skipped unless `--transcribe-missing` was passed.
- **After (v2.4.0)**: Episodes without transcripts are automatically transcribed by default.
- **Migration**: Use `--no-transcribe-missing` to restore old behavior.

**Whisper Model Default**:

- **Before (v2.3.2)**: Default model was `base` (multilingual).
- **After (v2.4.0)**: Default model is `base.en` (English-optimized).
- **Migration**: Use `--whisper-model base` for multilingual support.

**Output Directory Structure**:

- **Before (v2.3.2)**: Transcripts and metadata in same directory.
- **After (v2.4.0)**: Transcripts in `transcripts/`, metadata in `metadata/`.
- **Migration**: Fully backward compatible (reads both structures).

## Next Steps

- **Audio Preprocessing Pipeline** (RFC-040): Implement audio preprocessing stage to reduce file size and API costs.
- **Multi-Provider Ecosystem**: Add support for more cloud LLM providers (Anthropic, Gemini, Mistral, etc.).
- **Quality Improvements**: Continue refining summarization thresholds and prompts.

## Full Changelog

**Full Changelog**: <https://github.com/chipi/podcast_scraper/compare/v2.3.2...v2.4.0>
