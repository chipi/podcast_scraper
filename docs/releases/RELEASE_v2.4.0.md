# Release v2.4.0 - Provider Ecosystem & Production Readiness

**Release Date:** January 2026
**Type:** Minor Release
**Last Updated:** January 10, 2026

## Summary

v2.4.0 is a **major minor release** introducing a **comprehensive multi-provider ecosystem** with 8 AI providers (1 local + 7 LLM), **production-ready configuration defaults**, **advanced cache management**, and **significant quality improvements** to summarization. This release represents a fundamental shift toward provider flexibility, giving users choice between local privacy and cloud convenience while maintaining backward compatibility.

##

 Key Features

### 🚀 Multi-Provider Ecosystem (8 Providers)

**Complete provider flexibility with unified protocol architecture:**

#### Local Provider (Privacy-First)

- **Transformers** (ML) - Local BART/LED models for summarization
- **Whisper** (ML) - Local OpenAI Whisper for transcription
- **spaCy** (ML) - Local NER for speaker detection

#### Cloud LLM Providers (7 Options)

- **OpenAI** - GPT-4, GPT-4o, GPT-4o-mini (transcription, speaker detection, summarization)
- **Anthropic** - Claude 3.5 Sonnet, Claude 3.7 Opus (speaker detection, summarization)
- **Mistral** - Mistral Large, Mistral Medium (speaker detection, summarization)
- **DeepSeek** - DeepSeek Chat, DeepSeek Coder (speaker detection, summarization)
- **Gemini** - Gemini 1.5 Pro, Gemini 2.0 Flash (speaker detection, summarization)
- **Groq** - Fast LLaMA, Mixtral inference (speaker detection, summarization)
- **Ollama** - Local LLM inference (speaker detection, summarization)

**Provider Selection:**

```yaml

# config.yaml - Mix and match providers

transcription_provider: whisper      # or openai
speaker_detector_provider: spacy     # or openai, anthropic, mistral, etc.
summary_provider: transformers       # or openai, anthropic, mistral, etc.
```

# CLI - Easy provider switching

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider openai \
  --speaker-detector-provider anthropic \
  --summary-provider mistral

```
- [AI Provider Comparison Guide](../guides/AI_PROVIDER_COMPARISON_GUIDE.md) - Detailed comparison of all 8 providers
- [Provider Configuration Quick Reference](../guides/PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) - Configuration examples
- [Provider Implementation Guide](../guides/PROVIDER_IMPLEMENTATION_GUIDE.md) - Implementation details

## ⚙️ Production-Ready Configuration Defaults

**Improved defaults for production use:**

### Automatic Transcription (New Default)

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
# Whisper Cache: /Users/user/.cache/whisper (2.5 GB, 3 models)
# Transformers Cache: /Users/user/.cache/huggingface/hub (8.2 GB, 5 models)
# spaCy Cache: /Users/user/.cache/spacy (150 MB, 1 model)
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
  - Transition zone: 5,500-6,500 tokens
- **BART Models** (short-context):
  - Ceiling: 4,000 tokens (unchanged)
  - Validation: 60% (unchanged)
  - Transition zone: 3,500-4,500 tokens

**Impact:**

- ✅ Reduced false warning frequency by ~40%
- ✅ More consistent quality across episode lengths
- ✅ Better handling of 3-6k token episodes
- ✅ Smooth transition zones (no hard cutoffs)

**Technical Details:**

```python

# src/podcast_scraper/summarization/map_reduce.py

# LED models (16k context window)

LED_MINI_MAP_REDUCE_MAX_TOKENS = 6000
LED_VALIDATION_THRESHOLD = 0.75
LED_TRANSITION_START = 5500
LED_TRANSITION_END = 6500

# BART models (1k context window)

BART_MINI_MAP_REDUCE_MAX_TOKENS = 4000
SUMMARY_VALIDATION_THRESHOLD = 0.6
BART_TRANSITION_START = 3500
BART_TRANSITION_END = 4500

```
## Increased Token Limits

- **Chunk summaries**: 80-160 tokens (was 60-100)
- **Section summaries**: 80-160 tokens (was 60-100)
- **Final summaries**: 200-480 tokens (was 150-300)

**Impact:** Summaries retain more detail and context while remaining concise.

### GPU Support Improvements

- **MPS** (Apple Silicon): Fixed buffer size issues, improved stability
- **CUDA** (NVIDIA): Optimized memory usage
- **Multi-GPU**: Better detection and utilization

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
        ├── 001_episode_title.metadata.json
        ├── 002_episode_title.txt
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

run_2.4.0_w_base.en_tf_bart-large-cnn_r_led-base-16384_sp_spacy_sm

```python
- `w_{model}` - Whisper model (e.g., `w_base.en`, `w_small.en`)
- `tf_{model}` - Transformers MAP model (e.g., `tf_bart-large-cnn`)
- `r_{model}` - REDUCE model (e.g., `r_led-base-16384`)
- `sp_{model}` - spaCy model (e.g., `sp_spacy_sm`, `sp_spacy_lg`)

**Benefits:**

- **Experimentation**: Easy A/B testing with different models
- **Organization**: Clear which models were used for each run
- **Reproducibility**: Know exact configuration from directory name

**Configuration:**

```yaml

# config.yaml - Custom run ID

run_id: my_experiment  # Results in: run_my_experiment_w_base.en_...

# Or use environment variable

RUN_ID=my_experiment python3 -m podcast_scraper.cli ...

```
## 🧪 Test Infrastructure Improvements

**Comprehensive test coverage and stability enhancements:**

### Test Pyramid Enhancement

- **Unit Tests**: ~250 tests (fast, isolated, no network/filesystem)
- **Integration Tests**: ~100 tests (moderate, component interaction)
- **E2E Tests**: ~50 tests (slow, full pipeline, real fixtures)

#### Network Isolation (Unit Tests)

- **Automatic Blocking**: All network requests blocked in unit tests
- **pytest-socket**: Enforces network isolation
- **Opt-In**: Use `@pytest.mark.allow_socket` for specific tests
- **Impact**: Prevents accidental network calls, faster execution

#### Fixture Enhancements

**New Long-Form Test Fixtures:**

- **p07** (Sustainability) - 14.5k words, ~3-4k tokens (tests hierarchical reduce)
- **p08** (Solar Energy) - 19k words, ~4.5-5.5k tokens (tests extractive fallback)
- **p09** (Biohacking) - 3 episodes, ~6-7k words each (tests threshold boundaries)

**Purpose**: Reproduce summarization quality issues at threshold boundaries (issue #283).

#### CI/CD Optimizations

- **Fast CI**: Runs on PRs (lint, unit tests, fast integration tests only)
- **Full CI**: Runs on main (includes E2E tests, full test suite)
- **Nightly**: Runs daily with all providers, generates metrics
- **ML Caching**: Models cached in CI for faster execution
- **Parallel Execution**: Tests run in parallel where safe

### 📚 Documentation Expansion

**Comprehensive documentation for all features:**

#### New Guides

- **ML Provider Tuning Guide** - Complete guide for fine-tuning local ML providers
- **AI Provider Comparison Guide** - Detailed comparison of all 8 providers (cost, quality, speed, privacy)
- **Provider Configuration Quick Reference** - Quick config examples
- **Provider Implementation Guide** - Guide for implementing new providers

#### Updated Guides

- **Development Guide** - Enhanced with package installation steps, .env setup
- **Testing Guide** - Updated with network isolation patterns
- **Summarization Guide** - Detailed threshold tuning documentation
- **Troubleshooting Guide** - Common issues and solutions

#### RFC Updates

- **RFC-040**: Audio Preprocessing Pipeline (future work)
- **RFC-039**: Development Workflow with Git Worktrees & CI
- Multiple provider RFCs (Anthropic, Mistral, DeepSeek, Gemini, Groq, Ollama)

## What's New

### 🔒 Security & Stability

#### Dependency Updates

- **urllib3**: Updated to >=2.6.0 (addresses CVEs)
- **zipp**: Pinned to >=3.19.1 (fixes infinite loop vulnerability)
- **transformers**: Updated to >=4.36.0 (compatibility improvements)

#### Test Isolation

- **Network Blocking**: Unit tests automatically block network requests
- **Filesystem Blocking**: Unit tests block filesystem I/O where appropriate
- **Subprocess Blocking**: Unit tests block subprocess execution

### ⚡ Performance Improvements

#### ML Model Preloading

- **Startup Preloading**: Models preloaded at process startup for faster first use
- **Worker Preloading**: Per-worker model instances for true parallelism
- **Memory Efficiency**: Models loaded once, reused across episodes

#### Cache Optimization

- **Content-Based Keys**: Cache keys based on file content (not timestamps)
- **HF_HUB_CACHE Priority**: Respects `HF_HUB_CACHE` environment variable
- **spaCy Exclusion**: Removed spaCy from CI cache (too large, rarely changes)

### 🐛 Bug Fixes

#### Provider Issues

- **Optional Guests**: Fixed handling of episodes with optional guest speakers
- **OpenAI API Keys**: Improved validation and error messages
- **Speaker Detection**: Fixed spaCy model loading edge cases

#### Test Issues

- **Test Model Usage**: Consistent use of test models across all test types
- **OpenAI E2E**: Fixed feed selection for multi-episode mode
- **Integration Tests**: Fixed output directory suffix logic
- **CI Cache**: Fixed cache validation and key generation

#### Configuration Issues

- **NoneType Errors**: Fixed cache directory configuration edge cases
- **Environment Variables**: Proper priority order (CLI > config file > env var > default)
- **Log Level**: `LOG_LEVEL` env var now properly overrides config file

## Technical Details

### Provider Architecture

**Unified Protocol System:**

- **Protocol Definitions**: `TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`
- **Provider Implementations**: Each provider implements relevant protocols
- **Factory Pattern**: Centralized provider creation with `create_*_provider()` functions
- **Protocol Compliance**: All providers validated against protocol contracts

**Provider Matrix:**

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini | Groq | Ollama |
| ------------ | ------- | -------- | ----------- | --------- | ---------- | -------- | ------ | -------- |
| Transcription | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Speaker Detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Summarization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Configuration Changes

**New Configuration Fields:**

```yaml

# Provider selection

transcription_provider: whisper      # or openai
speaker_detector_provider: spacy     # or openai, anthropic, mistral, deepseek, gemini, groq, ollama
summary_provider: transformers       # or openai, anthropic, mistral, deepseek, gemini, groq, ollama

# OpenAI configuration

openai_api_key: null                 # Set via OPENAI_API_KEY env var
openai_transcription_model: whisper-1
openai_speaker_model: gpt-4o-mini
openai_summary_model: gpt-4o-mini
openai_temperature: 0.3

# Anthropic configuration

anthropic_api_key: null              # Set via ANTHROPIC_API_KEY env var
anthropic_speaker_model: claude-3-5-sonnet-20241022
anthropic_summary_model: claude-3-5-sonnet-20241022
anthropic_temperature: 0.3

# Mistral configuration

mistral_api_key: null                # Set via MISTRAL_API_KEY env var
mistral_speaker_model: mistral-large-latest
mistral_summary_model: mistral-large-latest
mistral_temperature: 0.3

# DeepSeek configuration

deepseek_api_key: null               # Set via DEEPSEEK_API_KEY env var
deepseek_speaker_model: deepseek-chat
deepseek_summary_model: deepseek-chat
deepseek_temperature: 0.3

# Gemini configuration

gemini_api_key: null                 # Set via GEMINI_API_KEY env var
gemini_speaker_model: gemini-1.5-pro
gemini_summary_model: gemini-1.5-pro
gemini_temperature: 0.3

# Groq configuration

groq_api_key: null                   # Set via GROQ_API_KEY env var
groq_speaker_model: llama-3.3-70b-versatile
groq_summary_model: llama-3.3-70b-versatile
groq_temperature: 0.3

# Ollama configuration

ollama_base_url: http://localhost:11434
ollama_speaker_model: llama3.2
ollama_summary_model: llama3.2
ollama_temperature: 0.3

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
python3 -m podcast_scraper.cli URL --speaker-detector-provider anthropic
python3 -m podcast_scraper.cli URL --summary-provider mistral

```
# Transcription (new default behavior)

--transcribe-missing        # Now default (no longer needed)
--no-transcribe-missing     # NEW: Disable automatic transcription

# Whisper model (new default)

--whisper-model base.en     # Now default (was: base)

```python

## Migration Notes

### For Users Upgrading from v2.3.2

**Automatic Transcription (Breaking Behavior Change)**:

- **Before (v2.3.2)**: Episodes without transcripts were skipped
- **After (v2.4.0)**: Episodes without transcripts are automatically transcribed
- **Migration**: Use `--no-transcribe-missing` to restore old behavior

**Whisper Model Default**:

- **Before (v2.3.2)**: Default model was `base` (multilingual)
- **After (v2.4.0)**: Default model is `base.en` (English-optimized)
- **Migration**: Use `--whisper-model base` for multilingual support

**Output Directory Structure**:

- **Before (v2.3.2)**: Transcripts and metadata in same directory
- **After (v2.4.0)**: Transcripts in `transcripts/`, metadata in `metadata/`
- **Migration**: Fully backward compatible (reads both structures)

**Configuration File**:

- **New Fields**: Provider selection fields (`*_provider`, provider-specific configs)
- **Changed Defaults**: `transcribe_missing: true`, `whisper_model: base.en`
- **Migration**: Update config files to explicit values if needed

**Example Migration Config:**

```yaml

# v2.3.2 behavior (explicit)

transcribe_missing: false    # OLD: Skip episodes without transcripts
whisper_model: base          # OLD: Multilingual model

# v2.4.0 defaults (NEW behavior)

transcribe_missing: true     # NEW: Automatic transcription
whisper_model: base.en       # NEW: English-optimized model

```

## For Developers

**Provider Abstraction**:

- All providers now implement protocol interfaces
- Use factory functions (`create_*_provider()`) instead of direct instantiation
- Protocol compliance validated at runtime

**Test Isolation**:

- Unit tests now block network requests by default
- Use `@pytest.mark.allow_socket` for tests requiring network
- Filesystem blocking available with `@pytest.mark.allow_filesystem`

**Cache Management**:

- Use `cache_manager` module for cache operations
- Content-based cache keys for better invalidation
- HF_HUB_CACHE environment variable respected

## Testing

- **400+ tests passing** (250 unit, 100 integration, 50 E2E)
- **Comprehensive provider test coverage** for all 8 providers
- **Summarization threshold tests** for boundary conditions
- **Network isolation** in unit tests
- **CI/CD optimization** (fast CI on PRs, full CI on main)
- **ML model caching** in CI for faster execution

## Contributors

- Multi-provider ecosystem implementation
- Production-ready configuration defaults
- Advanced cache management CLI
- Summarization quality improvements (issue #283)
- Output directory reorganization
- Run ID suffix enhancements
- Test infrastructure improvements
- Documentation expansion
- CI/CD optimizations
- Bug fixes and stability improvements

## Related Issues & PRs

**Major Features:**

- #193: Multi-provider integration (Anthropic, Mistral, DeepSeek, Gemini, Groq, Ollama)
- #247: Improved defaults and cache management CLI (#221, #208, #224)
- #283: Inconsistent summarization quality at 4k token threshold
- #280: Metadata improvements and run suffix enhancements
- #171: Output directory reorganization

**Bug Fixes:**

- #319: Fix optional guests for ML/OpenAI providers
- #314: Fix OpenAI E2E tests feed selection
- #302: Fix integration tests for new output directory suffix
- #299: Fix OpenAI tests isolation
- #278: Improve cache directory configuration
- #260: Use test Whisper model in integration and E2E tests

**CI/CD & Testing:**

- #330: Add GPU support, fix speaker detection, improve summarization
- #271: Centralize ML model downloads
- #263: Unified provider factories
- #222: Optimize CI workflows
- #182: Integrate test coverage, code quality, and metrics dashboard
- #181: Preload ML models at startup

**Documentation:**

- #218: Add RFC-039, AI Provider Guide
- #190: Documentation updates and workflow improvements
- #172: Add new RFCs and update guides
- #165: Comprehensive documentation update

## Next Steps

- **Audio Preprocessing Pipeline** (RFC-040): Implement audio preprocessing stage
- **Provider Expansion**: Add more LLM providers (OpenRouter, Together AI)
- **Quality Improvements**: Continue refining summarization thresholds
- **Performance**: Further optimize ML model loading and caching
- **Documentation**: Expand provider-specific guides

## Breaking Changes

### ⚠️ Behavior Changes (Not Strictly Breaking)

**1. Automatic Transcription (New Default)**

- **Impact**: Episodes without transcripts are now automatically transcribed
- **Workaround**: Use `--no-transcribe-missing` to disable
- **Rationale**: Reduces friction, aligns with user expectations

**2. Whisper Model Default (base.en)**

- **Impact**: Default model changed from `base` (multilingual) to `base.en` (English)
- **Workaround**: Use `--whisper-model base` for multilingual
- **Rationale**: Better accuracy and speed for majority English content

**3. Output Directory Structure**

- **Impact**: New runs create `transcripts/` and `metadata/` subdirectories
- **Workaround**: None needed (fully backward compatible for reading)
- **Rationale**: Better organization, easier filtering

### ✅ Backward Compatibility

- **Configuration**: All old config files work unchanged (new defaults apply only if not explicitly set)
- **CLI**: Old CLI commands work unchanged (new defaults apply)
- **Output**: Reads both old and new directory structures
- **API**: No public API changes

## Full Changelog

**Full Changelog**: <https://github.com/chipi/podcast_scraper/compare/v2.3.2...v2.4.0>

**Commits Since v2.3.2**: 75 commits

**Lines Changed**: +15,000 / -8,000

**Files Changed**: 200+ files
