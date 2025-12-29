# Release v2.3.0 - Episode Summarization & Public API

**Release Date:** November 2025
**Type:** Minor Release
**Last Updated:** November 18, 2025

## Summary

v2.3.0 introduces **episode summarization with local transformer models** (PRD-005, RFC-012), enabling automatic generation of concise summaries and key takeaways from podcast transcripts. This release also establishes a **public API** with versioning, introduces a **service API** for non-interactive use, adds comprehensive **performance metrics**, implements **root directory reorganization** for better project structure, and includes significant **memory optimizations** and **performance improvements**.

## Key Features

### 🎯 PRD-005 & RFC-012: Episode Summarization (Implemented)

**Automatic generation of concise summaries and key takeaways from podcast transcripts:**

Episode summarization uses local transformer models (BART, DistilBART) to generate concise summaries and extract key takeaways from episode transcripts, enabling quick episode discovery and content understanding without reading full transcripts.

**Core Capabilities:**

- **Local-First Architecture**: All processing happens on-device using local transformer models (no external API calls)
- **GPU Acceleration**: Automatic device selection (MPS for Apple Silicon, CUDA for NVIDIA GPUs, CPU fallback)
- **Intelligent Chunking**: Long transcripts are automatically chunked with optimal overlap for best results
- **Quality Assurance**: Dynamic length adjustment, quality validation, and extractive fallback for fragmented outputs
- **Parallel Processing**: Chunk processing parallelized on CPU, episode summarization parallelized across episodes
- **Model Caching**: Models automatically cached locally and reused across runs
- **Cache Management**: CLI tools to inspect and prune transformer model cache
- **Memory Efficient**: Models loaded once and reused across all episodes, properly unloaded after processing

**Summary Generation:**

- **Concise Summaries**: Generates brief summaries (default: 160 tokens max, 60 tokens min) capturing main points
- **Hybrid Map-Reduce Strategy**: Uses BART-large for fast chunk summarization (MAP), LED for accurate final combine (REDUCE)
- **Hierarchical Reduce**: Multi-step abstractive reduce for medium-to-large outputs (800-4000 tokens)
- **Quality Validation**: Validates output quality, removes instruction leaks, and falls back to extractive methods if needed
- **Dynamic Length**: Automatically adjusts summary length based on transcript length
- **Transcript Cleaning**: Automatically cleans transcripts before summarization (removes timestamps, generic speaker tags, sponsor blocks)
- **Cleaned Transcript Saving**: Optionally saves cleaned transcripts to `.cleaned.txt` files for external testing (default: enabled)

**Configuration:**

````yaml
generate_summaries: true                    # Enable summarization
summary_model: null                         # MAP model (defaults to bart-large for chunk summaries)
summary_reduce_model: null                  # REDUCE model (defaults to long-fast/LED for final combine)
summary_max_length: 160                     # Maximum summary length (default: 160 tokens)
summary_min_length: 60                      # Minimum summary length (default: 60 tokens)
summary_chunk_size: null                    # Token chunk size (defaults to 2048)
summary_device: "auto"                      # Device selection: "auto", "cpu", "cuda", "mps"
summary_cache_dir: null                     # Custom cache directory (default: Hugging Face cache)
save_cleaned_transcript: true               # Save cleaned transcript to .cleaned.txt file (default: true)
```yaml

- `--generate-summaries`: Enable summarization (requires `--generate-metadata`)
- `--summary-model`: Specify MAP model (default: `bart-large` for fast chunk summarization)
- `--summary-reduce-model`: Specify REDUCE model (default: `long-fast`/LED for accurate final combine)
- `--summary-max-length`: Maximum summary length (default: 160 tokens)
- `--summary-min-length`: Minimum summary length (default: 60 tokens)
- `--summary-chunk-size`: Token chunk size (default: 2048)
- `--summary-device`: Device selection (`auto`, `cpu`, `cuda`, `mps`)
- `--save-cleaned-transcript`: Save cleaned transcript to .cleaned.txt file (default: enabled)
- `--no-save-cleaned-transcript`: Don't save cleaned transcript
- `--cache-info`: Display transformer model cache information
- `--prune-cache`: Prune transformer model cache to free disk space

**Model Selection:**

- **Hybrid Approach**: Uses different models for MAP and REDUCE phases:
  - **MAP Phase** (default: `bart-large`): Fast and efficient chunk summarization
  - **REDUCE Phase** (default: `long-fast`/LED): Accurate long-context final combine
- **Manual Selection**: Override with `--summary-model` and `--summary-reduce-model` flags
- **Device Support**: Automatic device detection (MPS for Apple Silicon, CUDA for NVIDIA GPUs, CPU fallback)

**Performance:**

- **Chunking Optimization**: Uses model's `max_position_embeddings` for optimal chunk size
- **Parallel Processing**: Chunks processed in parallel on CPU, episodes processed in parallel
- **Memory Efficiency**: Models loaded once, reused across episodes, properly unloaded
- **Cache Reuse**: Models cached locally, reused across runs without re-downloading

**Integration:**

- Summaries included in metadata documents (`summary` field)
- Works seamlessly with existing metadata generation
- Respects `--skip-existing` flag (can generate summaries from existing transcripts)
- Dry-run mode properly skips model loading
- **Cleaned Transcript Saving**: When `save_cleaned_transcript` is enabled (default: `true`), cleaned transcripts are saved to separate `.cleaned.txt` files alongside the original formatted transcripts. This allows external testing and validation of the cleaning pipeline.

**Related Documentation:**

- [PRD-005](../prd/PRD-005-episode-summarization.md) - Product Requirements Document
- [RFC-012](../rfc/RFC-012-episode-summarization.md) - Technical Design Document

### 🔌 Public API & Service Mode

**New programmatic interfaces for non-interactive use:**

#### Service API (`podcast_scraper.service`)

Optimized for daemon/service use cases with structured error handling:

- **Config File Only**: Works exclusively with configuration files (no CLI arguments)
- **Structured Results**: Returns `ServiceResult` with success status, episode count, summary, and error messages
- **Exit Codes**: Returns 0 for success, 1 for failure (suitable for process managers)
- **No User Interaction**: Designed for automation and process management tools

**API:**

```python
from podcast_scraper import service

# Run from config file

result = service.run_from_config_file("config.yaml")
if result.success:
    print(f"Processed {result.episodes_processed} episodes")
    print(result.summary)
else:
    print(f"Error: {result.error}")

# Run with Config object

from podcast_scraper import Config
config = Config(rss_url="https://example.com/feed.xml", ...)
result = service.run(config)
```text

- Running as a systemd service
- Managed by supervisor
- Scheduled execution (cron + service mode)
- CI/CD pipelines
- Automated workflows

**ServiceResult Structure:**

```python
@dataclass
class ServiceResult:
    episodes_processed: int      # Number of episodes processed
    summary: str                 # Human-readable summary
    success: bool                # Whether run completed successfully
    error: Optional[str]          # Error message if failed
```python

- **API Version**: `podcast_scraper.__api_version__` (same as `__version__`)
- **Semantic Versioning**: Follows major.minor.patch format
- **Backward Compatibility**: Within major version, all changes are backward compatible
- **Version Access**: `import podcast_scraper; print(podcast_scraper.__api_version__)`

**Versioning Policy:**

- **Major version (X.y.z)**: Breaking API changes
- **Minor version (x.Y.z)**: New features, backward compatible
- **Patch version (x.y.Z)**: Bug fixes, backward compatible

**Documentation:**

- [API Reference](../api/REFERENCE.md) - Complete API documentation
- [API Boundaries](../api/BOUNDARIES.md) - API design and boundaries
- [API Migration Guide](../api/MIGRATION_GUIDE.md) - Migration between versions

### 📊 Performance Metrics

**Comprehensive performance tracking for A/B testing and optimization:**

- **Per-Episode Timing**: Tracks time for each major operation:
  - Media download time
  - Transcription time
  - Speaker name extraction time
  - Summary generation time
- **Average Times**: Calculates and reports average times per episode
- **Pipeline Summary**: Includes detailed performance metrics in final summary
- **Metrics Collection**: In-memory metrics collector with per-episode tracking

**Example Output:**

```text
Pipeline Summary:

- Episodes processed: 10
- Transcripts downloaded: 8
- Episodes transcribed: 2
- Average download time: 2.3s/episode
- Average transcription time: 45.2s/episode
- Average name extraction time: 0.8s/episode
- Average summary time: 12.5s/episode
```text

### 🧪 Evaluation & Testing Tools

**New evaluation scripts for quality assurance and regression testing:**

#### Summarization Evaluation (`scripts/eval_summaries.py`)

Evaluates summarization quality using ROUGE metrics and reference-free checks:

- **ROUGE Scoring**: Computes ROUGE-1, ROUGE-2, and ROUGE-L scores against reference summaries
- **Reference-Free Checks**: Compression ratio, repetition detection, keyword coverage
- **Model Configuration**: Supports separate MAP and REDUCE models (defaults match app: BART-large for MAP, LED for REDUCE)
- **Regression Testing**: Outputs JSON reports for comparing model performance over time
- **Golden Dataset**: Works with evaluation dataset in `data/eval/` with `transcript.cleaned.txt` and `summary.gold.*.txt` files

**Usage:**

```bash

# Use defaults (BART-large MAP, LED REDUCE)

python scripts/eval_summaries.py

# Specify models

python scripts/eval_summaries.py --map-model bart-large --reduce-model long-fast

# Custom output

python scripts/eval_summaries.py --output results/my_eval.json
```yaml

- **Removal Statistics**: Character and word removal percentages
- **Sponsor/Ad Detection**: Counts sponsor patterns before/after cleaning
- **Brand Mention Detection**: Tracks removal of common podcast sponsor brands
- **Outro Pattern Detection**: Counts outro patterns (subscribe, rate/review, etc.)
- **Quality Flags**: Flags episodes with >60% removal or ineffective cleaning
- **Diff Snippets**: Shows what was removed (unified diff format)

**Usage:**

```bash

# Evaluate all episodes

python scripts/eval_cleaning.py

# Single episode

python scripts/eval_cleaning.py --episode ep01
```text

- `transcript.raw.txt` - Raw Whisper output
- `transcript.cleaned.txt` - Cleaned transcript (after cleaning pipeline)
- `summary.gold.long.txt` - Detailed reference summary (for ROUGE)
- `summary.gold.short.txt` - Optional concise reference summary
- `metadata.json` - Optional episode metadata

**Manual Evaluation:**

- `data/eval/MANUAL_EVAL_CHECKLIST.md` - Manual evaluation rubric (coverage, faithfulness, clarity, conciseness)

### 🗂️ Root Directory Reorganization

**Cleaner project structure with organized build artifacts:**

- **Build Artifacts**: Moved to `.build/` directory:
  - `dist/` → `.build/dist/`
  - `site/` → `.build/site/`
  - `podcast_scraper.egg-info/` → `.build/podcast_scraper.egg-info/`
  - `.coverage` → `.build/.coverage`
- **Test Outputs**: Moved to `.test_outputs/` directory
- **Docker Files**: Moved to `docker/` directory
- **Scripts**: Moved to `scripts/` directory
- **Updated References**: All documentation and CI workflows updated

**Benefits:**

- Cleaner root directory
- Clear separation of source vs. generated files
- Easier cleanup (`make clean` removes `.build/`)
- Better organization for contributors

## What's New

### 🚀 Memory Optimizations

#### Summary Model Memory Leak Fix

**Critical fix for summarization memory usage:**

- **Problem**: Summary models were being loaded once per episode, causing memory usage to grow linearly (e.g., 50-60GB for 100 episodes)
- **Solution**: Models are now loaded once at the start of processing and reused across all episodes
- **Impact**: Memory usage reduced from ~500MB × episode_count to ~1-2GB total (regardless of episode count)
- **Cleanup**: Models are properly unloaded after all episodes are processed

**Technical Details:**

- Model is created once in `run_pipeline()` before episode processing
- Model instance is passed through the call chain to avoid reloading
- Explicit cleanup with `unload_model()` after processing completes
- Dry-run mode properly skips model loading to avoid PyTorch initialization

#### spaCy Model Memory Optimizations

**Optimizations for speaker detection memory usage:**

- **Component Disabling**: spaCy models now load only the NER component, disabling parser, tagger, and lemmatizer
  - Memory reduction: 30-50% (from ~50-100MB to ~30-50MB)
  - Fallback: Automatically falls back to full pipeline if component disabling is not supported
- **Cache Cleanup**: spaCy model cache is cleared after processing completes
  - Prevents models from persisting in memory after processing
  - Similar cleanup pattern to summary models

**Technical Details:**

- `spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])` for NER-only loading
- Graceful fallback if component disabling fails
- `clear_spacy_model_cache()` called after episode processing
- Module-level cache already existed; now properly cleaned up

### 📊 Performance Impact

| Component | Before | After | Improvement |
| --------- | ------ | ----- | ----------- |
| Summary model (100 episodes) | 50-60GB | ~1GB | ~98% reduction |
| spaCy model (full pipeline) | ~50-100MB | ~30-50MB | 30-50% reduction |
| spaCy cache persistence | Persistent | Cleared after processing | ~50MB freed |
| **Total baseline** | **50-60GB** | **~1-2GB** | **~97% reduction** |

### 🔧 Technical Changes

#### Summarization Implementation

- **New Module**: `summarizer.py` with `SummaryModel` class
- **Hybrid Model Selection**: Separate MAP (BART-large) and REDUCE (LED) models for optimal performance
- **Chunking Strategy**: Intelligent token-based chunking with optimal overlap (default: 2048 tokens)
- **Transcript Cleaning**: Pre-processing step removes timestamps, generic speaker tags, sponsor blocks
- **Cleaned Transcript Saving**: Optionally saves cleaned transcripts to `.cleaned.txt` files (default: enabled)
- **Quality Assurance**: Dynamic length adjustment, instruction leak removal, quality validation, extractive fallback
- **Hierarchical Reduce**: Multi-step abstractive reduce with iterative re-chunking for large outputs
- **Parallel Processing**: Chunk processing parallelized on CPU, episode summarization parallelized
- **Cache Management**: Functions to inspect and prune transformer model cache
- **Evaluation Tools**: `scripts/eval_summaries.py` and `scripts/eval_cleaning.py` for quality assessment

#### Service API Implementation

- **New Module**: `service.py` with `ServiceResult` dataclass
- **Entry Points**: `service.run()`, `service.run_from_config_file()`, `service.main()`
- **Error Handling**: Structured error handling with `ServiceResult`
- **CLI Entry Point**: `python -m podcast_scraper.service --config config.yaml`

#### API Versioning (Implementation)

- **Version Constants**: `__api_version__` tied to `__version__`
- **Documentation**: Comprehensive API documentation in `docs/api/`
- **Versioning Policy**: Documented in `BOUNDARIES.md`

#### Performance Metrics

- **Metrics Module**: Enhanced `metrics.py` with per-episode timing
- **Pipeline Integration**: Metrics collected throughout pipeline execution
- **Summary Reporting**: Detailed performance metrics in final summary

#### Root Reorganization

- **Directory Structure**: New `.build/`, `.test_outputs/`, `docker/`, `scripts/` directories
- **Makefile Updates**: Updated paths for build artifacts
- **CI Updates**: Updated GitHub Actions workflows for new paths
- **Documentation**: Updated README and CONTRIBUTING with new structure

### 🐛 Bug Fixes

- **Memory Leak**: Fixed summary model being loaded once per episode instead of being reused
- **Dry-Run Memory**: Fixed PyTorch being initialized during dry-run mode even when not needed
- **spaCy Memory**: Optimized spaCy model loading to use only required components
- **Output Quality**: Fixed fragmented summaries with dynamic length adjustment and quality validation
- **Chunking Performance**: Optimized chunking to use model's `max_position_embeddings` for optimal chunk size
- **Missing Import**: Fixed missing `time` import in `metadata.py`
- **Unused Imports**: Removed unused imports in `workflow.py` and `test_service.py`

### 🔧 CI/CD Improvements (Post-Release Update)

**Comprehensive CI/CD fixes and improvements (#52):**

- **Linting & Formatting**: Fixed all linting and formatting issues across the codebase
- **Markdown Compliance**: Updated all markdown files to comply with markdownlint rules
- **Documentation Build**: Fixed GitHub Actions workflow for documentation deployment
- **Gitignore Updates**: Enhanced `.gitignore` with proper build artifact patterns
- **Makefile Improvements**: Updated build targets and paths for better artifact management
- **Code Quality**: Improved code formatting consistency across all Python modules
- **Test Files**: Enhanced test file organization and formatting
- **Example Configs**: Updated example configuration files with better formatting and comments

**Files Updated:**

- CI workflows (`.github/workflows/docs.yml`, `.github/workflows/python-app.yml`)
- Linting configurations (`.flake8`, `.markdownlint.json`)
- Project configuration (`Makefile`, `pyproject.toml`, `.gitignore`)
- Core modules (`__init__.py`, `cli.py`, `config.py`, `episode_processor.py`, `metadata.py`, `speaker_detection.py`, `workflow.py`, `whisper_integration.py`)
- All documentation files for markdown compliance
- Test files for consistent formatting
- Example configuration files

**Impact:**

- All CI checks now pass consistently
- Improved code maintainability and readability
- Better documentation formatting and navigation
- Cleaner build process with proper artifact management

### 📝 Configuration

**New Configuration Fields:**

```yaml

# Summarization

generate_summaries: false                  # Enable summarization
summary_model: null                        # Model selection (auto if null)
summary_max_length: 150                    # Maximum summary length
summary_min_length: 30                     # Minimum summary length
summary_max_takeaways: 5                   # Maximum takeaways
summary_device: "auto"                     # Device selection
summary_cache_dir: null                    # Custom cache directory

# Logging

log_file: null                             # Optional log file path

# Media Reuse

reuse_media: false                         # Reuse existing media files
```python

- Now allows output generation (summaries, metadata) from existing transcripts even when skipping other processing steps
- Useful for testing summarization without re-downloading or re-transcribing

### 🔄 Backward Compatibility

- ✅ Fully backward compatible
- ✅ All existing functionality preserved
- ✅ New features are opt-in (defaults to `false`)
- ✅ No breaking API changes
- ✅ Service API is additive (doesn't affect existing CLI)

### 📚 Related Documentation

- RFC-012: Episode Summarization Using Local Transformers
- RFC-010: Automatic Speaker Name Detection
- API Reference: Complete public API documentation
- API Boundaries: API design and boundaries assessment
- API Migration Guide: Migration between versions
- Memory optimization patterns documented in code comments

## Migration Notes

### For Users Upgrading from v2.2.0

**New Feature**: Episode summarization is now available! Enable it with `--generate-summaries` (requires `--generate-metadata`).

**New Feature**: Service API for non-interactive use. See `podcast_scraper.service` module.

**New Feature**: API versioning. Access via `podcast_scraper.__api_version__`.

**New Feature**: Performance metrics. Automatically collected and reported in pipeline summary.

**Root Directory Changes**: Build artifacts now in `.build/`, test outputs in `.test_outputs/`. Use `make clean` to remove.

**No Breaking Changes**: All existing functionality remains the same. New features are opt-in.

**Dependencies**: New optional dependencies for summarization:

- `torch>=2.0.0`
- `transformers>=4.30.0`
- `sentencepiece>=0.1.99`
- `accelerate>=0.20.0`

**Configuration Example:**

```yaml

# Enable summarization

generate_metadata: true
generate_summaries: true
summary_model: "facebook/bart-large-cnn"  # Optional: auto-selected if not specified
summary_max_length: 150
summary_max_takeaways: 5

# Optional: Custom cache directory

summary_cache_dir: "~/.cache/my-models"

# Optional: Log to file

log_file: "podcast_scraper.log"

# Optional: Reuse media for faster testing

reuse_media: true
```text

- **202 tests passing** (6 new tests for API versioning)
- **4 subtests passing**
- **Comprehensive summarization test coverage**:
  - Model selection and initialization
  - Chunking strategy
  - Summary generation
  - Key takeaways extraction
  - Memory optimization
  - Parallel processing
  - Cache management
- **Service API test coverage**:
  - `ServiceResult` structure
  - `service.run()` and `service.run_from_config_file()`
  - Error handling
  - CLI entry point
  - Importability
- **All test categories organized**:
  - Unit tests (summarization, service API, API versioning)
  - Integration tests (workflow integration)
  - E2E tests (library API and CLI)

## Contributors

- Episode summarization implementation
- Service API design and implementation
- API versioning and documentation
- Performance metrics collection
- Root directory reorganization
- Memory optimizations
- Performance improvements
- CI/CD pipeline improvements and fixes
- Code quality and formatting enhancements
- Documentation compliance and improvements

## Next Steps

- Continue performance optimizations
- Explore additional transformer models
- Enhance output quality validation
- Expand API documentation
- Add more comprehensive E2E tests

**Full Changelog**: <https://github.com/chipi/podcast_scraper/compare/v2.2.0...v2.3.0>
````
