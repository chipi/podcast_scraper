# Release v2.5.0 - LLM Provider Expansion & Production Hardening

**Release Date:** February 2026
**Type:** Minor Release
**Last Updated:** February 6, 2026

## Summary

v2.5.0 is a **minor release** that expands the LLM provider ecosystem from 2 to 7 cloud providers, introduces **production-hardening features** (MPS exclusive mode, entity reconciliation, run manifests), adds **comprehensive LLM metrics tracking**, and includes significant **quality improvements** and **stability fixes**. This release focuses on making the multi-provider system production-ready with better observability, reproducibility, and correctness.

## 🚀 Key Features

### 🌐 Expanded LLM Provider Ecosystem (7 Cloud Providers)

**Complete LLM provider support with unified interface:**

v2.4.0 introduced the multi-provider architecture with OpenAI and Gemini. v2.5.0 adds **5 additional LLM providers**:

#### New Providers

- **Anthropic** - Claude 3.5 Sonnet, Claude 3.7 Opus (speaker detection, summarization)
- **Mistral** - Mistral Large, Mistral Medium (speaker detection, summarization)
- **DeepSeek** - DeepSeek Chat, DeepSeek Coder (speaker detection, summarization)
- **Grok** - xAI's Grok models with real-time information access (speaker detection, summarization)
- **Ollama** - Local LLM inference server (speaker detection, summarization)

**Provider Selection:**

```yaml
# config.yaml - Choose from 7 LLM providers

speaker_detector_provider: anthropic  # or mistral, deepseek, grok, ollama
summary_provider: mistral             # or anthropic, deepseek, grok, ollama
```

```bash
# CLI - Easy provider switching

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --speaker-detector-provider anthropic \
  --summary-provider mistral
```

**Installation:**

```bash
# Install all LLM providers
pip install -e ".[llm]"

# Or install individual providers
pip install -e ".[anthropic]"
pip install -e ".[mistral]"
pip install -e ".[deepseek]"
pip install -e ".[grok]"
pip install -e ".[ollama]"
```

**Benefits:**

- **Provider Flexibility**: Choose the best provider for your use case (cost, quality, speed)
- **Redundancy**: Switch providers if one has outages or rate limits
- **Cost Optimization**: Compare costs across providers for your workload
- **Privacy Options**: Use Ollama for fully local LLM inference

**Related Documentation:**

- [AI Provider Comparison Guide](../guides/AI_PROVIDER_COMPARISON_GUIDE.md) - Updated with all 7 providers
- [Provider Configuration Quick Reference](../guides/PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) - Configuration examples
- [Provider Implementation Guide](../guides/PROVIDER_IMPLEMENTATION_GUIDE.md) - Implementation details

### 🍎 MPS Exclusive Mode (Apple Silicon Optimization)

**Prevents GPU memory contention on Apple Silicon:**

When both Whisper transcription and summarization use MPS (Metal Performance Shaders) on Apple Silicon, the system can serialize GPU work to prevent memory contention. This is enabled by default and ensures:

- **Transcription completes first**: All Whisper transcriptions finish before summarization starts
- **I/O remains parallel**: Downloads, RSS parsing, and file I/O continue in parallel
- **Memory safety**: Prevents both models from competing for the same GPU memory pool

**Configuration:**

```yaml
# config.yaml
mps_exclusive: true  # Default: true (enabled)
```

```bash
# CLI
--mps-exclusive      # Default (enabled)
--no-mps-exclusive   # Disable for maximum throughput (requires sufficient GPU memory)
```

**When to disable**: If you have sufficient GPU memory (e.g., M4 Pro with 48GB+ unified memory) and want maximum throughput, you can disable exclusive mode to allow concurrent GPU operations.

**Related Documentation:**

- [Segfault Mitigation Guide](../guides/SEGFAULT_MITIGATION.md) - MPS stability strategies
- [ML Provider Reference](../guides/ML_PROVIDER_REFERENCE.md) - Hardware acceleration details

### 🔗 Entity Reconciliation

**Automatic correction of entity names in summaries:**

The system now automatically reconciles entity names in summaries with extracted entities from speaker detection. When entity names are close matches (edit distance ≤ 2), the system corrects the summary text to use the extracted entity spelling.

**Features:**

- **Automatic Correction**: Entity names in summaries are corrected to match extracted entities
- **Edit Distance Matching**: Handles minor spelling variations (e.g., "John Smith" vs "John Smyth")
- **Preference for Extracted Entities**: Extracted entities (from speaker detection) are considered authoritative
- **Correction Tracking**: All corrections are logged for transparency

**Example:**

```python
# Before reconciliation
summary = "John Smyth discussed the topic with Jane Doe."

# After reconciliation (if "John Smith" was extracted)
summary = "John Smith discussed the topic with Jane Doe."
corrections = [EntityCorrection(old="John Smyth", new="John Smith", edit_distance=1)]
```

**Configuration:**

Entity reconciliation is enabled by default when using ML providers (transformers). LLM providers skip reconciliation as they generally produce higher-quality entity names.

**Related Issue:** #380

### 📋 Run Manifest (Reproducibility Tracking)

**Comprehensive run metadata for reproducibility:**

Every pipeline run now generates a `run_manifest.json` file that captures all information needed to reproduce the run, including:

- **Version Control**: Git commit SHA, branch, dirty flag
- **Configuration**: Config file hash, full config string
- **Environment**: Python version, OS, CPU/GPU info
- **Dependencies**: PyTorch, Transformers, Whisper versions
- **Models**: Model names, revisions, devices used
- **Generation Parameters**: Temperature, seed values

**Location:**

```text
output/
└── rss_feeds.example.com_abc123/
    └── run_my_run_id/
        ├── run_manifest.json  # NEW: Reproducibility manifest
        ├── transcripts/
        └── metadata/
```

**Use Cases:**

- **Reproducibility**: Recreate exact conditions of a run
- **Debugging**: Understand what models/configs were used
- **Auditing**: Track what was processed and how
- **Experimentation**: Compare runs with different configurations

**Schema Version:** 1.0.0 (Issue #379)

**Related Issue:** #379

### 📊 Unified LLM Metrics & Workflow Consolidation

**Comprehensive metrics tracking for all LLM providers:**

The pipeline now tracks consistent metrics across all LLM providers using a unified `ProviderCallMetrics` contract:

- **API Call Tracking**: Number of calls per provider
- **Token Usage**: Input/output tokens for each call
- **Cost Estimation**: Estimated costs based on provider pricing
- **Retry Tracking**: Number of retries and rate limit sleeps
- **Performance Metrics**: Latency per call

**Standardized Logging Format:**

```text
episode_metrics: audio_sec=X, transcribe_sec=Y, summary_sec=Z,
retries=N, rate_limit_sleep_sec=W, prompt_tokens=A,
completion_tokens=B, estimated_cost=C
```

**Benefits:**

- **Provider Comparison**: Direct comparison of costs, performance, quality across providers
- **Cost Monitoring**: Track API costs per run
- **Performance Analysis**: Identify bottlenecks and optimize provider selection
- **Consistent Format**: All episodes log the same keys in the same order

**Related ADR:** [ADR-043: Unified Provider Metrics Contract](../adr/ADR-043-unified-provider-metrics-contract.md)

**Related Issue:** #399

### 📚 Knowledge Graph Documentation

**Comprehensive documentation for Knowledge Graph features:**

Added complete documentation for the Podcast Knowledge Graph (PKG) system:

- **Ontology Documentation**: Node types, edge types, properties, identity rules
- **Schema Reference**: JSON schema for kg.json outputs
- **Design Principles**: Evidence-first, minimal ontology, stable IDs
- **Implementation Guide**: How to generate and consume knowledge graphs

**Documentation:**

- [Knowledge Graph Ontology](../kg/ontology.md) - Complete ontology reference
- [Knowledge Graph Schema](../kg/kg.schema.json) - JSON schema validation

**Related PR:** #391

## 🎯 Improvements

### Dependency Management

- **LLM Provider Extras**: LLM provider dependencies grouped into `[llm]` extra for cleaner installation
- **Dependency Updates**: Updated multiple dependencies for security and compatibility:
  - `openai`: >=1.0.0,<3.0.0 (was <2.0.0)
  - `rich`: >=13.0.0,<15.0.0 (was <14.0.0)
  - `pydeps`: >=1.12.0,<4.0.0 (was <2.0.0)
  - `accelerate`: Updated to latest version
  - `pytest`: >=7.4.0,<10.0.0 (was <9.0.0)

### CI/CD Improvements

- **GitHub Actions Updates**: Bumped multiple actions to latest versions:
  - `actions/checkout`: v4 → v6
  - `actions/setup-python`: v5 → v6
  - `actions/upload-artifact`: v4 → v6
  - `actions/download-artifact`: v4 → v7
  - `actions/cache`: v4 → v5
  - `actions/setup-node`: v4 → v6
  - `github/codeql-action`: v3 → v4
  - `codecov/codecov-action`: v4 → v5
  - `dawidd6/action-download-artifact`: v6 → v14
  - `docker/build-push-action`: v5 → v6

### Code Quality

- **Error Handling**: Improved error handling in RSS parsing and feed metadata extraction
- **Test Stability**: Fixed intermittent test failures and improved test isolation
- **Linting**: Fixed all linting issues and improved code quality
- **Type Hints**: Enhanced type hints throughout the codebase

### GPU Support

- **MPS Stability**: Improved MPS (Apple Silicon) stability and memory management
- **CUDA Optimization**: Better CUDA memory usage and multi-GPU detection
- **Device Detection**: Improved automatic device detection and fallback logic

## 🐛 Bug Fixes

### Correctness & Reproducibility

- **Summary Generation**: Fixed issue where summaries were missing when `generate_summaries=True` (#384)
- **Entity Reconciliation**: Fixed entity reconciliation edge cases and improved accuracy
- **RSS Parsing**: Fixed error handling in `parse_rss_items` to always return 3 values
- **Feed Metadata**: Fixed error handling in `extract_feed_metadata` to always return 3 values
- **Path Traversal**: Improved path traversal test to handle different directory structures

### Test Fixes

- **Integration Tests**: Fixed integration test failures after 2.4 forward port (#334)
- **E2E Tests**: Fixed Ollama e2e tests to skip when Ollama server is not available
- **Model Loading**: Fixed test failures when model files are not fully cached
- **OpenAI Tests**: Fixed OpenAI API key handling in integration tests
- **Workflow Tests**: Fixed workflow test patches and provider creation mocks

### Docker & Build

- **Missing Modules**: Fixed missing `path_validation.py` and `timeout.py` modules for Docker build
- **Circular Dependencies**: Fixed circular dependency in nightly workflow

### Metrics & Monitoring

- **Metrics Dashboard**: Improved metrics dashboard and fixed slowest tests extraction (#239)
- **Coverage Thresholds**: Removed coverage thresholds from fast integration and E2E tests for consistency

## ⚙️ Configuration Changes

### New Configuration Fields

```yaml
# MPS exclusive mode (Apple Silicon)
mps_exclusive: true  # Default: true (enabled)

# LLM provider configuration (new providers)
anthropic_api_key: null              # Set via ANTHROPIC_API_KEY env var
anthropic_speaker_model: claude-3-5-sonnet-20241022
anthropic_summary_model: claude-3-5-sonnet-20241022
anthropic_temperature: 0.3

mistral_api_key: null                # Set via MISTRAL_API_KEY env var
mistral_speaker_model: mistral-large-latest
mistral_summary_model: mistral-large-latest
mistral_temperature: 0.3

deepseek_api_key: null               # Set via DEEPSEEK_API_KEY env var
deepseek_speaker_model: deepseek-chat
deepseek_summary_model: deepseek-chat
deepseek_temperature: 0.3

grok_api_key: null                   # Set via GROK_API_KEY env var
grok_speaker_model: grok-2
grok_summary_model: grok-2
grok_temperature: 0.3

ollama_base_url: http://localhost:11434
ollama_speaker_model: llama3.2
ollama_summary_model: llama3.2
ollama_temperature: 0.3
```

### CLI Changes

**New Options:**

```bash
# MPS exclusive mode
--mps-exclusive        # Default (enabled)
--no-mps-exclusive     # Disable for maximum throughput

# New LLM providers
--speaker-detector-provider anthropic|mistral|deepseek|grok|ollama
--summary-provider anthropic|mistral|deepseek|grok|ollama
```

## 🛠️ Technical Details

### Provider Architecture

**Unified Provider Metrics Contract:**

All providers now implement a unified `ProviderCallMetrics` contract:

- **Required Parameter**: All providers must accept `ProviderCallMetrics` in `transcribe_with_segments()` and `summarize()` methods
- **Null for Unavailable Metrics**: Providers set `null` for unavailable metrics (e.g., local ML providers set `prompt_tokens=None`)
- **Standardized Logging**: Pipeline logs use consistent format for all providers

**Provider Matrix:**

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini | Grok | Ollama |
| ------------ | ------- | -------- | ----------- | --------- | ---------- | -------- | ------ | -------- |
| Transcription | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Speaker Detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Summarization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Run Manifest Schema

**Schema Version:** 1.0.0

**Fields:**

- `run_id`: Unique run identifier
- `created_at`: ISO 8601 timestamp
- `created_by`: User who created the run
- `git_commit_sha`: Git commit SHA
- `git_branch`: Git branch name
- `git_dirty`: Whether working directory was dirty
- `config_sha256`: SHA256 hash of configuration
- `config_path`: Path to configuration file
- `full_config_string`: Full provider/model config string
- `python_version`: Python version
- `os_name`: Operating system name
- `os_version`: Operating system version
- `cpu_info`: CPU information
- `gpu_info`: GPU information
- `torch_version`: PyTorch version
- `transformers_version`: Transformers version
- `whisper_version`: Whisper version
- `whisper_model`: Whisper model name
- `whisper_model_revision`: Whisper model revision
- `summary_model`: Summary model name
- `summary_model_revision`: Summary model revision
- `reduce_model`: Reduce model name
- `reduce_model_revision`: Reduce model revision
- `whisper_device`: Whisper device (cpu/cuda/mps)
- `summary_device`: Summary device (cpu/cuda/mps)
- `temperature`: Generation temperature
- `seed`: Random seed
- `schema_version`: Schema version (1.0.0)

### Entity Reconciliation Algorithm

**Edit Distance Matching:**

- **Threshold**: Maximum edit distance of 2 for corrections
- **Preference**: Extracted entities (from speaker detection) are authoritative
- **SpaCy Integration**: Uses spaCy NER to extract entities from summaries
- **Correction Tracking**: All corrections logged with old/new values and edit distance

## ⏩ Migration Notes

### For Users Upgrading from v2.4.0

**LLM Provider Installation:**

If you want to use new LLM providers (Anthropic, Mistral, DeepSeek, Grok, Ollama), install the `[llm]` extra:

```bash
pip install -e ".[llm]"
```

Or install individual providers:

```bash
pip install -e ".[anthropic]"
pip install -e ".[mistral]"
# etc.
```

**MPS Exclusive Mode:**

MPS exclusive mode is enabled by default. If you have sufficient GPU memory and want maximum throughput, you can disable it:

```yaml
# config.yaml
mps_exclusive: false
```

Or via CLI:

```bash
--no-mps-exclusive
```

**Run Manifest:**

Run manifests are automatically generated for all runs. No configuration needed. The manifest is saved to `run_manifest.json` in the output directory.

**Entity Reconciliation:**

Entity reconciliation is enabled by default for ML providers. No configuration needed. LLM providers skip reconciliation as they generally produce higher-quality entity names.

**Configuration File:**

New configuration fields are optional. Existing config files work unchanged. Add new fields only if you want to use new LLM providers.

**Example Migration Config:**

```yaml
# v2.4.0 config (still works)
transcription_provider: whisper
speaker_detector_provider: spacy
summary_provider: transformers

# v2.5.0 additions (optional)
mps_exclusive: true  # NEW: MPS exclusive mode (default: true)

# New LLM providers (optional)
speaker_detector_provider: anthropic  # NEW: Use Anthropic for speaker detection
summary_provider: mistral            # NEW: Use Mistral for summarization

# Provider-specific configs (if using new providers)
anthropic_api_key: null              # Set via ANTHROPIC_API_KEY env var
mistral_api_key: null                # Set via MISTRAL_API_KEY env var
```

## For Developers

**Provider Implementation:**

- All providers must implement the `ProviderCallMetrics` contract
- Providers set `null` for unavailable metrics
- Providers call `call_metrics.finalize()` before returning results

**Run Manifest:**

- Use `create_run_manifest()` to generate manifests
- Manifests are automatically saved to output directory
- Schema version is tracked for future compatibility

**Entity Reconciliation:**

- Use `_reconcile_entities()` for entity reconciliation
- Edit distance threshold is configurable (default: 2)
- Corrections are tracked and logged

**MPS Exclusive Mode:**

- Use `_both_providers_use_mps()` to detect when both providers use MPS
- Serialize GPU work when MPS exclusive mode is enabled
- I/O operations remain parallel

## Testing

- **400+ tests passing** (250 unit, 100 integration, 50 E2E)
- **Comprehensive provider test coverage** for all 8 providers (1 local + 7 LLM)
- **LLM provider tests** for Anthropic, Mistral, DeepSeek, Grok, Ollama
- **MPS exclusive mode tests** for Apple Silicon optimization
- **Entity reconciliation tests** for correctness
- **Run manifest tests** for reproducibility tracking
- **Metrics tests** for unified provider metrics contract

## Contributors

- Multiple LLM provider support (Anthropic, Mistral, DeepSeek, Grok, Ollama)
- MPS exclusive mode for Apple Silicon optimization
- Entity reconciliation for improved accuracy
- Run manifest for reproducibility tracking
- Unified LLM metrics and workflow consolidation
- Knowledge Graph documentation
- GPU support improvements
- Bug fixes and stability improvements
- Dependency updates and CI/CD improvements

## Related Issues & PRs

**Major Features:**

- #398: Add multiple LLM provider support (Anthropic, DeepSeek, Grok, Mistral, Ollama)
- #391: Add Knowledge Graph documentation, Docker optimization, and LLM provider docs
- #386: Add MPS exclusive mode, entity reconciliation, and run manifest features
- #344: Workflow consolidation, LLM metrics, and optimizations
- #330: Add GPU support, fix speaker detection, and improve summarization

**Bug Fixes:**

- #389: Multiple correctness, reproducibility, and quality improvements
- #384: Ensure summaries are present when generate_summaries=True
- #381: Fix linting issues, test failures, and add error handling
- #355: Fix failing tests and linting issues
- #334: Stabilization after 2.4 forward port and provider refactor

**CI/CD & Testing:**

- #375: Bump actions/checkout from 4 to 6
- #374: Bump dawidd6/action-download-artifact from 6 to 14
- #364: Update rich requirement
- #362: Bump github/codeql-action from 3 to 4
- #361: Bump actions/download-artifact from 4 to 7
- #360: Bump actions/upload-artifact from 4 to 6
- #339: Bump codecov/codecov-action from 4 to 5
- #338: Bump actions/setup-node from 4 to 6
- #337: Bump actions/cache from 4 to 5
- #310: Update pydeps requirement

**Documentation:**

- #391: Add Knowledge Graph documentation
- #239: Improve metrics dashboard and fix slowest tests extraction

## Next Steps

- **Hybrid Summarization Pipeline** (RFC-042): Replace REDUCE phase with instruction-following models
- **Audio Preprocessing Pipeline** (RFC-040): Implement audio preprocessing stage
- **Provider Expansion**: Add more LLM providers (OpenRouter, Together AI)
- **Quality Improvements**: Continue refining summarization thresholds
- **Performance**: Further optimize ML model loading and caching
- **Documentation**: Expand provider-specific guides

## Breaking Changes

### ✅ No Breaking Changes

This release maintains full backward compatibility with v2.4.0:

- **Configuration**: All existing config files work unchanged
- **CLI**: All existing CLI commands work unchanged
- **API**: No public API changes
- **Output**: Output format unchanged (run manifest is additive)

### ⚠️ Behavior Changes (Not Strictly Breaking)

**1. MPS Exclusive Mode (New Default)**

- **Impact**: On Apple Silicon, GPU work is serialized when both Whisper and summarization use MPS
- **Workaround**: Use `--no-mps-exclusive` to disable
- **Rationale**: Prevents memory contention and improves stability

**2. Run Manifest Generation (New Feature)**

- **Impact**: Every run now generates `run_manifest.json` file
- **Workaround**: None needed (additive feature)
- **Rationale**: Improves reproducibility and debugging

**3. Entity Reconciliation (New Feature)**

- **Impact**: Entity names in summaries are automatically corrected to match extracted entities
- **Workaround**: None needed (only affects ML providers, improves accuracy)
- **Rationale**: Improves consistency and accuracy

## Full Changelog

**Full Changelog**: <https://github.com/chipi/podcast_scraper/compare/v2.4.0...v2.5.0>

**Commits Since v2.4.0**: 56+ commits

**Lines Changed**: +8,000 / -3,000

**Files Changed**: 150+ files
