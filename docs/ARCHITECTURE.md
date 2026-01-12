# Podcast Scraper Architecture

> **Strategic Overview**: This document provides high-level architectural decisions, design principles, and system structure. For detailed implementation guides, see the [Development Guide](guides/DEVELOPMENT_GUIDE.md) and other specialized documents linked below.

## Navigation

This architecture document is the central hub for understanding the system. For detailed information, see:

### Core Documentation

- **[Development Guide](guides/DEVELOPMENT_GUIDE.md)** — Detailed implementation instructions, dependency management, code patterns, and development workflows
- **[Testing Strategy](TESTING_STRATEGY.md)** — Testing philosophy, test pyramid, and quality standards
- **[Testing Guide](guides/TESTING_GUIDE.md)** — Quick reference and test execution commands
  - [Unit Testing Guide](guides/UNIT_TESTING_GUIDE.md) — Unit test mocking patterns
  - [Integration Testing Guide](guides/INTEGRATION_TESTING_GUIDE.md) — Integration test guidelines
  - [E2E Testing Guide](guides/E2E_TESTING_GUIDE.md) — E2E server, real ML models
  - [Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md) — What to test, prioritization
- **[CI/CD](ci/CD.md)** — Continuous integration and deployment pipeline

### API Documentation

- **[API Reference](api/REFERENCE.md)** — Complete API documentation
- **[Configuration](api/CONFIGURATION.md)** — Configuration options and examples
- **[CLI Reference](api/CLI.md)** — Command-line interface documentation

### Feature Documentation

- **[Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md)** — Complete guide for implementing new providers (includes OpenAI example, testing, E2E server mocking)
- **[ML Provider Reference](guides/ML_PROVIDER_REFERENCE.md)** — Detailed ML pipeline reference and tuning
- **[Configuration API](api/CONFIGURATION.md)** — Configuration API reference (includes environment variables)

### Specifications

- **[PRDs](prd/index.md)** — Product Requirements Documents
- **[RFCs](rfc/index.md)** — Request for Comments (design decisions)

## Goals and Scope

- Provide a resilient pipeline that collects podcast episode transcripts from RSS feeds and fills gaps via Whisper transcription.
- Offer both CLI and Python APIs with a single configuration surface (`Config`) and deterministic filesystem layout.
- Keep the public surface area small (`Config`, `load_config_file`, `run_pipeline`, `service.run`, `service.run_from_config_file`, `cli.main`) while exposing well-factored submodules for advanced use.
- Provide a service API (`service.py`) optimized for non-interactive use (daemons, process managers) with structured error handling and exit codes.

## High-Level Flow

1. **Entry**: `podcast_scraper.cli.main` parses CLI args (optionally merging JSON/YAML configs) into a validated `Config` object and applies global logging preferences.
2. **Run orchestration**: `workflow.run_pipeline` coordinates the end-to-end job: output setup, RSS acquisition, episode materialization, transcript download, optional media download, optional audio preprocessing, optional Whisper transcription, optional metadata generation, optional summarization, and cleanup.
3. **Episode handling**: For each `Episode`, `episode_processor.process_episode_download` either saves an existing transcript or enqueues media for transcription.
4. **Audio Preprocessing** (RFC-040): When transcription is required, downloaded media is preprocessed (mono conversion, resampling, normalization) to reduce file size (<25MB common denominator) and improve transcription quality.
5. **Speaker detection** (RFC-010): When automatic speaker detection is enabled, host names are extracted from RSS author tags or NER. Guest names are extracted from episode metadata using NER.
6. **Transcription**: When enabled, `episode_processor.transcribe_media_to_text` generates transcripts using the configured provider (Whisper or OpenAI API).
7. **Metadata generation** (PRD-004/RFC-011): Per-episode metadata documents are generated in JSON/YAML format and stored in the `metadata/` subdirectory within the run directory.
8. **Summarization** (PRD-005/RFC-012): Episode transcripts are summarized using local transformer models (BART, LED) or API providers. Implements model-specific thresholds and transition zones for consistent quality. Summaries are integrated into the metadata documents.
9. **Metrics/Dashboard**: Pipeline execution metrics are collected and saved to `metrics.json` in the run directory.
10. **Progress/UI**: All long-running operations report progress through the pluggable factory in `progress.py`, defaulting to `tqdm` in the CLI.

### Pipeline Flow Diagram

```mermaid
flowchart TD
    Start([CLI Entry]) --> Parse[Parse CLI Args & Config Files]
    Parse --> Validate[Validate & Normalize Config]
    Validate --> Setup[Setup Output Directory]
    Setup --> FetchRSS[Fetch & Parse RSS Feed]
    FetchRSS --> ExtractEpisodes[Extract Episode Metadata]
    ExtractEpisodes --> DetectSpeakers{Speaker Detection Enabled?}
    DetectSpeakers -->|Yes| ExtractHosts[Extract Host Names from RSS]
    ExtractHosts --> ExtractGuests[Extract Guest Names via NER]
    DetectSpeakers -->|No| ProcessEpisodes[Process Episodes]
    ExtractGuests --> ProcessEpisodes
    ProcessEpisodes --> CheckTranscript{Transcript Available?}
    CheckTranscript -->|Yes| DownloadTranscript[Download Transcript]
    CheckTranscript -->|No| QueueWhisper[Queue for Whisper]
    DownloadTranscript --> SaveTranscript[Save Transcript File]
    QueueWhisper --> DownloadMedia[Download Media File]
    DownloadMedia --> Preprocess[Audio Preprocessing]
    Preprocess --> Transcribe[Transcription Provider]
    Transcribe --> FormatScreenplay[Format with Speaker Names]
    FormatScreenplay --> SaveTranscript
    SaveTranscript --> GenerateMetadata{Metadata Generation?}
    GenerateMetadata -->|Yes| CreateMetadata[Generate Metadata JSON/YAML]
    GenerateMetadata -->|No| Cleanup
    CreateMetadata --> GenerateSummary{Summarization Enabled?}
    GenerateSummary -->|Yes| Summarize[Generate Summary]
    GenerateSummary -->|No| Cleanup
    Summarize --> AddSummaryToMetadata[Add Summary to Metadata]
    AddSummaryToMetadata --> Cleanup[Cleanup Temp Files]
    Cleanup --> End([Complete])

    style Start fill:#e1f5ff
    style End fill:#d4edda
    style ProcessEpisodes fill:#fff3cd
    style Transcribe fill:#f8d7da
    style GenerateMetadata fill:#d1ecf1
```

- `cli.py`: Parse/validate CLI arguments, integrate config files, set up progress reporting, trigger `run_pipeline`. Optimized for interactive command-line use.
- `service.py`: Service API for programmatic/daemon use. Provides `service.run()` and `service.run_from_config_file()` functions that return structured `ServiceResult` objects. Works exclusively with configuration files (no CLI arguments), optimized for non-interactive use (supervisor, systemd, etc.). Entry point: `python -m podcast_scraper.service --config config.yaml`.
- `config.py`: Immutable Pydantic model representing all runtime options; JSON/YAML loader with strict validation and normalization helpers. Includes language configuration, NER settings, and speaker detection flags (RFC-010).
- `workflow.py`: Pipeline coordinator that orchestrates directory prep, RSS parsing, download concurrency, Whisper lifecycle, speaker detection coordination, and cleanup.
- `rss_parser.py`: Safe RSS/XML parsing, discovery of transcript/enclosure URLs, and creation of `Episode` models.
- `downloader.py`: HTTP session pooling with retry-enabled adapters, streaming downloads, and shared progress hooks.
- `episode_processor.py`: Episode-level decision logic, transcript storage, Whisper job management, delay handling, and file naming rules. Integrates detected speaker names into Whisper screenplay formatting.
- `filesystem.py`: Filename sanitization, output directory derivation, run suffix logic, and creation of `transcripts/` and `metadata/` subdirectories. Implements audio preprocessing impact tracking (RFC-040).
- `cache_manager.py`: Management of ML model caches (Whisper, Transformers, spaCy), providing disk usage statistics and cleaning functionality. Supports CLI cache management commands (`cache --status`, `cache --clean`).
- `preprocessing.py`: Audio preprocessing logic using `ffmpeg` to optimize files for transcription API compatibility (<25 MB for OpenAI). Implements mono conversion, resampling to 16kHz, normalization, and format conversion (RFC-040).
- `metrics.py`: Collection and storage of pipeline execution metrics, including processing times, file sizes, provider statistics, and preprocessing impact. Metrics are saved to `metrics.json` in the effective output directory.
- **Provider System** (RFC-013): Protocol-based provider architecture for transcription, speaker detection, and summarization. Each capability has a protocol interface (`TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`) and factory functions that create provider instances based on configuration. Providers implement `initialize()`, protocol methods (e.g., `transcribe()`, `summarize()`), and `cleanup()`. See [Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md) for details.
  - **Transcription Providers**: `transcription/whisper_provider.py` (local Whisper), `transcription/openai_provider.py` (OpenAI Whisper API)
  - **Speaker Detection Providers**: `speaker_detectors/ner_detector.py` (spaCy NER), `speaker_detectors/openai_detector.py` (OpenAI GPT)
  - **Summarization Providers**: `summarization/local_provider.py` (local transformers), `summarization/openai_provider.py` (OpenAI GPT)
- `whisper_integration.py`: Lazy loading of the third-party `openai-whisper` library, transcription invocation with language-aware model selection (preferring `.en` variants for English), and screenplay formatting helpers that use detected speaker names. Now accessed via `WhisperTranscriptionProvider` (provider pattern).
- `speaker_detection.py` (RFC-010): Named Entity Recognition using spaCy to extract PERSON entities from episode metadata, distinguish hosts from guests, and provide speaker names for Whisper screenplay formatting. spaCy is a required dependency. Now accessed via `NERSpeakerDetector` (provider pattern).
- `summarizer.py` (PRD-005/RFC-012): Episode summarization using local transformer models (BART, PEGASUS, LED) to generate concise summaries from transcripts. Implements a hybrid map-reduce strategy. Now accessed via `TransformersSummarizationProvider` (provider pattern). See [ML Provider Reference](guides/ML_PROVIDER_REFERENCE.md) for details.
- `progress.py`: Minimal global progress publishing API so callers can swap in alternative UIs.
- `models.py`: Simple dataclasses (`RssFeed`, `Episode`, `TranscriptionJob`) shared across modules. May be extended to include detected speaker metadata.
- `metadata.py` (PRD-004/RFC-011): Per-episode metadata document generation, capturing feed-level and episode-level information, detected speaker names, transcript sources, processing metadata, and optional summaries in structured JSON/YAML format. Opt-in feature for backwards compatibility.

### Module Dependencies Diagram

```mermaid
graph TB
    subgraph "Public API"
        CLI[cli.py]
        Config[config.py]
        Workflow[workflow.py]
        CacheManager[cache_manager.py]
    end

    subgraph "Core Processing"
        RSSParser[rss_parser.py]
        EpisodeProc[episode_processor.py]
        Downloader[downloader.py]
        Preprocessing[preprocessing.py]
        Metrics[metrics.py]
    end

    subgraph "Support Modules"
        Filesystem[filesystem.py]
        Models[models.py]
        Progress[progress.py]
    end

    subgraph "Provider System"
        TranscriptionFactory[transcription/factory.py]
        SpeakerFactory[speaker_detectors/factory.py]
        SummaryFactory[summarization/factory.py]
        WhisperProvider[transcription/whisper_provider.py]
        OpenAITranscription[transcription/openai_provider.py]
        NERProvider[speaker_detectors/ner_detector.py]
        OpenAISpeaker[speaker_detectors/openai_detector.py]
        LocalSummary[summarization/local_provider.py]
        OpenAISummary[summarization/openai_provider.py]
    end

    subgraph "Optional Features"
        Whisper[whisper_integration.py]
        SpeakerDetect[speaker_detection.py]
        Metadata[metadata.py]
        Summarizer[summarizer.py]
    end

```python

    CLI --> Config
    CLI --> Workflow
    CLI --> Progress
    CLI --> CacheManager
    Workflow --> RSSParser
    Workflow --> EpisodeProc
    Workflow --> Downloader
    Workflow --> Preprocessing
    Workflow --> Metrics
    Workflow --> TranscriptionFactory
    Workflow --> SpeakerFactory
    Workflow --> SummaryFactory
    TranscriptionFactory --> WhisperProvider
    TranscriptionFactory --> OpenAITranscription
    SpeakerFactory --> NERProvider
    SpeakerFactory --> OpenAISpeaker
    SummaryFactory --> LocalSummary
    SummaryFactory --> OpenAISummary
    WhisperProvider --> Whisper
    NERProvider --> SpeakerDetect
    LocalSummary --> Summarizer
    Workflow --> Metadata
    Workflow --> Filesystem
    Workflow --> Models
    Workflow --> Progress
    EpisodeProc --> Downloader
    EpisodeProc --> Filesystem
    EpisodeProc --> Whisper
    EpisodeProc --> Models
    RSSParser --> Models
    Whisper --> SpeakerDetect
    Metadata --> Summarizer
    Metadata --> Models
    Summarizer --> Models
    SpeakerDetect --> Models
    CacheManager --> Filesystem

```python
- **Typed, immutable configuration**: `Config` is a frozen Pydantic model, ensuring every module receives canonicalized values (e.g., normalized URLs, integer coercions, validated Whisper models). This centralizes validation and guards downstream logic.
- **Resilient HTTP interactions**: A per-thread `requests.Session` with exponential backoff retry (`LoggingRetry`) handles transient network issues while logging retries for observability.
- **Concurrent transcript pulls**: Transcript downloads are parallelized via `ThreadPoolExecutor`, guarded with locks when mutating shared counters/job queues. Whisper remains sequential to avoid GPU/CPU thrashing and to keep the UX predictable.
- **Deterministic filesystem layout**: Output folders follow `output/rss_<host>_<hash>` conventions. Optional `run_id` and Whisper suffixes create run-scoped subdirectories while `sanitize_filename` protects against filesystem hazards.
- **Dry-run and resumability**: `--dry-run` walks the entire plan without touching disk, while `--skip-existing` short-circuits work per episode, making repeated runs idempotent.
- **Pluggable progress/UI**: A narrow `ProgressFactory` abstraction lets embedding applications replace the default `tqdm` progress without touching business logic.
- **Optional Whisper dependency**: Whisper is imported lazily and guarded so environments without GPU support or `openai-whisper` can still run transcript-only workloads.
- **Optional summarization dependency** (PRD-005/RFC-012): Summarization requires `torch` and `transformers` dependencies and is imported lazily. When dependencies are unavailable, summarization is gracefully skipped. Models are automatically selected based on available hardware (MPS for Apple Silicon, CUDA for NVIDIA GPUs, CPU fallback). See [ML Provider Reference](guides/ML_PROVIDER_REFERENCE.md) for details.
- **Language-aware processing** (RFC-010): A single `language` configuration drives both Whisper model selection (preferring English-only `.en` variants) and NER model selection (e.g., `en_core_web_sm`), ensuring consistent language handling across the pipeline.
- **Automatic speaker detection** (RFC-010): Named Entity Recognition extracts speaker names from episode metadata transparently. Manual speaker names (`--speaker-names`) are ONLY used as fallback when automatic detection fails, not as override. spaCy is a required dependency for speaker detection.
- **Host/guest distinction**: Host detection prioritizes RSS author tags (channel-level only) as the most reliable source, falling back to NER extraction from feed metadata when author tags are unavailable. Guests are always detected from episode-specific metadata using NER, ensuring accurate speaker labeling in Whisper screenplay output.
- **Provider-based architecture** (RFC-013): All capabilities (transcription, speaker detection, summarization) use a protocol-based provider system. Providers are created via factory functions based on configuration, allowing pluggable implementations (e.g., Whisper vs OpenAI for transcription, NER vs OpenAI for speaker detection, local transformers vs OpenAI for summarization). Providers implement consistent interfaces (`initialize()`, protocol methods, `cleanup()`) ensuring type safety and easy testing. See [Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md) for complete implementation details.
- **Local-first summarization** (PRD-005/RFC-012): Summarization defaults to local transformer models for privacy and cost-effectiveness. API-based providers (OpenAI) are supported via the provider system. Long transcripts are handled via chunking strategies, and memory optimization is applied for GPU backends (CUDA/MPS). Models are automatically cached and reused across runs, with cache management utilities available via CLI and programmatic APIs.

## Third-Party Dependencies

The project uses a layered dependency approach: **core dependencies** (always required) provide essential functionality, while **ML dependencies** (optional) enable advanced features like transcription and summarization.

**Core Dependencies**: `requests`, `pydantic`, `defusedxml`, `tqdm`, `platformdirs`, `PyYAML`

**ML Dependencies** (optional, install via `pip install -e .[ml]`): `openai-whisper`, `spacy`, `torch`, `transformers`, `sentencepiece`, `accelerate`, `protobuf`

**API Provider Dependencies** (optional, for OpenAI providers): `openai` (OpenAI Python SDK)

For detailed dependency information including rationale, alternatives considered, version requirements, and dependency management philosophy, see [Dependencies Guide](guides/DEPENDENCIES_GUIDE.md).

## Constraints and Assumptions

- Python 3.10+ with third-party packages: `requests`, `tqdm`, `defusedxml`, `platformdirs`, `pydantic`, `PyYAML`, `spacy` (required for speaker detection), and optionally `openai-whisper` + `ffmpeg` when transcription is required, and optionally `torch` + `transformers` when summarization is required.
- Network-facing operations assume well-formed HTTPS endpoints; malformed feeds raise early during parsing to avoid partial state.
- Whisper transcription supports multiple languages via `language` configuration, with English (`"en"`) as the default. Model selection automatically prefers `.en` variants for English content. Transcription remains sequential by design; concurrent transcription is intentionally out of scope due to typical hardware limits.
- Speaker name detection via NER (RFC-010) requires spaCy. When automatic detection fails, the system falls back to manual speaker names (if provided) or default `["Host", "Guest"]` labels.
- Output directories must live in safe roots (cwd, user home, or platform data/cache dirs); other locations trigger warnings for operator review.

### Configuration Flow

```mermaid

flowchart TD
    Input[CLI Args + Config Files] --> Merge[Merge Sources]
    Merge --> Validate[Pydantic Validation]
    Validate --> Normalize[Normalize Values]
    Normalize --> Config[Immutable Config Object]
    Config --> Workflow[workflow.run_pipeline]
    Config --> EpisodeProc[episode_processor]
    Config --> RSSParser[rss_parser]
    Config --> Downloader[downloader]
    Config --> Whisper[whisper_integration]
    Config --> SpeakerDetect[speaker_detection]
    Config --> Metadata[metadata]
    Config --> Summarizer[summarizer]

    style Input fill:#e1f5ff
    style Config fill:#fff3cd
    style Validate fill:#f8d7da

```python

- `models.Episode` encapsulates the RSS item, chosen transcript URLs, and media enclosure metadata, keeping parsing concerns separate from processing. May be extended to include detected speaker names (RFC-010).
- Transcript filenames follow `<####> - <episode_title>.<ext>` format with extensions inferred from declared types, HTTP headers, or URL heuristics. Transcripts are stored in the `transcripts/` subdirectory.
- Whisper output names follow the same format as transcript files. When `run_id` is specified or providers are configured, the run suffix (e.g., `run_2.4.0_w_base.en_tf_bart-large-cnn`) differentiates runs.
- Temporary media downloads land in `<output>/.tmp_media/` and are cleaned up (best effort) after transcription completes.
- Episode metadata documents (per PRD-004/RFC-011) are generated when `generate_metadata` is enabled, storing detected speaker names, feed information, transcript sources, and other episode details in the `metadata/` subdirectory in JSON/YAML format. When summarization is enabled, metadata documents include summary and key takeaways fields with model information and generation timestamps.
- Pipeline metrics are saved to `metrics.json` in the effective output directory (same level as `transcripts/` and `metadata/` subdirectories), capturing processing times, file sizes, provider statistics, and preprocessing impact (RFC-040).

### Filesystem Layout (v2.4.0+)

```mermaid

graph TD
    Root[output/rss_hostname_hash/] --> RunDir{Run ID?}
    RunDir -->|Yes| RunSubdir[run_id/]
    RunDir -->|No| BaseDir[Base Directory]
    RunSubdir --> Subdirs[Organized Subdirectories]
    BaseDir --> Subdirs
    Subdirs --> TranscriptsDir[transcripts/]
    Subdirs --> MetadataDir[metadata/]
    TranscriptsDir --> T1["0001 - Title.txt<br/>0002 - Title.txt"]
    MetadataDir --> M1["0001 - Title.metadata.json<br/>0002 - Title.metadata.json"]
    Subdirs --> MetricsFile[metrics.json]
    Root --> TempDir[.tmp_media/]
    TempDir --> TempFiles[Temporary Media Files<br/>Cleaned Up After Use]

    style Root fill:#e1f5ff
    style Subdirs fill:#fff3cd
    style TempDir fill:#f8d7da
    style MetadataDir fill:#d1ecf1

```
- Transcript/Media downloads log warnings rather than hard-fail the pipeline, allowing other episodes to proceed.
- Filesystem operations sanitize user-provided paths, emit warnings when outside trusted roots, and handle I/O errors gracefully.
- Unexpected exceptions inside worker futures are caught and logged without terminating the executor loop.

For detailed error handling patterns and implementation guidelines, see [Development Guide - Error Handling](guides/DEVELOPMENT_GUIDE.md#error-handling).

## Extensibility Points

- **Configuration**: Extend `Config` (and CLI) when introducing new features; validation rules keep downstream logic defensive. Language and NER configuration (RFC-010) demonstrate this pattern.
- **Progress**: Replace `progress.set_progress_factory` to integrate with custom UIs or disable progress output entirely.
- **Download strategy**: `downloader` centralizes HTTP behavior—alternate adapters or auth strategies can be injected by decorating `fetch_url`/`http_get`.
- **Episode transforms**: New transcript processors can reuse `models.Episode` and `filesystem` helpers without modifying the main pipeline.
- **CLI embedding**: `cli.main` accepts override callables (`apply_log_level_fn`, `run_pipeline_fn`, `logger`) to facilitate testing and reuse from other entry points.
- **Speaker detection** (RFC-010): NER implementation is modular and can be extended with custom heuristics, additional entity types, or alternative NLP libraries. Configuration allows disabling detection behavior or providing manual fallback names.
- **Language support**: Language configuration drives both Whisper and NER model selection, enabling multi-language support through consistent configuration. New languages can be added by extending model selection logic and spaCy model support.
- **Metadata generation** (PRD-004/RFC-011): Metadata document generation is opt-in and can be extended with additional fields or alternative output formats. The schema is versioned to support future evolution.
- **Provider system** (RFC-013): The provider architecture enables extensibility for all capabilities. New providers can be added by implementing protocol interfaces and registering in factory functions. The system supports both local implementations (Whisper, spaCy NER, local transformers) and API-based providers (OpenAI). E2E testing infrastructure includes mock endpoints for API providers, allowing tests to run without real API calls. See [Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md) for complete implementation patterns, testing strategies, and E2E server mocking details.
- **Summarization** (PRD-005/RFC-012): Summarization is opt-in and integrated with metadata generation. Local transformer models are preferred for privacy and cost-effectiveness, with automatic hardware-aware model selection. The implementation supports multiple model architectures (BART, DistilBART) and can be extended with additional models or API-based providers via the provider system. Long transcript handling via chunking strategies ensures scalability. Models are automatically cached locally and reused across runs. Cache management utilities (`get_cache_size`, `prune_cache`, `format_cache_size`) are available via CLI (`--cache-info`, `--prune-cache`) and programmatically for disk space management.

## Testing

The project follows a three-tier testing strategy (Unit, Integration, E2E). For comprehensive testing information:

| Document | Purpose |
| ---------- | --------- |
| **[Testing Strategy](TESTING_STRATEGY.md)** | Testing philosophy, test pyramid, decision criteria |
| **[Testing Guide](guides/TESTING_GUIDE.md)** | Quick reference, test execution commands |
| **[Unit Testing Guide](guides/UNIT_TESTING_GUIDE.md)** | Unit test mocking patterns and isolation |
| **[Integration Testing Guide](guides/INTEGRATION_TESTING_GUIDE.md)** | Integration test guidelines |
| **[E2E Testing Guide](guides/E2E_TESTING_GUIDE.md)** | E2E server, real ML models |
| **[Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md)** | What to test, prioritization |
| **[Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md)** | Provider-specific testing

````
