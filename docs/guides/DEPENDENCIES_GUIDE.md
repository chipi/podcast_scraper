# Dependencies Guide

This guide documents the key third-party dependencies used in the podcast scraper project,
including the rationale for their selection, alternatives considered, and dependency
management philosophy.

For high-level architectural decisions, see [Architecture](../ARCHITECTURE.md). For general
development practices, see [Development Guide](DEVELOPMENT_GUIDE.md).

## Overview

The project uses a **layered dependency approach**: core dependencies (always required)
provide essential functionality, while ML dependencies (optional) enable advanced features
like transcription and summarization.

## Core Dependencies (Always Required)

### `requests` (>=2.31.0)

- **Purpose**: HTTP client for downloading RSS feeds, transcripts, and media files
- **Why chosen**: Industry-standard library with excellent session management, connection
  pooling, and retry capabilities. Used throughout `downloader.py` and `rss_parser.py` for
  all network operations.
- **Key features utilized**: Session pooling with custom retry adapters (`LoggingRetry`),
  streaming downloads for large media files, configurable timeouts and headers
- **Alternatives considered**: `urllib3` (too low-level), `httpx` (less mature at project start)

### `pydantic` (>=2.6.0)

- **Purpose**: Data validation, serialization, and configuration management via the `Config`
  model
- **Why chosen**: Provides immutable, type-safe configuration with automatic validation,
  JSON/YAML parsing, and excellent error messages. Central to the architecture's "typed,
  immutable configuration" design principle.
- **Key features utilized**: Frozen dataclasses, field validators, JSON/YAML serialization,
  nested model validation, type coercion
- **Alternatives considered**: `dataclasses` (no validation), `attrs` (less validation
  features), `marshmallow` (more verbose)

### `defusedxml` (>=0.7.1)

- **Purpose**: Safe XML/RSS parsing that prevents XML bomb attacks and entity expansion vulnerabilities
- **Why chosen**: Security-first RSS parsing is critical when processing untrusted feeds
  from the internet. Drop-in replacement for stdlib XML parsers with security hardening.
- **Key features utilized**: Safe ElementTree parsing in `rss_parser.py`, automatic protection against XXE attacks
- **Alternatives considered**: Standard library `xml.etree` (vulnerable), `lxml` (heavier dependency)

### `tqdm` (>=4.66.0)

- **Purpose**: Progress bars for long-running operations (downloads, transcription)
- **Why chosen**: Rich, customizable progress visualization with minimal API surface.
  Integrates cleanly via the `progress.py` pluggable factory pattern.
- **Key features utilized**: Multi-level progress tracking, dynamic updates, thread-safe counters, custom formatting
- **Alternatives considered**: `click.progressbar` (less flexible), `rich.progress` (heavier dependency)

### `platformdirs` (>=3.11.0)

- **Purpose**: Cross-platform directory path resolution for output, cache, and configuration files
- **Why chosen**: Handles OS-specific conventions (Linux XDG, macOS Application Support,
  Windows AppData) transparently. Essential for determining safe output roots and
  validating user-provided paths.
- **Key features utilized**: User data directory resolution, cache directory paths
- **Alternatives considered**: Manual path construction (not portable), `appdirs` (unmaintained)

### `PyYAML` (>=6.0)

- **Purpose**: YAML configuration file parsing alongside JSON support
- **Why chosen**: YAML provides more human-friendly configuration syntax than JSON, with
  comments and multi-line string support. Widely adopted in operations/DevOps contexts.
- **Key features utilized**: Safe YAML loading, round-trip with Pydantic models
- **Alternatives considered**: JSON-only (less user-friendly), TOML (less mature Python support)

## ML Dependencies (Optional, Install via `pip install -e .[ml]`)

### `openai-whisper` (>=20231117)

- **Purpose**: Automatic speech recognition for podcast transcription fallback
- **Why chosen**: State-of-the-art open-source ASR with multiple model sizes, multilingual
  support, and screenplay formatting. Local-first approach ensures privacy and no API
  costs.
- **Key features utilized**: Model selection (tiny→large), language detection, speaker
  diarization hints, `.en` model variants for English optimization
- **Alternatives considered**: Google Speech-to-Text (API costs), Azure Speech (API costs),
  Vosk (less accurate), Mozilla DeepSpeech (deprecated)
- **Lazy loading**: Imported conditionally in `whisper_integration.py` to avoid hard dependency

### `spacy` (>=3.7.0)

- **Purpose**: Named Entity Recognition (NER) for automatic speaker detection from episode metadata
- **Why chosen**: Production-ready NLP library with pre-trained models for person name
  extraction. Fast, accurate, and supports multiple languages via consistent model naming
  (`en_core_web_sm`, `es_core_news_sm`, etc.).
- **Key features utilized**: PERSON entity extraction, language-aware model selection, efficient batch processing
- **Alternatives considered**: `transformers` NER (overkill for this use case), regex
  patterns (too brittle), `nltk` (less accurate)
- **Model requirements**: Language-specific models must be downloaded separately (e.g.,
  `python -m spacy download en_core_web_sm`)

### `torch` (>=2.0.0) and `transformers` (>=4.30.0)

- **Purpose**: Deep learning framework (torch) and pre-trained transformer models (transformers) for episode summarization
- **Why chosen**: `transformers` provides access to production-ready summarization models
  (BART, PEGASUS, LED) with automatic caching and hardware acceleration. `torch` is the de
  facto standard for deep learning in Python with excellent MPS (Apple Silicon) and CUDA
  (NVIDIA) support.
- **Key features utilized**:
  - **torch**: Device detection (MPS/CUDA/CPU), memory-efficient inference, gradient-free execution
  - **transformers**: Model auto-loading, tokenization, generation with beam search,
    automatic model caching (`~/.cache/huggingface/`)
- **Models used**: BART-large (map phase), LED/long-fast (reduce phase), PEGASUS (alternative)
- **Alternatives considered**: OpenAI API (costs/privacy), Anthropic Claude (costs/privacy), spaCy summarization (less sophisticated)
- **Lazy loading**: Imported conditionally in `summarizer.py` to avoid hard dependency when summarization is disabled

### `sentencepiece` (>=0.1.99)

- **Purpose**: Tokenization for certain transformer models (required by PEGASUS and others)
- **Why chosen**: Required dependency for models using SentencePiece tokenization. Lightweight and efficient.
- **Key features utilized**: Automatic integration via `transformers` library

### `accelerate` (>=0.20.0)

- **Purpose**: Optimized model loading and inference acceleration for large transformer models
- **Why chosen**: Reduces model loading time and memory usage, especially for 16-bit
  inference and device mapping. Official Hugging Face library for production deployments.
- **Key features utilized**: Fast model initialization, memory optimization for limited-RAM systems

### `protobuf` (>=3.20.0)

- **Purpose**: Protocol buffer serialization (required by some transformer model configurations)
- **Why chosen**: Transitive dependency for certain model formats. Pinned to avoid version conflicts.

## Dependency Management Philosophy

1. **Core vs Optional**: Core dependencies are minimal and stable. Heavy ML dependencies
   are optional (`pip install -e .[ml]`) to avoid forcing users to install GB-sized
   packages when only transcript downloading is needed.

2. **Version pinning**: Minimum versions are specified, but upper bounds are avoided to
   allow users to upgrade independently. Major version changes (e.g., Pydantic v2) are
   tracked carefully.

3. **Security**: Security-focused libraries (`defusedxml`) are preferred. Regular updates
   via `pip-audit` and `bandit` (dev tools) ensure vulnerability detection.

4. **Lazy loading**: Optional dependencies (`openai-whisper`, `torch`, `transformers`,
   `spacy`) are imported lazily with graceful fallbacks when unavailable.

5. **Platform compatibility**: All dependencies support Linux, macOS, and Windows.
   Platform-specific optimizations (MPS, CUDA) are detected at runtime.

## Development Dependencies (Optional, Install via `pip install -e .[dev]`)

### `pytest-json-report` (>=1.5.0,<2.0.0)

- **Purpose**: Generate structured JSON reports from pytest test runs for metrics collection and analysis
- **Why chosen**: Provides machine-readable test metrics (pass/fail counts, durations, flaky test detection) that
  integrate with our metrics collection system (RFC-025). Used in nightly workflow for comprehensive test metrics
  tracking.
- **Key features utilized**: JSON report generation (`--json-report`), test outcome tracking, rerun detection for
  flaky tests
- **Alternatives considered**: Custom pytest plugins (more maintenance), JUnit XML only (less structured data)
- **Usage**: Automatically used in nightly workflow via `--json-report --json-report-file=reports/pytest.json`

## Installation

### Core Dependencies Only

````bash
pip install -e .
```text
```bash
pip install -e .[ml]
```text
```bash
pip install -e .[dev]
```text
- [Architecture](../ARCHITECTURE.md) - High-level system design and dependency overview
- [Development Guide](DEVELOPMENT_GUIDE.md) - General development practices
- [Summarization Guide](SUMMARIZATION_GUIDE.md) - Details on ML dependencies for summarization
````
