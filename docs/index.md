# Podcast Scraper

**Download and process podcast transcripts with ease.**

Podcast Scraper is a Python tool that downloads transcripts for every episode in a podcast RSS feed.
It understands Podcasting 2.0 transcript tags, resolves relative URLs, and can fall back to
Whisper transcription when episodes lack published transcripts.

> **⚠️ Important:** This project is intended for **personal, non-commercial use only**. All downloaded
> content must remain local and not be shared or redistributed. See [Legal Notice & Appropriate
> Use](LEGAL.md) for details.

---

## ✨ Key Features

- **Transcript Downloads** — Automatic detection and download of podcast transcripts from RSS feeds
- **Whisper Fallback** — Generate transcripts using OpenAI Whisper when none exist
- **Speaker Detection** — Automatic speaker name detection using Named Entity Recognition (NER)
- **Screenplay Formatting** — Format Whisper transcripts as dialogue with speaker labels
- **Episode Summarization** — Generate concise summaries using local transformer models (BART + LED)
- **Metadata Generation** — Create database-friendly JSON/YAML metadata documents per episode
- **Multi-threaded Downloads** — Concurrent processing with configurable worker pools
- **Resumable Operations** — Skip existing files, reuse media, and handle interruptions gracefully
- **Configuration Files** — JSON/YAML config support for repeatable workflows
- **Service Mode** — Non-interactive daemon mode for automation and process managers

---

## 🚀 Quick Start

### Installation

```bash

# Clone or download the repository

git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Install the package

pip install -e .

# For Whisper transcription, ensure ffmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

```

## Basic Usage

```bash

# Download transcripts from a podcast

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --max-episodes 10 \
  --output-dir ./my_transcripts

# Use Whisper when transcripts are missing

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --whisper-model base

# Generate metadata and summaries

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```

## Configuration File

```yaml

# config.yaml

rss: https://example.com/feed.xml
output_dir: ./transcripts
max_episodes: 50
transcribe_missing: true
whisper_model: base
screenplay: true
num_speakers: 2
auto_speakers: true
generate_metadata: true
generate_summaries: true
workers: 4
skip_existing: true
```

**Example configs:**

- [config.example.json](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.json) - JSON format
- [config.example.yaml](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.yaml) - YAML format

---

## 📚 Documentation

### Getting Started

| Resource | Description |
| -------- | ----------- |
| **[Quick Start](#quick-start)** | Installation, basic usage, and first commands |
| **[Configuration Guide](api/CONFIGURATION.md)** | Complete configuration options and examples |
| **[CLI Reference](api/CLI.md)** | Command-line interface documentation |
| **[Python API](api/REFERENCE.md)** | Public API for programmatic usage |
| **[Legal Notice](LEGAL.md)** | ⚠️ Important usage restrictions and fair use |

### User Guides

| Guide | Description |
| ----- | ----------- |
| **[Service Mode](api/SERVICE.md)** | Running as daemon or service (systemd, supervisor, cron) |
| **[Service Examples](#service-daemon-examples)** | systemd and supervisor configuration templates |
| **[Advanced Features](#advanced-features)** | Speaker detection, summarization, language support |
| **[Configuration File](#configuration-file)** | JSON/YAML config examples |
| **[Configuration API](api/CONFIGURATION.md)** | Configuration API reference (includes environment variables) |

### For Developers

| Resource | Description |
| -------- | ----------- |
| **[Quick Reference](guides/QUICK_REFERENCE.md)** | ⭐ One-page cheat sheet for common commands |
| **[Contributing Guide](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)** | Development workflow, code style, testing requirements |
| **[Architecture Overview](ARCHITECTURE.md)** | High-level system design and module responsibilities |
| **[Testing Strategy](TESTING_STRATEGY.md)** | Test coverage, quality assurance, and testing guidelines |
| **[Testing Guide](guides/TESTING_GUIDE.md)** | Detailed test execution, fixtures, and coverage information |
| **[CI/CD Pipeline](CI_CD.md)** | GitHub Actions workflows, parallel execution |
| **[Development Guide](guides/DEVELOPMENT_GUIDE.md)** | Development environment setup and tooling |
| **[Dependencies Guide](guides/DEPENDENCIES_GUIDE.md)** | Third-party dependencies, rationale, and management |
| **[Troubleshooting](guides/TROUBLESHOOTING.md)** | Common issues and solutions |
| **[Glossary](guides/GLOSSARY.md)** | Key terms and concepts |
| **[Markdown Linting](guides/MARKDOWN_LINTING_GUIDE.md)** | Markdown linting practices and workflows |
| **[Summarization Guide](guides/SUMMARIZATION_GUIDE.md)** | Summarization implementation details |
| **[API Boundaries](api/BOUNDARIES.md)** | API design principles and stability guarantees |
| **[API Migration Guide](api/MIGRATION_GUIDE.md)** | Upgrading between versions |
| **[API Versioning](api/VERSIONING.md)** | Versioning strategy and compatibility |

### Provider System

| Guide | Description |
| ----- | ----------- |
| **[Provider Configuration Quick Reference](guides/PROVIDER_CONFIGURATION_QUICK_REFERENCE.md)** | **Quick guide for configuring providers via CLI, config files, and programmatically** |
| **[Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md)** | Complete guide for implementing new providers (includes OpenAI example, testing, E2E server mocking) |
| **[Protocol Extension Guide](guides/PROTOCOL_EXTENSION_GUIDE.md)** | Extending protocols and adding new methods to providers |

### API Reference

Complete documentation of all public modules:

| Module | Description |
| ------ | ----------- |
| **[Core API](api/CORE.md)** | Main entry point (`run_pipeline`, `Config`) |
| **[Configuration](api/CONFIGURATION.md)** | Configuration model and file loading |
| **[Service API](api/SERVICE.md)** | Non-interactive service mode for daemons |
| **[CLI Interface](api/CLI.md)** | Command-line interface |
| **[Models](api/MODELS.md)** | Data models (Episode, RssFeed, etc.) |

### Product & Technical Specs

#### Product Requirements (PRDs)

PRDs define the **what** and **why** behind each major feature:

| PRD | Title | Version | Description |
| --- | ----- | ------- | ----------- |
| **[PRD-001](prd/PRD-001-transcript-pipeline.md)** | Transcript Acquisition Pipeline | v2.0.0 | Core pipeline for downloading published transcripts |
| **[PRD-002](prd/PRD-002-whisper-fallback.md)** | Whisper Fallback Transcription | v2.0.0 | Automatic transcription when transcripts are missing |
| **[PRD-003](prd/PRD-003-user-interface-config.md)** | User Interfaces & Configuration | v2.0.0 | CLI interface and config file support |
| **[PRD-004](prd/PRD-004-metadata-generation.md)** | Per-Episode Metadata Generation | v2.2.0 | Database-friendly metadata documents |
| **[PRD-005](prd/PRD-005-episode-summarization.md)** | Episode Summarization | v2.3.0 | Automatic summaries using transformer models |
| **[PRD-006](prd/PRD-006-openai-provider-integration.md)** | OpenAI Provider Integration | Draft | Add OpenAI API as optional provider |
| **[PRD-007](prd/PRD-007-ai-experiment-pipeline.md)** | AI Experiment Pipeline | Draft | Configuration-driven experiment pipeline |

[View all PRDs →](prd/index.md)

#### Technical Specifications (RFCs)

RFCs define the **how** behind each feature implementation:

| RFC | Title | Version | Description |
| --- | ----- | ------- | ----------- |
| **[RFC-001](rfc/RFC-001-workflow-orchestration.md)** | Workflow Orchestration | v2.0.0 | Central orchestrator for transcript pipeline |
| **[RFC-002](rfc/RFC-002-rss-parsing.md)** | RSS Parsing & Episode Modeling | v2.0.0 | RSS feed parsing and episode data model |
| **[RFC-003](rfc/RFC-003-transcript-downloads.md)** | Transcript Download Processing | v2.0.0 | Resilient transcript downloads with retries |
| **[RFC-004](rfc/RFC-004-filesystem-layout.md)** | Filesystem Layout & Run Management | v2.0.0 | Output directory structure and run scoping |
| **[RFC-005](rfc/RFC-005-whisper-integration.md)** | Whisper Integration Lifecycle | v2.0.0 | Whisper model loading and transcription |
| **[RFC-006](rfc/RFC-006-screenplay-formatting.md)** | Whisper Screenplay Formatting | v2.0.0 | Speaker-attributed transcript formatting |
| **[RFC-007](rfc/RFC-007-cli-interface.md)** | CLI Interface & Validation | v2.0.0 | Command-line argument parsing |
| **[RFC-008](rfc/RFC-008-config-model.md)** | Configuration Model & Validation | v2.0.0 | Pydantic-based configuration |
| **[RFC-009](rfc/RFC-009-progress-integration.md)** | Progress Reporting Integration | v2.0.0 | Pluggable progress reporting |
| **[RFC-010](rfc/RFC-010-speaker-name-detection.md)** | Automatic Speaker Name Detection | v2.1.0 | NER-based host/guest identification |
| **[RFC-011](rfc/RFC-011-metadata-generation.md)** | Per-Episode Metadata Generation | v2.2.0 | Structured metadata document generation |
| **[RFC-012](rfc/RFC-012-episode-summarization.md)** | Episode Summarization | v2.3.0 | Local transformer-based summarization |
| **[RFC-013](rfc/RFC-013-openai-provider-implementation.md)** | OpenAI Provider Implementation | Draft | Technical design for OpenAI API providers |
| **[RFC-015](rfc/RFC-015-ai-experiment-pipeline.md)** | AI Experiment Pipeline | Draft | Configuration-driven experiment pipeline |
| **[RFC-016](rfc/RFC-016-modularization-for-ai-experiments.md)** | Modularization for AI Experiments | Draft | Provider system architecture |
| **[RFC-017](rfc/RFC-017-prompt-management.md)** | Prompt Management | Draft | Versioned, parameterized prompt management |

[View all RFCs →](rfc/index.md)

### Release History

| Version | Date | Highlights |
| ------- | ---- | ---------- |
| **[v2.3.2](releases/RELEASE_v2.3.2.md)** | Latest | Latest release |
| **[v2.3.1](releases/RELEASE_v2.3.1.md)** | - | Patch release |
| **[v2.3.0](releases/RELEASE_v2.3.0.md)** | - | Episode summarization, public API, cleaned transcripts |
| **[v2.2.0](releases/RELEASE_v2.2.0.md)** | - | Metadata generation, code quality improvements |
| **[v2.1.0](releases/RELEASE_v2.1.0.md)** | - | Automatic speaker detection using NER |
| **[v2.0.1](releases/RELEASE_v2.0.1.md)** | - | Bug fixes and stability improvements |
| **[v2.0.0](releases/RELEASE_v2.0.0.md)** | - | Modular architecture, public API foundation |
| **[v1.0.0](releases/RELEASE_v1.0.0.md)** | - | Initial release |

---

## 📦 Service & Daemon Examples

Run podcast_scraper as a background service:

### systemd Service

```ini
[Unit]
Description=Podcast Scraper Service
After=network.target

[Service]
Type=oneshot
User=your_username
WorkingDirectory=/path/to/working/directory
ExecStart=/usr/bin/python3 -m podcast_scraper.service --config /path/to/config.yaml
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Supervisor Configuration

```ini
[program:podcast_scraper]
command=/usr/bin/python3 -m podcast_scraper.service --config /path/to/config.yaml
directory=/path/to/working/directory
user=your_username
autostart=true
autorestart=true
stdout_logfile=/var/log/podcast_scraper/stdout.log
stderr_logfile=/var/log/podcast_scraper/stderr.log
```

---

## 🎙️ Advanced Features

### Speaker Detection

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --num-speakers 2 \
  --auto-speakers
```

### Summarization

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-summaries \
  --summary-model bart-large \
  --summary-device mps
```

---

## 📊 Evaluation Scripts

| Script | Purpose |
| ------ | ------- |
| **eval_cleaning.py** | Evaluate transcript cleaning quality |
| **eval_summaries.py** | Evaluate summarization quality |

```bash

# Evaluate transcript cleaning

python scripts/eval_cleaning.py --episode ep01

# Evaluate summarization quality

python scripts/eval_summaries.py --map-model bart-large --reduce-model long-fast
```

---

## 🛠️ Development

```bash
source .venv/bin/activate
pip install -e .

## Run full CI suite locally

```bash

make ci
```

## Python API

```python

import podcast_scraper

cfg = podcast_scraper.Config(
    rss="https://example.com/feed.xml",
    output_dir="./out",
    transcribe_missing=True,
    generate_metadata=True
)

podcast_scraper.run_pipeline(cfg)

```json
**Other entry points:**

- `podcast_scraper.cli.main(argv)` — CLI entry point

[Complete API reference →](api/REFERENCE.md)

---

## 🐳 Docker

The Docker image uses the service API, which requires a configuration file:

```bash

docker build -t podcast-scraper -f Dockerfile .

docker run --rm \
  -v "$(pwd)/output_docker:/app/output" \
  -v "$(pwd)/config.yaml:/app/config.yaml:ro" \
  podcast-scraper \
  --config /app/config.yaml
```

---

## 👩‍💻 Contributing

```bash

# Set up development environment

make init

# Run full CI suite (matches GitHub Actions)

make ci

# Common commands

make format        # Auto-format code
make lint          # Run linting
make type          # Type checking
make test-unit     # Run unit tests with coverage
make docs          # Build documentation
```

**Resources:**

- **[Contributing Guide](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)** — Development workflow
- **[Architecture](ARCHITECTURE.md)** — Module boundaries and design
- **[Testing Strategy](TESTING_STRATEGY.md)** — Test coverage and quality

[View full contributing guide →](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)

> ⚠️ **Note:** WIP documents are temporary, may be incomplete, and are not part of official documentation.

---

## ⚖️ Legal & Fair Use

This project is intended for **personal, non-commercial use only**. All downloaded
content must remain local to your device and must not be shared, uploaded, or
redistributed.

**You are responsible for ensuring compliance with:**

- Copyright law
- RSS feed terms of service
- Podcast platform policies

This software is provided for educational and personal-use purposes only. It is not
intended to power a public dataset, index, or any commercial service without explicit
permission from rights holders.

[Read full legal notice →](LEGAL.md)

---

## 📄 License

MIT License - See [LICENSE](https://github.com/chipi/podcast_scraper/blob/main/LICENSE)
for details.

**Important:** The MIT license applies only to the source code. It does not grant any
rights to redistribute third-party podcast content.

---

## 🔗 Quick Links

### Project Resources

- **Repository:** [github.com/chipi/podcast_scraper](https://github.com/chipi/podcast_scraper)
- **Documentation:** [chipi.github.io/podcast_scraper](https://chipi.github.io/podcast_scraper/)
- **Issues:** [Report bugs or request features](https://github.com/chipi/podcast_scraper/issues)
- **License:** [MIT License](https://github.com/chipi/podcast_scraper/blob/main/LICENSE)

### Essential Documentation

**Getting Started:**
[Quick Start](#quick-start) • [Configuration](api/CONFIGURATION.md) • [CLI Ref](api/CLI.md)

**Examples:**
[Config YAML](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.yaml) •
[Config JSON](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.json)

**Development:**
[Contributing](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md) •
[Architecture](ARCHITECTURE.md) • [Testing](guides/TESTING_GUIDE.md) • [CI/CD](CI_CD.md)

**Provider System:**
[Quick Reference](guides/PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) •
[Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md)

**Specifications:**
[PRDs](prd/index.md) • [RFCs](rfc/index.md) • [Releases](releases/RELEASE_v2.3.0.md)

---

**Need help?** Check the [documentation](https://chipi.github.io/podcast_scraper/),
search [existing issues](https://github.com/chipi/podcast_scraper/issues), or open a
new issue for support.
