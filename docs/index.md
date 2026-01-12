# Podcast Scraper

**Download and process podcast transcripts with ease.**

Podcast Scraper is a Python tool that downloads transcripts for every episode in a podcast RSS feed. It understands Podcasting 2.0 transcript tags, resolves relative URLs, and can fall back to Whisper transcription when episodes lack published transcripts.

> **⚠️ Important:** This project is intended for **personal, non-commercial use only**. All downloaded content must remain local and not be shared or redistributed. See [Legal Notice & Appropriate Use](LEGAL.md) for details.

---

## ✨ Key Features

- **Transcript Downloads** — Automatic detection and download of podcast transcripts from RSS feeds.
- **Whisper Fallback** — Generate transcripts using OpenAI Whisper when none exist.
- **Speaker Detection** — Automatic speaker name detection using Named Entity Recognition (NER).
- **Screenplay Formatting** — Format Whisper transcripts as dialogue with speaker labels.
- **Episode Summarization** — Generate concise summaries using local transformer models (BART + LED).
- **Metadata Generation** — Create database-friendly JSON/YAML metadata documents per episode.
- **Multi-threaded Downloads** — Concurrent processing with configurable worker pools.
- **Resumable Operations** — Skip existing files, reuse media, and handle interruptions gracefully.
- **Configuration Files** — JSON/YAML config support for repeatable workflows.
- **Service Mode** — Non-interactive daemon mode for automation and process managers.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Install the package in editable mode
pip install -e .

# For Whisper transcription, ensure ffmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### Basic Usage

```bash
# Download transcripts from a podcast
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --max-episodes 10 \
  --output-dir ./my_transcripts

# Use Whisper when transcripts are missing (now default in v2.4.0)
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --whisper-model base.en

# Generate metadata and summaries
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```

### Configuration File

```yaml
# config.yaml
rss: https://example.com/feed.xml
output_dir: ./transcripts
max_episodes: 50
transcribe_missing: true
whisper_model: base.en
screenplay: true
num_speakers: 2
auto_speakers: true
generate_metadata: true
generate_summaries: true
workers: 4
skip_existing: true
```

**Example configs:**

- [config.example.json](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.json) — JSON format
- [config.example.yaml](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.yaml) — YAML format

---

## 📚 Documentation

### Getting Started

| Resource | Description |
| :--- | :--- |
| **[Quick Start](#quick-start)** | Installation, basic usage, and first commands |
| **[Configuration Guide](api/CONFIGURATION.md)** | Complete configuration options and examples |
| **[CLI Reference](api/CLI.md)** | Command-line interface documentation |
| **[Python API](api/REFERENCE.md)** | Public API for programmatic usage |
| **[Legal Notice](LEGAL.md)** | ⚠️ Important usage restrictions and fair use |

### User Guides

| Guide | Description |
| :--- | :--- |
| **[Service Mode](api/SERVICE.md)** | Running as daemon or service (systemd, supervisor, cron) |
| **[ML Provider Reference](guides/ML_PROVIDER_REFERENCE.md)** | ML implementation details, models, and tuning |
| **[Troubleshooting](guides/TROUBLESHOOTING.md)** | Common issues and solutions |
| **[Glossary](guides/GLOSSARY.md)** | Key terms and concepts |

### For Developers

| Resource | Description |
| :--- | :--- |
| **[Quick Reference](guides/QUICK_REFERENCE.md)** | ⭐ One-page cheat sheet for common commands |
| **[Architecture Overview](ARCHITECTURE.md)** | High-level system design and module responsibilities |
| **[Testing Strategy](TESTING_STRATEGY.md)** | Test coverage, quality assurance, and testing guidelines |
| **[Testing Guide](guides/TESTING_GUIDE.md)** | Detailed test execution, fixtures, and coverage information |
| **[Quality Evaluation Guide](guides/EVALUATION_GUIDE.md)** | **ROUGE scoring, golden datasets, and quality metrics** |
| **[CI/CD Overview](ci/index.md)** | CI/CD pipeline documentation |
| **[Engineering Process](guides/ENGINEERING_PROCESS.md)** | **The "Triad of Truth": PRDs, RFCs, and ADRs** |
| **[Development Guide](guides/DEVELOPMENT_GUIDE.md)** | Development environment setup and tooling |
| **[Dependencies Guide](guides/DEPENDENCIES_GUIDE.md)** | Third-party dependencies, rationale, and management |
| **[API Boundaries](api/BOUNDARIES.md)** | API design principles and stability guarantees |
| **[API Migration Guide](api/MIGRATION_GUIDE.md)** | Upgrading between versions |
| **[API Versioning](api/VERSIONING.md)** | Versioning strategy and compatibility |

### Provider System (v2.4.0+)

| Guide | Description |
| :--- | :--- |
| **[AI Provider Comparison Guide](guides/AI_PROVIDER_COMPARISON_GUIDE.md)** | Detailed comparison of all 8 supported AI providers |
| **[ML Model Comparison Guide](guides/ML_MODEL_COMPARISON_GUIDE.md)** | **Compare ML models: Whisper, spaCy, Transformers (BART/LED)** |
| **[Provider Configuration Quick Reference](guides/PROVIDER_CONFIGURATION_QUICK_REFERENCE.md)** | **Quick guide for configuring providers via CLI, config files, and programmatically** |
| **[Provider Implementation Guide](guides/PROVIDER_IMPLEMENTATION_GUIDE.md)** | Complete guide for implementing new providers |
| **[Protocol Extension Guide](guides/PROTOCOL_EXTENSION_GUIDE.md)** | Extending protocols and adding new methods to providers |

---

## ⚖️ Legal & Fair Use

This project is intended for **personal, non-commercial use only**. All downloaded content must remain local to your device and must not be shared, uploaded, or redistributed.

**You are responsible for ensuring compliance with:**

- Copyright law
- RSS feed terms of service
- Podcast platform policies

This software is provided for educational and personal-use purposes only. It is not intended to power a public dataset, index, or any commercial service without explicit permission from rights holders.

[Read full legal notice →](LEGAL.md)

---

## 📄 License

MIT License — See [LICENSE](https://github.com/chipi/podcast_scraper/blob/main/LICENSE) for details.

**Important:** The MIT license applies only to the source code. It does not grant any rights to redistribute third-party podcast content.

---

## 🔗 Quick Links

- **Repository:** [github.com/chipi/podcast_scraper](https://github.com/chipi/podcast_scraper)
- **Issues:** [Report bugs or request features](https://github.com/chipi/podcast_scraper/issues)
- **Documentation:** [chipi.github.io/podcast_scraper](https://chipi.github.io/podcast_scraper/)
