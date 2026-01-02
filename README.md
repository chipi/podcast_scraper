# Podcast Scraper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Personal Use Only](https://img.shields.io/badge/Use-Personal%20Only-orange)](docs/LEGAL.md)

Podcast Scraper downloads transcripts for every episode in a podcast RSS feed. It understands
Podcasting 2.0 transcript tags, resolves relative URLs, and can fall back to Whisper transcription
when episodes lack published transcripts.

> **⚠️ Important:** This project is intended for **personal, non-commercial use only**. All downloaded
> content must remain local and not be shared or redistributed. See [Legal Notice & Appropriate
> Use](docs/LEGAL.md) for details.

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

## 🚀 Quick Start

### Requirements

- Python 3.10+
- `ffmpeg` (required for Whisper transcription)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

All Python dependencies are managed via `pyproject.toml`. See [Installation](#installation) below.

### Installation

````bash

# Clone the repository

git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Create virtual environment (recommended)

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with ML dependencies (speaker detection, transcription, summarization)

pip install -e ".[ml]"

# Or for development (includes dev tools + ML dependencies)

make init
```text

- `pip install -e .` — Core dependencies only
- `pip install -e ".[ml]"` — Core + ML dependencies (Whisper, spaCy, transformers)
- `pip install -e ".[dev,ml]"` — Core + ML + development tools

See `pyproject.toml` for complete dependency specifications.

## Basic Usage

```bash

# Download transcripts from a podcast RSS feed

python3 -m podcast_scraper.cli https://example.com/feed.xml

# Use Whisper when transcripts are missing

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --whisper-model base

# Generate metadata and summaries

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```text

- **Getting Started Guide** — Installation, configuration, and first steps
- **CLI Reference** — Complete command-line interface documentation
- **Python API** — Public API for programmatic usage
- **Configuration Guide** — JSON/YAML configuration options
- **Service Mode** — Running as daemon or service (systemd, supervisor)
- **Environment Variables** — Complete reference for all environment variables
- **Architecture** — Module boundaries and design principles
- **Development Notes** — Development setup, testing, and contribution guidelines

**Local preview:**

```bash
pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings mkdocstrings-python
mkdocs serve
```text

- Development setup and workflow
- Code style and testing requirements
- Architecture and design principles

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important:** The MIT license applies only to the source code in this repository. It does not grant any rights to redistribute or publicly share any third-party podcast content retrieved or processed using this software. See [Legal Notice & Appropriate Use](docs/LEGAL.md) for more information.

````
