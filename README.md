# Podcast Scraper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Personal Use Only](https://img.shields.io/badge/Use-Personal%20Only-orange)](docs/LEGAL.md)
[![codecov](https://codecov.io/gh/chipi/podcast_scraper/branch/main/graph/badge.svg)](https://codecov.io/gh/chipi/podcast_scraper)
[![Snyk Security](https://snyk.io/test/github/chipi/podcast_scraper/badge.svg)](https://snyk.io/test/github/chipi/podcast_scraper)

Download, transcribe, and summarize podcast episodes. Fetches transcripts from RSS feeds
(Podcasting 2.0), generates them when missing, detects speakers, and creates summaries.
Use local models (Whisper, BART) or OpenAI API — your choice.

🎓 **Learning Project:** This is a personal project where I'm exploring AI-assisted coding
and hands-on work with edge and cloud AI/ML technologies.

> **⚠️ Personal use only.** Downloaded content must remain local and not be redistributed.
> See [Legal Notice](docs/LEGAL.md).

---

## Features

- **Transcript Downloads** — Automatic detection and download from RSS feeds
- **Transcription** — Generate transcripts with Whisper or OpenAI API
- **Audio Preprocessing** — Optimize audio files before transcription (reduce size, remove silence, normalize loudness)
- **Speaker Detection** — Identify speakers using spaCy NER
- **Summarization** — Episode summaries with BART/LED (local) or OpenAI
- **Metadata Generation** — JSON/YAML metadata per episode
- **Resumable** — Skip existing files, handle interruptions gracefully
- **Provider System** — Swap between local and cloud providers via config

---

## Quick Start

### Requirements

- Python 3.10+
- `ffmpeg` (for Whisper): `brew install ffmpeg` / `apt install ffmpeg`

### Install

#### Stable (recommended)

Use the latest released version for normal usage.

```bash
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
git checkout <latest-release-tag>   # e.g. v2.3.0
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"  # Install package in editable mode (required for CLI commands)
```

#### Development (main)

Use this if you are contributing or experimenting. This branch may contain
unreleased changes.

```bash
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"  # Install package in editable mode (required for CLI commands)
```

**Note:** The `pip install -e ".[ml]"` step is **required** before running CLI commands. Without it,
you'll get `ModuleNotFoundError: No module named 'podcast_scraper'` when trying to run
`python3 -m podcast_scraper.cli`. The `-e` flag installs the package in editable mode, so code
changes are immediately available.

### Configure Environment Variables (Optional but Recommended)

If you plan to use OpenAI providers (transcription, speaker detection, or summarization), or want to
customize logging, paths, or performance settings, set up a `.env` file:

```bash
# Copy the template
cp examples/.env.example .env

# Edit .env and add your OpenAI API key (required for OpenAI providers)
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important variables:**

- `OPENAI_API_KEY` - **Required** if using OpenAI providers (transcription, speaker detection, or summarization)
- `LOG_LEVEL` - Controls logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `OUTPUT_DIR` - Custom output directory (default: `./output/`)
- `CACHE_DIR` - ML model cache location
- Performance tuning variables (WORKERS, TIMEOUT, etc.)

See `examples/.env.example` for all available options and detailed documentation.

### Run

**Prerequisite:** Make sure you've completed the installation steps above and activated your virtual environment.

Replace `https://example.com/feed.xml` with your podcast's RSS feed URL.

```bash
# Download transcripts (automatically generates missing ones with Whisper)
python3 -m podcast_scraper.cli https://example.com/feed.xml

# Only download existing transcripts (skip transcription)
python3 -m podcast_scraper.cli https://example.com/feed.xml --no-transcribe-missing

# Full processing: transcripts + metadata + summaries
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```

Output is organized into `output/` with subdirectories: `transcripts/` for transcript files
and `metadata/` for JSON/YAML metadata. Use `--output-dir` to customize the location (default: `./output/`).

---

## Documentation

| Resource | Description |
| -------- | ----------- |
| [CLI Reference](docs/api/CLI.md) | All command-line options |
| [Configuration](docs/api/CONFIGURATION.md) | Config files and environment variables |
| [Guides](docs/guides/) | Development, testing, and usage guides |
| [Troubleshooting](docs/guides/TROUBLESHOOTING.md) | Common issues and solutions |
| [Full Documentation](https://chipi.github.io/podcast_scraper/) | Complete docs site |

**Contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License — see [LICENSE](LICENSE).

**Note:** The license applies to source code only, not to podcast content downloaded
using this software. See [Legal Notice](docs/LEGAL.md).
