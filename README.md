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

- **Python 3.10+** — Check with `python3 --version`
- **ffmpeg** (for Whisper transcription):
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` or `yum install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Install

#### Stable (recommended)

Use the latest released version for normal usage.

```bash
# Clone the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
git checkout <latest-release-tag>   # e.g. v2.3.0

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version (must be 3.10+)
python --version  # Should show Python 3.10.x or higher

# ⚠️ CRITICAL: Upgrade pip and setuptools before installing
# This is required for editable installs with pyproject.toml
pip install --upgrade pip setuptools wheel

# Install package with ML dependencies
pip install -e ".[ml]"
```

#### Development (main)

Use this if you are contributing or experimenting. This branch may contain
unreleased changes.

```bash
# Clone the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version (must be 3.10+)
python --version  # Should show Python 3.10.x or higher

# ⚠️ CRITICAL: Upgrade pip and setuptools before installing
# This is required for editable installs with pyproject.toml
pip install --upgrade pip setuptools wheel

# Install package with ML dependencies
pip install -e ".[ml]"
```

**Important Notes:**

- **Python 3.10+ is REQUIRED** — The project uses features that require Python 3.10 or higher. Always verify with `python --version` after activating the venv.
- **The `pip install -e ".[ml]"` step is required** before running CLI commands. Without it, you'll get `ModuleNotFoundError: No module named 'podcast_scraper'` when trying to run `python3 -m podcast_scraper.cli`.
- **Upgrade pip/setuptools first** — If you see `"editable mode currently requires a setuptools-based build"` error, run `pip install --upgrade pip setuptools wheel` and try again.
- **Always activate the venv** — Remember to activate your virtual environment (`source .venv/bin/activate`) before running any commands.

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

### Verify Installation

Before running, verify the installation worked:

```bash
# Make sure venv is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Test that the package is installed
python -c "import podcast_scraper; print('✓ Installation successful')"

# Check CLI is available
python -m podcast_scraper.cli --help
```

### Run

**Prerequisite:** Make sure you've completed the installation steps above and activated your virtual environment.

Replace `https://example.com/feed.xml` with your podcast's RSS feed URL.

```bash
# Make sure venv is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Download transcripts (automatically generates missing ones with Whisper)
python -m podcast_scraper.cli https://example.com/feed.xml

# Only download existing transcripts (skip transcription)
python -m podcast_scraper.cli https://example.com/feed.xml --no-transcribe-missing

# Full processing: transcripts + metadata + summaries
python -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```

**Output:** Files are organized in `output/` with subdirectories:

- `transcripts/` — Transcript files
- `metadata/` — JSON/YAML metadata files

Use `--output-dir` to customize the location (default: `./output/`).

### Troubleshooting Installation

**Problem:** `ModuleNotFoundError: No module named 'podcast_scraper'`

**Solution:** Make sure you:

1. Activated the virtual environment: `source .venv/bin/activate`
2. Installed the package: `pip install -e ".[ml]"`
3. Are using the venv's Python: `python -m podcast_scraper.cli` (not `python3` if system Python is different)

**Problem:** `"editable mode currently requires a setuptools-based build"`

**Solution:** Upgrade pip and setuptools first:

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[ml]"
```

**Problem:** Python version is < 3.10

**Solution:** Create venv with a newer Python:

```bash
# Find available Python versions
python3.11 --version  # or python3.12, etc.

# Create venv with specific version
python3.11 -m venv .venv
source .venv/bin/activate
```

For more help, see [Troubleshooting Guide](docs/guides/TROUBLESHOOTING.md).

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
