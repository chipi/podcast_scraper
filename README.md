# Podcast Scraper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Personal Use Only](https://img.shields.io/badge/Use-Personal%20Only-orange)](docs/LEGAL.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://chipi.github.io/podcast_scraper/)

[![CI](https://github.com/chipi/podcast_scraper/actions/workflows/python-app.yml/badge.svg)](https://github.com/chipi/podcast_scraper/actions/workflows/python-app.yml)
[![Nightly](https://github.com/chipi/podcast_scraper/actions/workflows/nightly.yml/badge.svg)](https://github.com/chipi/podcast_scraper/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/chipi/podcast_scraper/branch/main/graph/badge.svg)](https://codecov.io/gh/chipi/podcast_scraper)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

[![Snyk Security](https://snyk.io/test/github/chipi/podcast_scraper/badge.svg)](https://snyk.io/test/github/chipi/podcast_scraper)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-025E8C?logo=dependabot)](https://github.com/chipi/podcast_scraper/network/updates)

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
pip install -e ".[ml]"
```

#### Development (main)

Use this if you are contributing or experimenting. This branch may contain
unreleased changes.

```bash
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"
```

**Alternative (for contributors):**

Replace `https://example.com/feed.xml` with your podcast's RSS feed URL.

```bash
make init  # Creates venv, installs package, sets up pre-commit hooks
```

### Configure (Optional but Recommended)

If using OpenAI providers or customizing paths/performance:

```bash

# Copy the environment template

cp examples/.env.example .env

# Edit .env and add your settings (especially OPENAI_API_KEY if using OpenAI providers)

nano .env  # or use your preferred editor
```

**Key settings in `.env`:**

- `OPENAI_API_KEY` - Required for OpenAI transcription/summarization
- `LOG_LEVEL` - Set to DEBUG for verbose output
- `OUTPUT_DIR` - Custom location for transcripts/metadata
- `CACHE_DIR` - Cache location for ML models
- Performance tuning (workers, parallelism, device)

See [`examples/.env.example`](examples/.env.example) for all available options.

## Run

Replace `https://example.com/feed.xml` with your podcast's RSS feed URL.

```bash

# Download existing transcripts

python3 -m podcast_scraper.cli https://example.com/feed.xml

# Generate transcripts with Whisper when missing

python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing

# Full processing: transcripts + metadata + summaries

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --generate-metadata \
  --generate-summaries
```yaml

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
