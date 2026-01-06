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

### Run

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
