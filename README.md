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
- **Transcription** — Generate transcripts with Whisper, OpenAI API, or Google Gemini API
- **Audio Preprocessing** — Optimize audio files before transcription (reduce size, remove silence, normalize loudness)
- **Speaker Detection** — Identify speakers using spaCy NER, OpenAI, Google Gemini, Grok (real-time info), or other providers
- **Summarization** — Episode summaries with BART/LED (local), OpenAI, Google Gemini, Grok (real-time info), or other providers
- **Metadata Generation** — JSON/YAML metadata per episode
- **Resumable** — Skip existing files, handle interruptions gracefully
- **Provider System** — Swap between local and cloud providers via config
- **MPS Exclusive Mode** — Serialize GPU work on Apple Silicon to prevent memory contention and improve reliability (enabled by default)
- **Reproducibility** — Deterministic runs with seed control, pinned model revisions, and comprehensive run manifests (Issue #379)
- **Operational Hardening** — Retry policies with exponential backoff, timeout enforcement, failure handling flags, and structured JSON logging (Issue #379)
- **Security** — Path validation, model allowlist validation, safetensors format preference, and `trust_remote_code=False` enforcement (Issue #379)
- **Diagnostics** — `doctor` command for environment validation and dependency checks (Issue #379)
- **Run Tracking** — Per-episode stage timings, run summaries, and episode index files for complete pipeline observability (Issue #379)

---

## Quick Start

### Requirements

- **Python 3.10+** — Check with `python3 --version`
- **ffmpeg** (only needed for local Whisper transcription):
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` or `yum install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - **Note:** Not required if using OpenAI providers only

### Installation Options

Choose the installation method based on your use case:

| Use Case | Installation Command | What You Get | Disk Space |
| -------- | --------------------- | ----------- | --------- |
| **LLM-only** (recommended) | `pip install -e .` | Core package + LLM SDKs (OpenAI, Gemini, etc.) | ~50MB |
| **Local ML only** | `pip install -e ".[ml]"` | Core + Whisper + spaCy + Transformers | ~1-3GB |
| **Both options** | `pip install -e ".[ml]"` | Everything (LLM + local ML) | ~1-3GB |

**Quick decision guide:**

- Using LLM APIs (OpenAI, etc.) for transcription/summarization? → Install core only (`pip install -e .`)
- Want to run models locally (Whisper, spaCy, Transformers)? → Install with ML (`pip install -e ".[ml]"`)
- Want both options? → Install with ML (`pip install -e ".[ml]"`)

**Note:** LLM provider SDKs (like `openai`, `google-generativeai`) are included in core dependencies, so LLM-based providers work without ML dependencies. For Gemini, install with `pip install -e ".[gemini]"` or `pip install -e ".[dev,ml,gemini]"` for development.

### Install

> **💡 Tip:** For end users who just want to run the tool, consider using [pipx](#method-2-pipx-recommended-for-end-users) or [uv](#method-3-uv-fast-installation) for easier installation. See the [Installation Guide](docs/guides/INSTALLATION_GUIDE.md) for all methods.

#### Method 1: pip (Standard - Recommended for Development)

Use the latest released version for normal usage.

**For LLM-only users (no ML dependencies needed):**

```bash
# Clone the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
git checkout <latest-release-tag>   # e.g. v2.5.0

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version (must be 3.10+)
python --version  # Should show Python 3.10.x or higher

# ⚠️ CRITICAL: Upgrade pip and setuptools before installing
# This is required for editable installs with pyproject.toml
pip install --upgrade pip setuptools wheel

# Install core package only (includes LLM SDKs like OpenAI, no ML dependencies)
# This is sufficient if you're using OpenAI providers for transcription, speaker detection, and summarization
pip install -e .

# For Gemini providers, install with gemini extras:
pip install -e ".[gemini]"
```

**For local ML users (Whisper, spaCy, Transformers):**

```bash
# Same setup as above, then:
# Install package with ML dependencies (includes Whisper, spaCy, Transformers)
pip install -e ".[ml]"
```

**Note:** LLM provider SDKs (like `openai`) are included in core dependencies, so LLM-based providers work without installing `[ml]`.

#### Method 2: pipx (Recommended for End Users)

For isolated installation without managing virtual environments:

**For LLM-only users:**

```bash
# Install pipx (if not already installed)
# macOS: brew install pipx
# Linux: pip install --user pipx && pipx ensurepath

# Clone and install (core package only, no ML dependencies)
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
pipx install -e .

# Verify
podcast_scraper --version
```

**For local ML users:**

```bash
# Same as above, but install with ML dependencies:
pipx install -e ".[ml]"
```

See [Installation Guide](docs/guides/INSTALLATION_GUIDE.md#method-2-pipx-isolated-installation---recommended-for-end-users) for details.

#### Method 3: uv (Fast Installation)

For faster installation using `uv`:

**For LLM-only users:**

```bash
# Install uv (if not already installed)
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# macOS: brew install uv

# Clone and install (core package only)
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
uv venv
source .venv/bin/activate
uv pip install -e .
```

**For local ML users:**

```bash
# Same as above, but install with ML dependencies:
uv pip install -e ".[ml]"
```

See [Installation Guide](docs/guides/INSTALLATION_GUIDE.md#method-3-uv-fast-installation---recommended-for-speed) for details.

#### Method 4: Docker (Containerized - Recommended for Production)

For containerized deployments and service-oriented usage:

**LLM-only variant (small, fast, ~200-300MB):**

```bash
# Build LLM-only image
docker build --build-arg INSTALL_EXTRAS="" -t podcast-scraper:llm-only .

# Run with config file
docker run -v ./config.yaml:/app/config.yaml \
           -v ./output:/app/output \
           -e OPENAI_API_KEY=sk-your-key \
           podcast-scraper:llm-only
```

**ML-enabled variant (full features, ~1-3GB):**

```bash
# Build ML-enabled image (default)
docker build --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .

# Run with config file
docker run -v ./config.yaml:/app/config.yaml \
           -v ./output:/app/output \
           podcast-scraper:ml
```

**Quick start with Docker Compose:**

```bash
# Use provided docker-compose.yml
docker-compose up -d
```

See [Docker Service Guide](docs/guides/DOCKER_SERVICE_GUIDE.md) and [Docker Variants Guide](docs/guides/DOCKER_VARIANTS_GUIDE.md) for complete Docker documentation.

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
- **Installation is required** — You must run `pip install -e .` (or `pip install -e ".[ml]"` for ML) before running CLI commands. Without it, you'll get `ModuleNotFoundError: No module named 'podcast_scraper'`.
- **LLM-only users** — If you're using LLM-based providers (OpenAI, Gemini, etc.) only, install with `pip install -e .` for OpenAI or `pip install -e ".[gemini]"` for Gemini (no `[ml]` needed). LLM provider SDKs are included in core dependencies.
- **Local ML users** — If you want to use local Whisper, spaCy, or Transformers, install with `pip install -e ".[ml]"` to get ML dependencies.
- **Upgrade pip/setuptools first** — If you see `"editable mode currently requires a setuptools-based build"` error, run `pip install --upgrade pip setuptools wheel` and try again.
- **Always activate the venv** — Remember to activate your virtual environment (`source .venv/bin/activate`) before running any commands.

### Configure Environment Variables (Required for LLM Providers)

If you plan to use LLM-based providers (OpenAI, etc.) for transcription, speaker detection, or summarization, you **must** set up a `.env` file with your API key:

```bash
# Copy the template
cp examples/.env.example .env

# Edit .env and add your LLM API key (REQUIRED for LLM providers)
# For OpenAI: OPENAI_API_KEY=sk-your-actual-api-key-here
# For Gemini: GEMINI_API_KEY=your-gemini-api-key-here
```

**Important variables:**

- `OPENAI_API_KEY` - **Required** if using OpenAI providers (transcription, speaker detection, or summarization)
- `GEMINI_API_KEY` - **Required** if using Gemini providers (transcription, speaker detection, or summarization)
- `LOG_LEVEL` - Controls logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `OUTPUT_DIR` - Custom output directory (default: `./output/`)
- `CACHE_DIR` - ML model cache location (only needed for local ML providers)
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

#### Basic Usage with Example Config (Recommended for First-Time Users)

The easiest way to get started is using an example config file:

```bash
# Make sure venv is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Set your API key (if using LLM providers)
export OPENAI_API_KEY=sk-your-actual-api-key-here

# Run with an example config file
python -m podcast_scraper.cli --config examples/config.example.yaml
```

**Available example config:**

- `examples/config.example.yaml` — Example configuration with local ML providers (Whisper, spaCy, Transformers)

**To customize:** Copy the example config and edit the `rss` field with your podcast feed URL:

```bash
# Copy the example config
cp examples/config.example.yaml my-config.yaml

# Edit my-config.yaml and change the RSS feed URL
# Then run:
python -m podcast_scraper.cli --config my-config.yaml
```

#### Advanced Usage (Command-Line Options)

Replace `https://example.com/feed.xml` with your podcast's RSS feed URL.

**For LLM-only users (no ML dependencies):**

```bash
# Make sure venv is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Set LLM API key (if not in .env file)
# For OpenAI:
export OPENAI_API_KEY=sk-your-actual-api-key-here

# For Gemini:
export GEMINI_API_KEY=your-gemini-api-key-here

# Download transcripts (automatically generates missing ones with LLM API)
# Using OpenAI:
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider openai

# Using Gemini:
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider gemini

# Only download existing transcripts (skip transcription)
python -m podcast_scraper.cli https://example.com/feed.xml \
  --no-transcribe-missing

# Full processing with OpenAI providers: transcripts + speaker detection + summaries + metadata
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider openai \
  --speaker-detector-provider openai \
  --summary-provider openai \
  --generate-metadata \
  --generate-summaries

# Full processing with Gemini providers:
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider gemini \
  --speaker-detector-provider gemini \
  --summary-provider gemini \
  --generate-metadata \
  --generate-summaries
```

**For local ML users (with ML dependencies installed):**

```bash
# Make sure venv is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Download transcripts (automatically generates missing ones with local Whisper)
python -m podcast_scraper.cli https://example.com/feed.xml

# Full processing with local ML: transcripts + speaker detection + summaries + metadata
python -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```

**Using a config file:**

See the [Basic Usage with Example Config](#basic-usage-with-example-config-recommended-for-first-time-users) section above for the recommended approach. You can also use any config file from the `examples/` directory or create your own.

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
