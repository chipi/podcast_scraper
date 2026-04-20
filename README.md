# Podcast Scraper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Personal Use Only](https://img.shields.io/badge/Use-Personal%20Only-orange)](docs/LEGAL.md)
[![codecov](https://codecov.io/gh/chipi/podcast_scraper/branch/main/graph/badge.svg)](https://codecov.io/gh/chipi/podcast_scraper)
[![Snyk Security](https://snyk.io/test/github/chipi/podcast_scraper/badge.svg)](https://snyk.io/test/github/chipi/podcast_scraper)

Download, transcribe, and summarize podcast episodes. Fetches transcripts from RSS feeds
(Podcasting 2.0), generates them when missing, detects speakers, creates summaries,
and optionally extracts structured insights (GIL) and knowledge graphs (KG).
Use local models (Whisper, transformers) or cloud APIs (OpenAI, Gemini, Anthropic,
Mistral, DeepSeek, Grok, Ollama) — your choice.

**Learning project:** This is a personal project where I'm exploring AI-assisted coding
and hands-on work with edge and cloud AI/ML technologies.

> **Personal use only.** Downloaded content must remain local and not be redistributed.
> See [Legal Notice](docs/LEGAL.md).

---

## Features

- **Transcript Downloads** — Automatic detection and download from RSS feeds
- **Episode selection** — Order (`newest` / `oldest`), optional publish-date window (`--since` / `--until`), offset, and `max_episodes` for large back-catalogs ([CONFIGURATION.md](docs/api/CONFIGURATION.md#episode-selection-github-521), GitHub #521)
- **Multi-feed corpus** — One config or CLI invocation for multiple shows: `feeds` / `rss_urls` in YAML, **`--feeds-spec`** for structured `feeds.spec.yaml` / JSON, or repeatable `--rss` / legacy `--rss-file`; isolated output under `<output_dir>/feeds/<stable_id>/` per feed. With `vector_search` + FAISS, a **single parent index** is built under `<output_dir>/search` after all feeds finish; **`corpus_manifest.json`**, **`corpus_run_summary.json`**, and structured log lines record batch status ([RFC-063](docs/rfc/RFC-063-multi-feed-corpus-append-resume.md), [CONFIGURATION.md](docs/api/CONFIGURATION.md#rss-and-multi-feed-corpus-github-440)). Inspect offline: `python -m podcast_scraper.cli corpus-status --output-dir <corpus_parent>`.
- **Transcription** — Generate transcripts with Whisper, OpenAI API, or Google Gemini API
- **Audio Preprocessing** — Optimize audio files before transcription (reduce size, remove silence, normalize loudness)
- **Speaker Detection** — Identify speakers using spaCy NER, OpenAI, Google Gemini, Grok (real-time info), or other providers
- **Summarization** — Episode summaries via local transformers (BART/LED), hybrid MAP-REDUCE (`hybrid_ml`), or any of 7 LLM providers
- **GIL Extraction** — Structured insights and verbatim quotes with evidence grounding (`gi.json` per episode, [PRD-017](docs/prd/PRD-017-grounded-insight-layer.md))
- **KG Extraction** — Topic-entity knowledge graphs from transcripts (`kg.json` per episode, [RFC-055](docs/rfc/RFC-055-knowledge-graph-layer-core.md))
- **Metadata Generation** — JSON/YAML metadata per episode
- **Resumable** — Skip existing files, handle interruptions gracefully
- **Provider System** — Swap between local and cloud providers via config
- **MPS Exclusive Mode** — Serialize GPU work on Apple Silicon to prevent memory contention and improve reliability (enabled by default)
- **Reproducibility** — Deterministic runs with seed control, pinned model revisions, and comprehensive run manifests (Issue #379)
- **Operational Hardening** — Configurable HTTP retries (media vs RSS), application-level episode retries on transient errors, timeout enforcement, `--fail-fast` / `--max-failures`, structured JSON logging, and `failure_summary` in `run.json` when episodes fail ([CONFIGURATION.md — Download resilience](docs/api/CONFIGURATION.md#download-resilience), Issue #379)
- **Security** — Path validation, model allowlist validation, safetensors format preference, and `trust_remote_code=False` enforcement (Issue #379)
- **Diagnostics** — `doctor` command for environment validation and dependency checks (Issue #379)
- **Semantic corpus search** — Optional FAISS index (`vector_search` in config), `search` / `index` CLIs, and semantic `gi explore --topic` when an index exists ([guide](docs/guides/SEMANTIC_SEARCH_GUIDE.md), RFC-061)
- **Run Tracking** — Per-episode stage timings, run summaries, episode index files, and `metrics.json` fields for download retries (`http_urllib3_retry_events`, `episode_download_retries`) (Issue #379)

- **Live pipeline monitor** — Optional **`--monitor`**: separate process + **`rich`** dashboard (or **`.monitor.log`** if stderr is not a TTY) and **`.pipeline_status.json`**. Optional **`pip install -e ".[monitor]"`**: **`--memray`** for heap profiling; with monitor + TTY, **`f`** runs **py-spy** to **`debug/flamegraph_*.svg`** ([RFC-065](docs/rfc/RFC-065-live-pipeline-monitor.md), [guide](docs/guides/LIVE_PIPELINE_MONITOR.md), #512)
- **GI / KG Viewer (v2)** — Optional browser UI: graph visualization, dashboard, semantic search, explore, **Corpus Library** (feed/episode browser, [RFC-067](docs/rfc/RFC-067-corpus-library-api-viewer.md)), and **Corpus Digest** (time-windowed highlights, [RFC-068](docs/rfc/RFC-068-corpus-digest-api-viewer.md)) — all against a pipeline output folder ([RFC-062](docs/rfc/RFC-062-gi-kg-viewer-v2.md); see below)

## Repository layout (Python + web)

The **Python** package (CLI, pipeline, FastAPI) lives at the **repo root** (`pyproject.toml`,
`Makefile`, `tests/`). The **GI/KG viewer** is a **Node** app under **`web/gi-kg-viewer/`**
(`package.json`, Vite, Vitest, Playwright). They share one repository but use **separate**
install steps and **different** `.env.example` files (root vs `web/gi-kg-viewer/`) by design.

**Onboarding in one place:** [Polyglot repository guide](docs/guides/POLYGLOT_REPO_GUIDE.md).

---

## GI / KG Viewer (optional)

Browse **Grounded Insight** and **Knowledge Graph** artifacts, use **semantic search** and **explore** on your corpus, and view a **dashboard**—all against the same **`--output-dir`** you use for the pipeline.

### What you need

| Goal | Install | Notes |
| ---- | ------- | ----- |
| **API + built UI in one process** | `pip install -e ".[server]"` and, once, `cd web/gi-kg-viewer && npm install && npm run build` | `make init` does **not** include `[server]` by default; add the extra if you use `serve`. |
| **Graph only, no Python API** | Just open the UI (e.g. Vite dev) and use **Choose .gi.json / .kg.json files** | Works when `/api/health` fails; no list/search/index from the server. |
| **Semantic search / index stats** | `[server]` + `[ml]` (FAISS, embeddings) as for `podcast search` | Index lives under `<output_dir>/search/`. See [Semantic Search Guide](docs/guides/SEMANTIC_SEARCH_GUIDE.md). |

### Run the server (typical)

From the repository root, with your virtualenv active:

```bash
pip install -e ".[server]"
cd web/gi-kg-viewer && npm install && npm run build && cd ../..
python -m podcast_scraper.cli serve --output-dir /path/to/your/run
```

Then open **<http://127.0.0.1:8000>** (default port). In the sidebar, set **Corpus root folder** to that **same** directory, click **List files**, select `.gi.json` / `.kg.json`, and **Load selected into graph**.

**FastAPI / HTTP API:** Full **`/api/*`** reference, architecture, and tests — [Server Guide](docs/guides/SERVER_GUIDE.md). With the server running, interactive OpenAPI (Swagger) is at **`/docs`** and **`/openapi.json`**.

**Development (API + hot-reload UI):** `make serve SERVE_OUTPUT_DIR=/path/to/your/run` runs the FastAPI app and the Vite dev server together (UI proxied to the API). See [web/gi-kg-viewer/README.md](web/gi-kg-viewer/README.md) and the Server Guide above.

**Testing the viewer:** `make test-ui` (Vitest unit tests, ~1s) and `make test-ui-e2e` (Playwright browser E2E, Firefox).

---

## Quick Start

### Requirements

- **Python 3.10+** — Check with `python3 --version`
- **ffmpeg** (only needed for local Whisper transcription):
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` or `yum install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - **Note:** Not required if using OpenAI providers only
- **Graphviz** (only needed to regenerate architecture diagrams locally):
  - macOS: `brew install graphviz`
  - Linux: `apt install graphviz` or `yum install graphviz`
  - Windows: Download from [graphviz.org](https://graphviz.org/download/)
  - **Note:** Required for `make visualize`. Diagrams must be committed; `make ci` / `make ci-fast` and CI run `visualize` and fail if they are stale.

### Installation Options

Choose the installation method based on your use case:

| Use Case | Installation Command | What You Get | Disk Space |
| -------- | --------------------- | ----------- | --------- |
| **OpenAI only** | `pip install -e ".[llm]"` | Core + OpenAI, Gemini, Anthropic, Mistral SDKs | ~50MB |
| **Local ML only** | `pip install -e ".[ml]"` | Core + Whisper, spaCy, torch, transformers, FAISS, **llama-cpp-python** (GGUF), etc. | ~1-3GB |
| **Both** (recommended) | `pip install -e ".[ml,llm]"` | Local ML + all LLM API SDKs | ~1-3GB |
| **Run comparison UI** (eval runs) | `pip install -e ".[compare]"` | Streamlit compare tool (RFC-047; `make run-compare`) | moderate |
| **GI/KG viewer API** | `pip install -e ".[server]"` | FastAPI + uvicorn for `python -m podcast_scraper.cli serve` | small |
| **Monitor profiling extras** (optional) | `pip install -e ".[monitor]"` | `py-spy` + `memray` integrated with `--monitor` / `--memray` (RFC-065); **not** required for `--monitor` alone | small |

**Quick decision guide:**

- Using LLM APIs (OpenAI, Gemini, etc.)? → `pip install -e ".[llm]"`
- Want to run models locally (Whisper, spaCy, Transformers)? → `pip install -e ".[ml]"`
- Want both? → `pip install -e ".[ml,llm]"` (recommended)
- Contributing / developing? → `pip install -e ".[dev,ml,llm]"`

**Note:** LLM provider SDKs (`openai`, `google-genai`, `anthropic`, `mistralai`, `httpx`) are **not** in core — they require the `[llm]` extra. Core (`pip install -e .`) gives you the pipeline framework only.

### FAISS / `vector_search` and embedding cache

Corpus indexing (`vector_search` with the default FAISS backend) calls the embedder with **`allow_download=False`**, so sentence-transformers weights must **already** be on disk in the Hugging Face hub cache. Before offline or sandbox runs, run **`make preload-ml-models`** and **omit `SKIP_GIL=1`** so evidence embeddings (including the model named by **`vector_embedding_model`**) are pre-cached; GIL’s **`gi_embedding_model`** is checked separately when GIL uses transformers. If you set **`HF_HUB_CACHE`**, use the same value when preloading and when running the pipeline. See [Semantic Search Guide](docs/guides/SEMANTIC_SEARCH_GUIDE.md).

### Install

> **Tip:** For end users who just want to run the tool, consider using [pipx](#method-2-pipx-recommended-for-end-users) or [uv](#method-3-uv-fast-installation) for easier installation. See the [Installation Guide](docs/guides/INSTALLATION_GUIDE.md) for all methods.

#### Method 1: pip (Standard - Recommended for Development)

Use the latest released version for normal usage.

**For LLM API users (OpenAI, Gemini, etc. — no local ML):**

```bash
# Clone the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
git checkout <latest-release-tag>   # e.g. v2.6.0

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version (must be 3.10+)
python --version  # Should show Python 3.10.x or higher

# CRITICAL: Upgrade pip and setuptools before installing
pip install --upgrade pip setuptools wheel

# Install with LLM provider SDKs (OpenAI, Gemini, Anthropic, Mistral, etc.)
pip install -e ".[llm]"
```

**For local ML users (Whisper, spaCy, Transformers):**

```bash
# Same setup as above, then:
# Install package with ML dependencies (includes Whisper, spaCy, Transformers)
pip install -e ".[ml]"
```

**Note:** LLM provider SDKs require the `[llm]` extra. For both local ML and LLM APIs: `pip install -e ".[ml,llm]"`.

#### Method 2: pipx (Recommended for End Users)

For isolated installation without managing virtual environments:

**For LLM-only users:**

```bash
# Install pipx (if not already installed)
# macOS: brew install pipx
# Linux: pip install --user pipx && pipx ensurepath

# Clone and install with LLM SDKs
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
pipx install -e ".[llm]"

# Verify
python -m podcast_scraper.cli --help
```

**For local ML users:**

```bash
# Same as above, but install with ML dependencies:
pipx install -e ".[ml]"
```

See [Installation Guide](docs/guides/INSTALLATION_GUIDE.md#method-2-pipx-isolated-installation---recommended-for-end-users) for details.

#### Method 3: uv (Fast Installation)

For faster installation using `uv`:

**For LLM API users:**

```bash
# Install uv (if not already installed)
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# macOS: brew install uv

# Clone and install with LLM SDKs
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper
uv venv
source .venv/bin/activate
uv pip install -e ".[llm]"
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

# CRITICAL: Upgrade pip and setuptools before installing
# This is required for editable installs with pyproject.toml
pip install --upgrade pip setuptools wheel

# Install package with ML dependencies
pip install -e ".[ml]"
```

**Important Notes:**

- **Python 3.10+ is REQUIRED** — The project uses features that require Python 3.10 or higher. Always verify with `python --version` after activating the venv.
- **Installation is required** — You must run `pip install -e .` (or `pip install -e ".[ml]"` for ML) before running CLI commands. Without it, you'll get `ModuleNotFoundError: No module named 'podcast_scraper'`.
- **LLM API users** — If you're using LLM-based providers (OpenAI, Gemini, Anthropic, etc.), install with `pip install -e ".[llm]"`. LLM provider SDKs are **not** in core — the `[llm]` extra is required.
- **Local ML users** — If you want to use local Whisper, spaCy, or Transformers, install with `pip install -e ".[ml]"` to get ML dependencies.
- **Upgrade pip/setuptools first** — If you see `"editable mode currently requires a setuptools-based build"` error, run `pip install --upgrade pip setuptools wheel` and try again.
- **Always activate the venv** — Remember to activate your virtual environment (`source .venv/bin/activate`) before running any commands.

### Configure Environment Variables (Required for LLM Providers)

If you plan to use LLM-based providers (OpenAI, etc.) for transcription, speaker detection, or summarization, you **must** set up a `.env` file with your API key:

```bash
# Copy the template
cp config/examples/.env.example .env

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

See `config/examples/.env.example` for all available options and detailed documentation.

### Verify Installation

Before running, verify the installation worked:

```bash
# Make sure venv is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Test that the package is installed
python -c "import podcast_scraper; print('[ok] Installation successful')"

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
python -m podcast_scraper.cli --config config/examples/config.example.yaml
```

**Available example config:**

- `config/examples/config.example.yaml` — Example configuration with local ML providers (Whisper, spaCy, Transformers)

**To customize:** Copy the example config and edit the `rss` field with your podcast feed URL:

```bash
# Copy the example config
cp config/examples/config.example.yaml my-config.yaml

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

See the [Basic Usage with Example Config](#basic-usage-with-example-config-recommended-for-first-time-users) section above for the recommended approach. You can also use any config file from the `config/examples/` directory or create your own.

**Output:** Files are organized in `output/` with subdirectories:

- `transcripts/` — Transcript files (RSS may serve `.vtt`/`.srt`; the pipeline normalizes those to `.txt` plus `*.segments.json` for GI quote audio timing when cues parse cleanly; plain `.txt`/`.html` feeds have no segment timing unless you add a sidecar)
- `metadata/` — JSON/YAML metadata, plus `gi.json` (GIL) and `kg.json` (KG) when enabled
- `search/` — FAISS vector index (when `vector_search` is enabled)
- `run.json`, `index.json`, `metrics.json` — Run tracking artifacts

Multi-feed corpora add `feeds/<stable_id>/` per feed, plus `corpus_manifest.json` and `corpus_run_summary.json` at the corpus root.

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
| [Roadmap](docs/ROADMAP.md) | Project roadmap with prioritized PRDs and RFCs |
| [Architecture](docs/architecture/ARCHITECTURE.md) | System design and module responsibilities |
| [Testing Strategy](docs/architecture/TESTING_STRATEGY.md) | Testing approach and test pyramid |
| [CLI Reference](docs/api/CLI.md) | All command-line options |
| [Configuration](docs/api/CONFIGURATION.md) | Config files and environment variables |
| [Server Guide](docs/guides/SERVER_GUIDE.md) | FastAPI viewer API, endpoints, development |
| [Semantic Search Guide](docs/guides/SEMANTIC_SEARCH_GUIDE.md) | FAISS indexing, search CLI, configuration |
| [AI Provider Comparison](docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md) | Compare all providers: cost, quality, speed, privacy |
| [Provider Deep Dives](docs/guides/PROVIDER_DEEP_DIVES.md) | Per-provider reference cards, benchmarks, and magic quadrant |
| [Experiment Guide](docs/guides/EXPERIMENT_GUIDE.md) | Eval datasets, baselines, experiments (`data/eval/`) |
| [Performance Profiles](docs/guides/PERFORMANCE_PROFILE_GUIDE.md) | Per-release stage timing snapshots (`data/profiles/`) |
| [Guides](docs/guides/) | Development, testing, and usage guides |
| [Troubleshooting](docs/guides/TROUBLESHOOTING.md) | Common issues and solutions |
| [Full Documentation](https://chipi.github.io/podcast_scraper/) | Complete docs site |

**Contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License — see [LICENSE](LICENSE).

**Note:** The license applies to source code only, not to podcast content downloaded
using this software. See [Legal Notice](docs/LEGAL.md).
