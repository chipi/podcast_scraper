# Podcast Scraper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Personal Use Only](https://img.shields.io/badge/Use-Personal%20Only-orange)](docs/legal.md)

Podcast Scraper downloads transcripts for every episode in a podcast RSS feed. It understands Podcasting 2.0 transcript tags, resolves relative URLs, resumes partially completed runs, and can fall back to Whisper transcription when an episode has no published transcript. Features include automatic speaker name detection using Named Entity Recognition (NER), language-aware Whisper model selection, multi-threaded downloads, resumable/cleanable output directories, dry-run previews, progress bars, configurable run folders, screenplay formatting, per-episode metadata document generation (JSON/YAML), episode summarization with local transformer models, and JSON/YAML configuration files.

> **⚠️ Important:** This project is intended for **personal, non-commercial use only**. All downloaded content must remain local and not be shared or redistributed. See [Legal Notice & Appropriate Use](docs/legal.md) for details.

## Documentation

- Live site: [https://chipi.github.io/podcast_scraper/](https://chipi.github.io/podcast_scraper/) — Architecture overview, PRDs, RFCs, and API guides.

- Local preview:

  ```bash
  pip install -r docs/requirements.txt
  mkdocs serve
  ```

  Visit [http://localhost:8000](http://localhost:8000) and edit files under `docs/` to see live updates.

  > **Note:** The documentation site is built to `.build/site/` directory. Build artifacts are organized in `.build/` to keep the root directory clean.

## Requirements

- Python 3.10+
- `requests`
- `tqdm`
- `defusedxml`
- `platformdirs`
- `pydantic`
- `PyYAML` (for YAML config support)
- `spacy` (for automatic speaker name detection)
- `openai-whisper` (for Whisper transcription)
- `ffmpeg` (required by Whisper)
- `torch` (for episode summarization, optional)
- `transformers` (for episode summarization, optional)
- `protobuf` (required for PEGASUS models, optional)

## Installation

Install the package from the repository root:

```bash
pip install -e .
```

**Note:** spaCy language models are automatically downloaded when needed (similar to Whisper models). The default English model (`en_core_web_sm`) will be downloaded automatically on first use. You can also manually download models if needed:

```bash
python -m spacy download en_core_web_sm
```

For Whisper transcription, ensure `ffmpeg` is available on your system (e.g., `brew install ffmpeg` on macOS).

When using a virtual environment, activate it first (see below) and run the same commands.

## Project Layout

- `podcast_scraper/cli.py` — command-line entry point (interactive use)
- `podcast_scraper/service.py` — service API (daemon/non-interactive use)
- `podcast_scraper/workflow.py` — high-level orchestration (`run_pipeline`)
- `podcast_scraper/config.py` — configuration models and file loader
- `podcast_scraper/downloader.py` — resilient HTTP helpers (`fetch_url`, `http_get`, `http_download_to_file`)
- `podcast_scraper/filesystem.py` — filesystem utilities and output-directory helpers
- `podcast_scraper/rss_parser.py` — RSS parsing helpers that build `RssFeed`/`Episode`
- `podcast_scraper/episode_processor.py` — transcript downloads and Whisper fallbacks
- `podcast_scraper/whisper_integration.py` — Whisper integration
- `podcast_scraper/speaker_detection.py` — automatic speaker name detection using NER
- `podcast_scraper/metadata.py` — metadata document generation
- `podcast_scraper/summarizer.py` — episode summarization using local transformer models
- `podcast_scraper/progress.py` — pluggable progress reporting interface

## Usage

> **Note:** All downloaded content must remain local to your device. Do not share, upload, or redistribute podcast transcripts, audio files, or other copyrighted material obtained using this tool.

### CLI

> **Tip:** When running from a local clone, execute the command from the
> repository root or set `PYTHONPATH` accordingly.

```bash
python3 -m podcast_scraper.cli <rss_url> [options]
```

Examples:

```bash
# Process all episodes and save transcripts to an auto-named folder
python3 -m podcast_scraper.cli https://example.com/feed.xml

# Limit episodes and add delay between requests
python3 -m podcast_scraper.cli https://example.com/feed.xml --max-episodes 50 --delay-ms 200

# Use multiple download workers
python3 -m podcast_scraper.cli https://example.com/feed.xml --workers 8

# Prefer specific transcript formats
python3 -m podcast_scraper.cli https://example.com/feed.xml --prefer-type text/plain --prefer-type .vtt

# Preview without writing files
python3 -m podcast_scraper.cli https://example.com/feed.xml --dry-run

# Custom output directory
python3 -m podcast_scraper.cli https://example.com/feed.xml --output-dir ./my_transcripts

# Whisper fallback when no transcript exists
python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing --whisper-model base

# Screenplay formatting with automatic speaker detection
python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing --screenplay \
  --num-speakers 2 --screenplay-gap 1.5

# Screenplay formatting with manual speaker names (overrides auto-detection)
python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing --screenplay \
  --num-speakers 2 --speaker-names "Host,Guest" --screenplay-gap 1.5

# Language-aware transcription (non-English)
python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing \
  --language fr --whisper-model base

# Separate runs
python3 -m podcast_scraper.cli https://example.com/feed.xml --run-id auto
python3 -m podcast_scraper.cli https://example.com/feed.xml --run-id vtt_vs_plain

# Resume skip-existing
python3 -m podcast_scraper.cli https://example.com/feed.xml --skip-existing

# Reuse existing media files (skip re-downloading for faster testing)
python3 -m podcast_scraper.cli https://example.com/feed.xml --reuse-media --transcribe-missing

# Test summarization with existing transcripts (skip download/transcription, generate summaries)
python3 -m podcast_scraper.cli https://example.com/feed.xml --skip-existing --generate-metadata --generate-summaries

# Generate metadata documents alongside transcripts
python3 -m podcast_scraper.cli https://example.com/feed.xml --generate-metadata

# Generate metadata in YAML format
python3 -m podcast_scraper.cli https://example.com/feed.xml --generate-metadata --metadata-format yaml

# Store metadata in separate subdirectory
python3 -m podcast_scraper.cli https://example.com/feed.xml --generate-metadata --metadata-subdirectory metadata

# Generate summaries for episodes (uses hybrid BART→LED by default)
python3 -m podcast_scraper.cli https://example.com/feed.xml --generate-metadata --generate-summaries

# Generate summaries with custom model and settings
python3 -m podcast_scraper.cli https://example.com/feed.xml --generate-metadata --generate-summaries \
  --summary-model bart-large --summary-reduce-model long-fast --summary-max-length 200

# Generate summaries with GPU acceleration (CUDA/MPS)
python3 -m podcast_scraper.cli https://example.com/feed.xml --generate-metadata --generate-summaries \
  --summary-device mps  # Use MPS for Apple Silicon, or 'cuda' for NVIDIA GPUs

# Check transformer model cache size and location
python3 -m podcast_scraper.cli --cache-info

# Prune transformer model cache to free disk space
python3 -m podcast_scraper.cli --prune-cache

# Run as a service/daemon (config file only, no user interaction)
python3 -m podcast_scraper.service --config config.yaml
```

### Service Mode

The service API (`podcast_scraper.service`) is optimized for non-interactive use, such as running as a daemon or service:

- **Config file only**: Works exclusively with configuration files (no CLI arguments)
- **Structured results**: Returns `ServiceResult` with success status and error messages
- **Exit codes**: Returns 0 for success, 1 for failure (suitable for process managers)
- **No user interaction**: Designed for automation and process management tools

**Use cases:**

- Running as a systemd service
- Managed by supervisor
- Scheduled execution (cron + service mode)
- CI/CD pipelines
- Automated workflows

**Example service usage:**

```python
from podcast_scraper import service

# Run from config file
result = service.run_from_config_file("config.yaml")
if result.success:
    print(f"Processed {result.episodes_processed} episodes")
    print(f"Summary: {result.summary}")
else:
    print(f"Error: {result.error}")
    sys.exit(1)
```

See `examples/supervisor.conf.example` and `examples/systemd.service.example` for process manager configuration examples.

### Configuration Files

Example configuration files are available in the `examples/` directory:

- `examples/config.example.json` - JSON format example
- `examples/config.example.yaml` - YAML format example

```bash
python3 -m podcast_scraper.cli --config config.json
```

`config.json`

```json
{
  "rss": "https://example.com/feed.xml",
  "timeout": 45,
  "transcribe_missing": true,
  "prefer_type": ["text/vtt", ".srt"],
  "run_id": "experiment",
  "workers": 4,
  "skip_existing": true,
  "reuse_media": false,
  "dry_run": false,
  "generate_metadata": true,
  "metadata_format": "json",
  "generate_summaries": true,
  "summary_provider": "local",
  "summary_model": "facebook/bart-base",
  "summary_max_length": 150,
  "summary_max_takeaways": 10,
  "summary_cache_dir": null
}
```

`config.yaml`

```yaml
rss: https://example.com/feed.xml
timeout: 30
transcribe_missing: true
prefer_type:
  - text/vtt
language: en
auto_speakers: true
speaker_names:  # Optional: manual override (takes precedence over auto-detection)
  - Host
  - Guest
workers: 6
skip_existing: true
reuse_media: false  # Reuse existing media files instead of re-downloading (for faster testing)
dry_run: false
generate_metadata: true  # Generate metadata documents alongside transcripts
metadata_format: json  # json or yaml
metadata_subdirectory: null  # Optional: store metadata in subdirectory
generate_summaries: true  # Generate summaries (requires torch/transformers)
summary_provider: local  # local, openai, or anthropic (only 'local' currently implemented)
summary_model: null  # Optional: MAP model (defaults to bart-large for chunk summaries)
summary_reduce_model: null  # Optional: REDUCE model (defaults to long-fast/LED for final combine)
summary_max_length: 160  # Maximum summary length in tokens (default: 160)
summary_min_length: 60  # Minimum summary length in tokens (default: 60)
summary_chunk_size: null  # Optional: token chunk size (defaults to 2048)
summary_device: null  # Optional: cuda, mps, cpu, or null for auto-detection
summary_cache_dir: null  # Optional: custom cache directory for transformer models (default: ~/.cache/huggingface/hub)
save_cleaned_transcript: true  # Save cleaned transcript to .cleaned.txt file (default: true)
```

### Virtual Environment

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate

# install project into the virtual environment
pip install -e .

# spaCy models are downloaded automatically when needed
# (or manually: python -m spacy download en_core_web_sm)

python -m podcast_scraper.cli <rss_url> [options]
```

### Docker

The Docker image uses the service API, which requires a configuration file:

```bash
docker build -t podcast-scraper -f docker/Dockerfile .

docker run --rm \
  -v "$(pwd)/output_docker:/app/output" \
  -v "$(pwd)/config.yaml:/app/config.yaml:ro" \
  podcast-scraper \
  --config /app/config.yaml
```

**Note:** The Docker image preloads the `base.en` Whisper model by default for optimal English transcription performance. To preload different models during build, use the `WHISPER_PRELOAD_MODELS` build argument:

```bash
# Preload multiple models
docker build --build-arg WHISPER_PRELOAD_MODELS="base.en,small.en" -t podcast-scraper .

# Preload multilingual model
docker build --build-arg WHISPER_PRELOAD_MODELS="base" -t podcast-scraper .
```

The service API is optimized for non-interactive use and provides structured exit codes:

- `0`: Success
- `1`: Error (configuration or runtime)

**Note:** The service API only accepts configuration files (no CLI arguments). Ensure your `config.yaml` includes all necessary settings like `rss_url` and `output_dir`.

#### Testing Docker Builds

Test Docker builds locally using Make targets:

```bash
# Build Docker image
make docker-build

# Build and run smoke tests
make docker-test

# Clean up Docker test images
make docker-clean
```

Docker builds are automatically validated in CI on every push to `main` and on pull requests that affect Docker-related files.

## Python API

The top-level package exposes a minimal stable API:

```python
import podcast_scraper

cfg = podcast_scraper.Config(rss="https://example.com/feed.xml", output_dir="./out")
podcast_scraper.run_pipeline(cfg)
```

Utilities:

- `podcast_scraper.load_config_file(path)` — parse JSON/YAML configuration
- `podcast_scraper.run_pipeline(cfg)` — run the full download/transcription pipeline
- `podcast_scraper.cli.main(argv)` — CLI entry point (also accessible via `python -m podcast_scraper.cli`)

Advanced helpers remain accessible in submodules (`podcast_scraper.downloader.fetch_url`, `podcast_scraper.filesystem.setup_output_directory`, etc.) but the public API documented above is stable.

## Options Summary

- `--output-dir` (path): Output directory (default: `output_rss_<host>_<hash>`)
- `--max-episodes` (int): Maximum episodes to process
- `--prefer-type` (repeatable): Preferred transcript MIME types/extensions
- `--user-agent` (str): User-Agent header
- `--timeout` (int): Request timeout in seconds (default: 20)
- `--delay-ms` (int): Delay between requests in milliseconds
- `--log-file` (path): Path to log file (logs written to both console and file)
- `--log-level` (str): Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `INFO`)
- `--transcribe-missing`: Use Whisper when no transcript is provided
- `--whisper-model` (str): Whisper model (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `tiny.en`, `base.en`, `small.en`, `medium.en`, `large.en`)
- `--screenplay`: Format Whisper transcript as screenplay
- `--screenplay-gap` (float): Gap (seconds) to trigger speaker change
- `--num-speakers` (int): Number of speakers to alternate between
- `--speaker-names` (str): Comma-separated speaker names (manual override, takes precedence over auto-detection)
- `--language` (str): Language for transcription and NER (default: `en`)
- `--ner-model` (str): spaCy NER model to use (default: derived from language, e.g., `en_core_web_sm` for English)
- `--auto-speakers`: Enable automatic speaker name detection (default: `true`)
- `--no-auto-speakers`: Disable automatic speaker name detection
- `--cache-detected-hosts`: Cache detected hosts across episodes (default: `true`)
- `--no-cache-detected-hosts`: Disable caching of detected hosts
- `--generate-metadata`: Generate metadata documents alongside transcripts
- `--metadata-format` (str): Format for metadata files (`json` or `yaml`, default: `json`)
- `--metadata-subdirectory` (str): Store metadata files in subdirectory (default: same as transcripts)
- `--generate-summaries`: Generate summaries for episodes (requires `torch` and `transformers`)
- `--summary-provider` (str): Summary provider (`local`, `openai`, `anthropic`, default: `local`)
- `--summary-model` (str): MAP model for chunk summarization (default: `bart-large`, fast and efficient)
- `--summary-reduce-model` (str): REDUCE model for final combine (default: `long-fast`/LED, accurate and long-context)
- `--summary-max-length` (int): Maximum summary length in tokens (default: 160)
- `--summary-min-length` (int): Minimum summary length in tokens (default: 60)
- `--summary-device` (str): Device for summarization (`cuda`, `mps`, `cpu`, `auto`, default: auto-detect)
- `--summary-chunk-size` (int): Chunk size for long transcripts in tokens (default: 2048)
- `--save-cleaned-transcript`: Save cleaned transcript to .cleaned.txt file (default: enabled)
- `--no-save-cleaned-transcript`: Don't save cleaned transcript
- `--summary-cache-dir` (str): Custom cache directory for transformer models (default: `~/.cache/huggingface/hub`)
- `--cache-info`: Show Hugging Face model cache information and exit
- `--prune-cache`: Remove all cached transformer models to free disk space
- `--cache-dir` (str): Custom cache directory for transformer models (alias for `--summary-cache-dir`)
- `--run-id` (str): Subfolder/run identifier (`auto` to timestamp)
- `--workers` (int): Concurrent download workers (default derives from CPU count)
- `--skip-existing`: Skip episodes with existing output (transcripts/metadata). When combined with `--generate-summaries`, still generates summaries from existing transcripts.
- `--reuse-media`: Reuse existing media files instead of re-downloading (for faster testing). Media files are kept after transcription instead of being deleted.
- `--clean-output`: Remove output directory/run folder before processing
- `--dry-run`: Log planned work without writing files

## Notes

- Transcript links are detected via Podcasting 2.0 `podcast:transcript` tags or `<transcript>` elements.
- Episodes without transcripts are skipped unless `--transcribe-missing` is enabled.
- **Automatic speaker detection**: When enabled (`--auto-speakers`, default), speaker names are automatically extracted from episode metadata using Named Entity Recognition (NER). Manual `--speaker-names` always takes precedence.
- **Language-aware processing**: The `--language` flag controls both Whisper model selection (preferring `.en` variants for English) and NER model selection.
- **spaCy models**: Language models are automatically downloaded when needed (similar to Whisper). The default model (`en_core_web_sm` for English) will be downloaded on first use. You can also manually download models if needed.
- **Metadata generation**: When enabled (`--generate-metadata`), generates comprehensive metadata documents (JSON/YAML) for each episode containing feed-level and episode-level information, detected speaker names, transcript sources, and processing metadata. Metadata files are database-friendly (snake_case fields, ISO 8601 dates) and can be directly loaded into PostgreSQL, MongoDB, Elasticsearch, or ClickHouse.
- **Episode summarization**: When enabled (`--generate-summaries`), generates concise summaries from episode transcripts using a hybrid map-reduce strategy with local transformer models. **Default configuration uses BART-large for MAP** (fast, efficient chunk summarization) and **LED (long-fast) for REDUCE** (accurate, long-context final combine). This hybrid approach is widely used in production systems. The map phase chunks long transcripts and summarizes each chunk. The reduce phase intelligently combines summaries: single-pass abstractive (≤800 tokens), mini map-reduce (800-4000 tokens, fully abstractive with re-chunking), or extractive selection (>4000 tokens). The mini map-reduce approach re-chunks combined summaries into 3-5 sections (650 words each), summarizes each section, then performs a final abstractive reduce. Summaries are stored in metadata documents and include model information, word count, and generation timestamp. Requires `torch` and `transformers` dependencies. Supports GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon). Both MAP and REDUCE models are configurable via `--summary-model` and `--summary-reduce-model`. Transformer models are automatically cached locally (default: `~/.cache/huggingface/hub/`) and reused across runs. Use `--cache-info` to check cache size and `--prune-cache` to free disk space.
- Progress integrates with `tqdm` by default; packages embedding the library can override via `podcast_scraper.set_progress_factory`.
- Whisper transcription requires `ffmpeg` to be installed on your system.
- Downloads run in parallel (with configurable worker count); Whisper transcription remains sequential.
- Automatic retries handle transient HTTP failures (429/5xx, connect/read errors).
- Combine `--skip-existing` to resume long runs, `--clean-output` for fresh runs, and `--dry-run` to inspect planned work.
- Use `--reuse-media` to skip re-downloading media files during testing (files are kept in `.tmp_media` directory).
- When testing summarization, combine `--skip-existing` with `--generate-summaries` to reuse existing transcripts and generate summaries without re-downloading or re-transcribing.

## Project Intent & Fair Use Notice

This project is intended solely for personal, non-commercial use.

It provides tools for downloading, organizing, and analyzing publicly available podcast metadata and audio only for private research, experimentation, and personal study.

This project does not host, republish, or redistribute any copyrighted podcast content, including:

- full transcripts

- full show notes

- audio files

- images or artwork

All copyrighted material obtained with the tools in this repository must remain local to the user and must not be shared, uploaded, or made publicly accessible.

Users are responsible for ensuring compliance with:

- copyright law,

- RSS feed terms,

- podcast platform policies.

This software is provided for educational and personal-use purposes only and is not intended to power a public dataset, index, or any commercial service without explicit permission from rights holders.

For complete details, see [Legal Notice & Appropriate Use](docs/legal.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important:** The MIT license applies only to the source code in this repository. It does not grant any rights to redistribute or publicly share any third-party podcast content retrieved or processed using this software. See [Legal Notice & Appropriate Use](docs/legal.md) for more information.
