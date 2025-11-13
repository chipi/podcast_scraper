# Podcast Scraper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Podcast Scraper downloads transcripts for every episode in a podcast RSS feed. It understands Podcasting 2.0 transcript tags, resolves relative URLs, resumes partially completed runs, and can fall back to Whisper transcription when an episode has no published transcript. Features include automatic speaker name detection using Named Entity Recognition (NER), language-aware Whisper model selection, multi-threaded downloads, resumable/cleanable output directories, dry-run previews, progress bars, configurable run folders, screenplay formatting, and JSON/YAML configuration files.

## Documentation

- Live site: [https://chipi.github.io/podcast_scraper/](https://chipi.github.io/podcast_scraper/) — Architecture overview, PRDs, RFCs, and API guides.

- Local preview:

  ```bash
  pip install -r docs/requirements.txt
  mkdocs serve
  ```

  Visit [http://localhost:8000](http://localhost:8000) and edit files under `docs/` to see live updates.

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

- `podcast_scraper/cli.py` — command-line entry point
- `podcast_scraper/workflow.py` — high-level orchestration (`run_pipeline`)
- `podcast_scraper/config.py` — configuration models and file loader
- `podcast_scraper/downloader.py` — resilient HTTP helpers (`fetch_url`, `http_get`, `http_download_to_file`)
- `podcast_scraper/filesystem.py` — filesystem utilities and output-directory helpers
- `podcast_scraper/rss_parser.py` — RSS parsing helpers that build `RssFeed`/`Episode`
- `podcast_scraper/episode_processor.py` — transcript downloads and Whisper fallbacks
- `podcast_scraper/whisper.py` — Whisper integration
- `podcast_scraper/speaker_detection.py` — automatic speaker name detection using NER
- `podcast_scraper/progress.py` — pluggable progress reporting interface

## Usage

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
```

### Configuration Files

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
  "dry_run": false
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
dry_run: false
```

### Virtual Environment

```bash
bash setup_venv.sh
source .venv/bin/activate

# install project into the virtual environment
pip install -e .

# download spaCy language model
python -m spacy download en_core_web_sm

python -m podcast_scraper.cli <rss_url> [options]
```

### Docker

```bash
docker build -t podcast-scraper -f podcast_scraper/Dockerfile podcast_scraper

docker run --rm \
  -v "$(pwd)/output_docker:/app/output" \
  podcast-scraper \
  https://example.com/feed.xml --output-dir /app/output
```

Mount configuration:

```bash
docker run --rm \
  -v "$(pwd)/output_docker:/app/output" \
  -v "$(pwd)/podcast_scraper/config.yaml:/app/config.yaml:ro" \
  podcast-scraper \
  --config /app/config.yaml --output-dir /app/output
```

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
- `--run-id` (str): Subfolder/run identifier (`auto` to timestamp)
- `--workers` (int): Concurrent download workers (default derives from CPU count)
- `--skip-existing`: Skip episodes with existing output
- `--clean-output`: Remove output directory/run folder before processing
- `--dry-run`: Log planned work without writing files

## Notes

- Transcript links are detected via Podcasting 2.0 `podcast:transcript` tags or `<transcript>` elements.
- Episodes without transcripts are skipped unless `--transcribe-missing` is enabled.
- **Automatic speaker detection**: When enabled (`--auto-speakers`, default), speaker names are automatically extracted from episode metadata using Named Entity Recognition (NER). Manual `--speaker-names` always takes precedence.
- **Language-aware processing**: The `--language` flag controls both Whisper model selection (preferring `.en` variants for English) and NER model selection.
       - **spaCy models**: Language models are automatically downloaded when needed (similar to Whisper). The default model (`en_core_web_sm` for English) will be downloaded on first use. You can also manually download models if needed.
- Progress integrates with `tqdm` by default; packages embedding the library can override via `podcast_scraper.set_progress_factory`.
- Whisper transcription requires `ffmpeg` to be installed on your system.
- Downloads run in parallel (with configurable worker count); Whisper transcription remains sequential.
- Automatic retries handle transient HTTP failures (429/5xx, connect/read errors).
- Combine `--skip-existing` to resume long runs, `--clean-output` for fresh runs, and `--dry-run` to inspect planned work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
