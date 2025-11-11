# Podcast Scraper

Podcast Scraper downloads transcripts for every episode in a podcast RSS feed. It understands Podcasting 2.0 transcript tags, resolves relative URLs, resumes partially completed runs, and can fall back to Whisper transcription when an episode has no published transcript. Multi-threaded downloads, resumable/cleanable output directories, dry-run previews, progress bars, configurable run folders, screenplay formatting, and JSON/YAML configuration files make it easy to collect, compare, and archive podcast transcripts.

## Requirements

- Python 3.10+
- `requests`
- `tqdm`
- `defusedxml`
- `platformdirs`
- `pydantic`
- `PyYAML` (for YAML config support)
- Optional: `openai-whisper` and `ffmpeg` when using Whisper fallback transcription

## Project Layout

- `podcast_scraper/cli.py` — command-line entry point
- `podcast_scraper/workflow.py` — high-level orchestration (`run_pipeline`)
- `podcast_scraper/config.py` — configuration models and file loader
- `podcast_scraper/downloader.py` — resilient HTTP helpers (`fetch_url`, `http_get`, `http_download_to_file`)
- `podcast_scraper/filesystem.py` — filesystem utilities and output-directory helpers
- `podcast_scraper/rss_parser.py` — RSS parsing helpers that build `RssFeed`/`Episode`
- `podcast_scraper/episode_processor.py` — transcript downloads and Whisper fallbacks
- `podcast_scraper/whisper.py` — Whisper integration
- `podcast_scraper/progress.py` — pluggable progress reporting interface

## Usage

### CLI

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

# Screenplay formatting
python3 -m podcast_scraper.cli https://example.com/feed.xml --transcribe-missing --screenplay \
  --num-speakers 2 --speaker-names "Host,Guest" --screenplay-gap 1.5

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
speaker_names:
  - Host
  - Guest
workers: 6
skip_existing: true
dry_run: false
```

### Virtual Environment

```bash
bash setup_venv.sh
# activate if needed
source .venv/bin/activate

# run without activating
.venv/bin/python -m podcast_scraper.cli <rss_url> [options]
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
- `--speaker.names` (str): Comma-separated speaker names
- `--run-id` (str): Subfolder/run identifier (`auto` to timestamp)
- `--workers` (int): Concurrent download workers (default derives from CPU count)
- `--skip-existing`: Skip episodes with existing output
- `--clean-output`: Remove output directory/run folder before processing
- `--dry-run`: Log planned work without writing files

## Notes

- Transcript links are detected via Podcasting 2.0 `podcast:transcript` tags or `<transcript>` elements.
- Episodes without transcripts are skipped unless `--transcribe-missing` is enabled.
- Progress integrates with `tqdm` by default; packages embedding the library can override via `podcast_scraper.set_progress_factory`.
- Whisper transcription requires `openai-whisper` and `ffmpeg`.
- Downloads run in parallel (with configurable worker count); Whisper transcription remains sequential.
- Automatic retries handle transient HTTP failures (429/5xx, connect/read errors).
- Combine `--skip-existing` to resume long runs, `--clean-output` for fresh runs, and `--dry-run` to inspect planned work.
