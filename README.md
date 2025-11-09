# Podcast Scraper

Command-line tool that downloads transcripts for every episode in a podcast RSS feed. It understands Podcasting 2.0 transcript tags, resolves relative URLs, resumes partially completed runs, and can fall back to Whisper transcription when an episode has no published transcript. Multi-threaded downloads, resumable/cleanable output directories, dry-run previews, progress bars, configurable run folders, screenplay formatting, and JSON/YAML configuration files make it easy to collect, compare, and archive podcast transcripts.

## Requirements

- Python 3.10+
- `requests`
- `tqdm`
- `defusedxml`
- `platformdirs`
- `pydantic`
- `PyYAML` (for YAML config support)
- Optional: `openai-whisper` and `ffmpeg` when using Whisper fallback transcription

## File

- `podcast_scraper.py` — CLI entry point and implementation

## Usage

```bash
python3 podcast_scraper.py <rss_url> [options]
```

### Configuration file

Common options can be stored in a JSON or YAML file and loaded with `--config`.
Values from the command line still override the configuration file. When the
config includes an `rss` entry, the positional argument may be omitted.

```bash
python3 podcast_scraper.py --config config.json
```

Example `config.json` (see `config.example.json` in the repo):

```json
{
  "timeout": 45,
  "transcribe_missing": true,
  "prefer_type": ["text/vtt", ".srt"],
  "run_id": "experiment",
  "workers": 4,
  "skip_existing": true,
  "dry_run": false
}
```

Example `config.yaml` (see `config.example.yaml`; PyYAML is included in `requirements.txt`):

```yaml
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

### Virtual environment (recommended)

Create and use a project-local virtual environment with the one dependency:

```bash
bash setup_venv.sh
# activate if you prefer
source .venv/bin/activate

# run without activating
.venv/bin/python podcast_scraper.py <rss_url> [options]
```

### Docker

You can run the scraper in an isolated container. Build the image from the project root:

```bash
docker build -t podcast-scraper -f podcast_scraper/Dockerfile podcast_scraper
```

Run the container, mounting a host directory so transcripts persist outside the container:

```bash
mkdir -p output_docker
docker run --rm \
  -v "$(pwd)/output_docker:/app/output" \
  podcast-scraper \
  https://example.com/feed.xml --output-dir /app/output
```

Configuration files can be mounted in the same way:

```bash
docker run --rm \
  -v "$(pwd)/output_docker:/app/output" \
  -v "$(pwd)/podcast_scraper/config.lenny.yaml:/app/config.yaml:ro" \
  podcast-scraper \
  --config /app/config.yaml --output-dir /app/output
```

### Examples

```bash
# Basic: process all episodes and save transcripts to auto-named folder
python3 podcast_scraper.py https://example.com/feed.xml

# Limit number of episodes and add a small delay between requests
python3 podcast_scraper.py https://example.com/feed.xml --max-episodes 50 --delay-ms 200

# Use multiple download workers (parallelizes media fetching)
python3 podcast_scraper.py https://example.com/feed.xml --workers 8

# Prefer specific transcript formats (Podcasting 2.0: text/plain, WebVTT, SRT)
python3 podcast_scraper.py https://example.com/feed.xml --prefer-type text/plain --prefer-type .vtt

# Preview a run without saving files
python3 podcast_scraper.py https://example.com/feed.xml --dry-run

# Custom output directory
python3 podcast_scraper.py https://example.com/feed.xml --output-dir ./my_transcripts

# Transcribe with Whisper when episodes have no transcript
python3 podcast_scraper.py https://example.com/feed.xml --transcribe-missing --whisper-model base

# Screenplay-style formatting (alternate speakers by pause gaps)
python3 podcast_scraper.py https://example.com/feed.xml --transcribe-missing --screenplay \
  --num-speakers 2 --speaker-names "Host,Guest" --screenplay-gap 1.5

# Keep results from different runs separate (avoid overwrite)
python3 podcast_scraper.py https://example.com/feed.xml --run-id auto
python3 podcast_scraper.py https://example.com/feed.xml --run-id vtt_vs_plain

# When using Whisper, model name is automatically added to run folder
python3 podcast_scraper.py https://example.com/feed.xml --transcribe-missing --whisper-model base
# Creates: output_rss_.../run_whisper_base/
# Or with explicit run-id: output_rss_.../run_my_experiment_whisper_base/
# Filenames also include run identifier: "0001 - Episode Title_whisper_base.txt"

# Resume a run, skipping already-downloaded transcripts
python3 podcast_scraper.py https://example.com/feed.xml --skip-existing
```

## Options

- `--output-dir` (path): Output directory (default: `output_rss_<host>_<hash>`)
- `--max-episodes` (int): Maximum number of episodes to process
- `--prefer-type` (repeatable): Preferred transcript MIME types or extensions (e.g., `text/plain`, `.vtt`, `.srt`)
- `--user-agent` (str): User-Agent header
- `--timeout` (int): Request timeout in seconds (default: 20)
- `--delay-ms` (int): Delay between requests in milliseconds
- `--transcribe-missing`: Use Whisper to transcribe when no transcript is provided
- `--whisper-model` (str): Whisper model (one of `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `tiny.en`, `base.en`, `small.en`, `medium.en`, `large.en`)
- `--screenplay`: Format Whisper transcript as screenplay with speaker turns
- `--screenplay-gap` (float): Gap (seconds) to trigger speaker change (default: 1.25)
- `--num-speakers` (int): Number of speakers to alternate between (default: 2)
- `--speaker-names` (str): Comma-separated names to label speakers
- `--run-id` (str): Create a subfolder under output dir for this run; use `auto` to timestamp
- `--workers` (int): Number of concurrent download workers (default: 4, Whisper transcription remains sequential)
- `--skip-existing`: Skip episodes whose transcript output already exists (resume capability)
- `--clean-output`: Remove the target output directory/run folder before processing (fresh start)
- `--dry-run`: Print planned downloads/transcriptions without writing files

## Notes

- The scraper detects transcript links via Podcasting 2.0 `podcast:transcript` or `<transcript>` tags.
- If a feed does not expose transcript URLs, those episodes are skipped.
- Progress is printed to stderr; per-item progress bars stay visible after completion while detailed logs go to stdout.
- Whisper transcription is optional. If you want this feature within the venv:
  - `bash setup_venv.sh` (installs `openai-whisper` into `.venv`)
  - `brew install ffmpeg` (macOS) or install ffmpeg for your OS
- Downloads run in parallel using a worker pool, while Whisper transcription is processed sequentially because the reference implementation is not thread-safe.
- HTTP requests include automatic retries with exponential backoff for transient failures (5 attempts; applies to 429/5xx codes, connects, and reads).
- Combine `--skip-existing` to resume long runs and `--clean-output` to force a fresh transcription/output pass. Use `--dry-run` to inspect what would happen before launching a full run.
