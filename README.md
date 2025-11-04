# Podcast Scraper

Single-file CLI to download episode transcripts from a podcast RSS feed.

## Requirements

- Python 3.8+
- No external dependencies (uses Python standard library)
  - Optional: `openai-whisper` and `ffmpeg` for fallback transcription

## File

- `podcast_scraper.py` — CLI entry point and implementation

## Usage

```bash
python3 podcast_scraper.py <rss_url> [options]
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

### Examples

```bash
# Basic: process all episodes and save transcripts to auto-named folder
python3 podcast_scraper.py https://example.com/feed.xml

# Limit number of episodes and add a small delay between requests
python3 podcast_scraper.py https://example.com/feed.xml --max-episodes 50 --delay-ms 200

# Prefer specific transcript formats (Podcasting 2.0: text/plain, WebVTT, SRT)
python3 podcast_scraper.py https://example.com/feed.xml --prefer-type text/plain --prefer-type .vtt

# Custom output directory
python3 podcast_scraper.py https://example.com/feed.xml --output-dir ./my_transcripts

# Transcribe with Whisper when episodes have no transcript
python3 podcast_scraper.py https://example.com/feed.xml --transcribe-missing --whisper-model base
```

## Options

- `--output-dir` (path): Output directory (default: `output_rss_<host>_<hash>`)
- `--max-episodes` (int): Maximum number of episodes to process
- `--prefer-type` (repeatable): Preferred transcript MIME types or extensions (e.g., `text/plain`, `.vtt`, `.srt`)
- `--user-agent` (str): User-Agent header
- `--timeout` (int): Request timeout in seconds (default: 20)
- `--delay-ms` (int): Delay between requests in milliseconds
- `--transcribe-missing`: Use Whisper to transcribe when no transcript is provided
- `--whisper-model` (str): Whisper model (e.g., `tiny`, `base`, `small`, `medium`)

## Notes

- The scraper detects transcript links via Podcasting 2.0 `podcast:transcript` or `<transcript>` tags.
- If a feed does not expose transcript URLs, those episodes are skipped.
- Progress is printed to stderr; saved filenames are logged as they are written.
- Whisper transcription is optional. If you want this feature within the venv:
  - `bash setup_venv.sh` (installs `openai-whisper` into `.venv`)
  - `brew install ffmpeg` (macOS) or install ffmpeg for your OS

