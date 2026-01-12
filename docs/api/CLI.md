# CLI Interface

The command-line interface provides an interactive way to use podcast_scraper.

## Overview

The CLI is designed for:

- Interactive command-line use
- Quick one-off transcript downloads
- Testing and experimentation
- Integration with shell scripts

For non-interactive use (daemons, services), see the [Service API](SERVICE.md) instead.

## Quick Start

````bash

# Basic usage

python -m podcast_scraper.cli https://example.com/feed.xml

# With options

python -m podcast_scraper.cli https://example.com/feed.xml \
  --max-episodes 50 \
  --transcribe-missing \
  --workers 8

# With config file

python -m podcast_scraper.cli --config config.yaml
```text

    options:
      show_root_heading: true
      heading_level: 3
      members:

```yaml
        - main
        - version_option
      show_source: false

```text

    options:
      show_root_heading: true
      heading_level: 3

## Common Options

### Basic Options

- `RSS_URL` - RSS feed URL (required if not using `--config`)
- `--output-dir PATH` - Output directory
- `--max-episodes N` - Maximum episodes to process
- `--workers N` - Number of concurrent download workers

### Transcription Options

- `--transcription-provider TYPE` - Provider (`whisper` or `openai`, default: `whisper`)
- `--transcribe-missing` - Use transcription for episodes without transcripts (DEFAULT)
- `--no-transcribe-missing` - Disable automatic transcription
- `--whisper-model MODEL` - Whisper model to use (tiny.en, base.en, small.en, etc., default: `base.en`)
- `--whisper-device DEVICE` - Device for Whisper (cuda/mps/cpu/auto, default: `auto`)
- `--screenplay` - Format output as screenplay with speaker labels
- `--screenplay-gap SECONDS` - Minimum gap to trigger speaker change (default: 1.25)
- `--num-speakers N` - Number of speakers for diarization (default: 2)
- `--speaker-names NAMES` - Comma-separated speaker names (manual override)

### Speaker Detection & Summarization

- `--speaker-detector-provider TYPE` - Provider (`spacy` or `openai`, default: `spacy`)
- `--language CODE` - Language for transcription and NER (default: `en`)
- `--generate-metadata` - Generate per-episode metadata (required for summaries)
- `--metadata-format FORMAT` - Format (`json` or `yaml`, default: `json`)
- `--metadata-subdirectory DIR` - Subdirectory for metadata files
- `--generate-summaries` - Generate AI episode summaries
- `--summary-provider TYPE` - Provider (`transformers` or `openai`, default: `transformers`)
- `--summary-model MODEL` - Model name for summarization (MAP phase)
- `--summary-reduce-model MODEL` - Model name for summarization (REDUCE phase)
- `--summary-max-length N` - Maximum summary length in tokens
- `--summary-min-length N` - Minimum summary length in tokens
- `--summary-device DEVICE` - Device for summarization model (cuda/mps/cpu/auto)
- `--summary-chunk-size N` - Chunk size for long transcripts in tokens
- `--summary-prompt PROMPT` - Custom prompt/instruction for summarization
- `--no-save-cleaned-transcript` - Don't save cleaned transcript to separate file
- `--metrics-output PATH` - Path to save pipeline metrics JSON

### Cache Management

- `cache --status` - View disk usage of all ML model caches
- `cache --clean [TYPE]` - Clean ML model caches. TYPE can be `whisper`, `transformers`, `spacy`, or `all` (default).
- `cache --clean --yes` - Non-interactive clean (skips confirmation prompt)

### legacy Cache Management (Deprecated)

- `--cache-info` - Show Hugging Face model cache information and exit
- `--prune-cache` - Remove all cached transformer models
- `--cache-dir PATH` - Custom cache directory for transformer models

### Control Options

- `--dry-run` - Preview without writing files
- `--skip-existing` - Skip episodes with existing output (resumable)
- `--clean-output` - Remove output directory before processing
- `--reuse-media` - Reuse existing media files in `.tmp_media` (faster testing)
- `--run-id ID` - Custom identifier for this run
- `--log-level LEVEL` - Set logging level (DEBUG, INFO, etc.)
- `--config PATH` - Path to JSON/YAML configuration file

## Configuration Files

The CLI supports JSON and YAML configuration files:

```bash
python -m podcast_scraper.cli --config config.json
```text

- `0` - Success
- `1` - Error (invalid config, network failure, etc.)

## Examples

### Download Transcripts

```bash

# Download all available transcripts

python -m podcast_scraper.cli https://example.com/feed.xml

# Limit to 10 episodes

python -m podcast_scraper.cli https://example.com/feed.xml --max-episodes 10
```text

# Transcribe missing episodes

python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --whisper-model base

# With screenplay formatting

python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --screenplay \
  --num-speakers 2 \
  --speaker-names "Host,Guest"

```text

# Generate metadata and summaries

python -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries

# With custom summary model

python -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries \
  --summary-model bart-large
```text

- [Core API](CORE.md) - Programmatic usage
- [Service API](SERVICE.md) - Non-interactive daemon usage
- [Configuration](CONFIGURATION.md) - All configuration options
- README - Complete CLI documentation

````
