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

- `--transcribe-missing` - Use Whisper for episodes without transcripts
- `--whisper-model MODEL` - Whisper model to use (tiny, base, small, medium, large)
- `--whisper-device DEVICE` - Device for Whisper (cuda/mps/cpu/auto, default: auto-detect)
- `--screenplay` - Format Whisper output as screenplay
- `--num-speakers N` - Number of speakers (default: 2)
- `--speaker-names NAMES` - Comma-separated speaker names

### Metadata & Summarization

- `--generate-metadata` - Generate metadata documents
- `--metadata-format FORMAT` - Format (json or yaml)
- `--generate-summaries` - Generate episode summaries
- `--summary-model MODEL` - Summary model to use

### Control Options

- `--dry-run` - Preview without writing files
- `--skip-existing` - Skip episodes with existing output
- `--clean-output` - Remove output directory before processing

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
