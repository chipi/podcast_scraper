# CLI Interface

The command-line interface provides an interactive way to use `podcast_scraper`.

## Overview

The CLI is designed for:

- Interactive command-line use
- Quick one-off transcript downloads
- Testing and experimentation
- Integration with shell scripts

For non-interactive use (daemons, services), see the [Service API](SERVICE.md) instead.

## Quick Start

```bash
# Basic usage
python -m podcast_scraper.cli https://example.com/feed.xml

# With options
python -m podcast_scraper.cli https://example.com/feed.xml \
  --max-episodes 50 \
  --transcribe-missing \
  --workers 8

# With config file
python -m podcast_scraper.cli --config config.yaml
```

## API Reference

::: podcast_scraper.cli
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - main
        - version_option
      show_source: false

## Common Options

### Basic Options

- `RSS_URL` - RSS feed URL (required if not using `--config`)
- `--output-dir PATH` - Output directory
- `--max-episodes N` - Maximum episodes to process
- `--workers N` - Number of concurrent download workers

### Provider Selection (v2.4.0+)

- `--transcription-provider PROVIDER` - Provider for transcription (`whisper`, `openai`)
- `--speaker-detector-provider PROVIDER` - Provider for speaker detection (`spacy`, `openai`, `anthropic`, etc.)
- `--summary-provider PROVIDER` - Provider for summarization (`transformers`, `openai`, `anthropic`, etc.)

### Transcription Options

- `--transcribe-missing` - Use Whisper for episodes without transcripts (now default)
- `--no-transcribe-missing` - Disable automatic transcription
- `--whisper-model MODEL` - Whisper model to use (tiny, base, small, medium, large)
- `--whisper-device DEVICE` - Device for Whisper (cuda/mps/cpu/auto, default: auto-detect)
- `--screenplay` - Format Whisper output as screenplay
- `--num-speakers N` - Number of speakers (default: 2)
- `--speaker-names NAMES` - Comma-separated speaker names

### Metadata & Summarization

- `--generate-metadata` - Generate metadata documents
- `--metadata-format FORMAT` - Format (json or yaml)
- `--generate-summaries` - Generate episode summaries
- `--summary-model MODEL` - Summary model to use (MAP-phase)
- `--summary-reduce-model MODEL` - Summary reduce model to use (REDUCE-phase)

### Cache Management (v2.4.0+)

- `cache --status` - View cache status for all ML models
- `cache --clean [TYPE]` - Clean ML caches (types: `whisper`, `transformers`, `spacy`, `all`)

### Control Options

- `--dry-run` - Preview without writing files
- `--skip-existing` - Skip episodes with existing output
- `--clean-output` - Remove output directory before processing

## Configuration Files

The CLI supports JSON and YAML configuration files:

```bash
python -m podcast_scraper.cli --config config.json
```

## Exit Codes

- `0` - Success
- `1` - Error (invalid config, network failure, etc.)

## Examples

### Download Transcripts

```bash
# Download all available transcripts
python -m podcast_scraper.cli https://example.com/feed.xml

# Limit to 10 episodes
python -m podcast_scraper.cli https://example.com/feed.xml --max-episodes 10
```

### Advanced Usage

```bash
# Transcribe missing episodes with specific model
python -m podcast_scraper.cli https://example.com/feed.xml \
  --whisper-model small.en

# With screenplay formatting and metadata
python -m podcast_scraper.cli https://example.com/feed.xml \
  --screenplay \
  --num-speakers 2 \
  --speaker-names "Host,Guest" \
  --generate-metadata
```

### Multi-Provider Examples

```bash
# Use OpenAI for everything
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider openai \
  --speaker-detector-provider openai \
  --summary-provider openai

# Mix and match
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider whisper \
  --speaker-detector-provider anthropic \
  --summary-provider mistral
```

## See Also

- [Core API](CORE.md) - Programmatic usage
- [Service API](SERVICE.md) - Non-interactive daemon usage
- [Configuration](CONFIGURATION.md) - All configuration options
- [Home](../index.md) - Complete documentation
