# Release v1.0.0 - Initial Stable Release

**Release Date:** November 11, 2025
**Type:** Initial Release

## Summary

v1.0.0 marks the first stable release of Podcast Scraper, a command-line tool for downloading podcast transcripts from RSS feeds with Whisper fallback transcription support. This release provides a complete, production-ready implementation with multi-threaded downloads, configuration file support, progress tracking, and robust error handling.

## 🎉 Initial Features

### Core Functionality

- **RSS Feed Parsing**: Parse podcast RSS feeds and extract episode information
- **Podcasting 2.0 Support**: Understand and parse `podcast:transcript` tags for transcript discovery
- **Transcript Downloads**: Download transcripts from feed-provided URLs with automatic format detection
- **Whisper Fallback**: Automatically transcribe episodes without published transcripts using OpenAI Whisper
- **Screenplay Formatting**: Format transcripts with speaker labels and timestamps in screenplay style

### Performance & Reliability

- **Multi-threaded Downloads**: Concurrent episode processing with configurable worker count
- **Resilient HTTP Handling**: Automatic retries for transient failures (429/5xx errors, network issues)
- **Progress Tracking**: Real-time progress bars using `tqdm` for downloads and transcription
- **Resumable Processing**: Skip already-processed episodes with `--skip-existing` flag
- **Dry Run Mode**: Preview planned operations without writing files

### Configuration & Flexibility

- **Configuration Files**: Support for both JSON and YAML configuration files
- **Pydantic Validation**: Type-safe configuration with automatic validation
- **Flexible Transcript Preferences**: Specify preferred transcript formats/MIME types
- **Run Identification**: Organize output by run with `--run-id` (supports auto-timestamping)
- **Output Management**: Configurable output directories with automatic cleanup options

### Developer Experience

- **Single-File Implementation**: Complete functionality in `podcast_scraper.py` for easy deployment
- **Comprehensive Testing**: Integration tests covering core workflows
- **Type Hints**: Full type annotations throughout codebase
- **Logging**: Structured logging with configurable levels
- **Version Management**: Version flag (`--version`) for easy version tracking

### Deployment

- **Docker Support**: Pre-configured Dockerfile for containerized deployments
- **Virtual Environment**: Helper script for local development setup
- **Platform Support**: Cross-platform compatibility (Linux, macOS, Windows)

## Key Components

- **`podcast_scraper.py`**: Complete implementation including:
  - CLI argument parsing
  - RSS feed parsing with `defusedxml`
  - HTTP operations with `requests`
  - Whisper integration
  - Progress reporting
  - Configuration management with Pydantic
  - Filesystem utilities

## Usage Examples

````bash

# Basic usage

python podcast_scraper.py https://example.com/feed.xml

# Multi-threaded with Whisper fallback

python podcast_scraper.py https://example.com/feed.xml --workers 8 --transcribe-missing

# With configuration file

python podcast_scraper.py --config config.yaml

# Screenplay format with custom speakers

python podcast_scraper.py https://example.com/feed.xml --transcribe-missing --screenplay \
  --num-speakers 2 --speaker-names "Host,Guest"

# Resume previous run

python podcast_scraper.py https://example.com/feed.xml --skip-existing
```text

rss: https://example.com/feed.xml
timeout: 30
transcribe_missing: true
prefer_type:

  - text/vtt
  - .srt
speaker_names:


  - Host
  - Guest
workers: 6

skip_existing: true
dry_run: false

```text
- `tqdm` - Progress bar visualization
- `defusedxml` - Safe XML parsing
- `platformdirs` - Cross-platform directory paths
- `pydantic` - Configuration validation
- `PyYAML` - YAML configuration support
- `openai-whisper` - Transcription engine (optional)
- `ffmpeg` - Audio processing for Whisper (optional)

### Architecture

- Single-file monolithic architecture for simplicity
- Function-based design with clear separation of concerns
- Modular functions for parsing, downloading, and transcription
- Pluggable progress reporting
- Configurable retry logic with exponential backoff

### Quality & Testing

- Integration tests covering main workflows
- Type hints throughout for better IDE support
- Defensive programming with proper error handling
- Secure XML parsing with `defusedxml`
- Logging for debugging and monitoring

## Installation

```bash

# Clone repository

git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Install dependencies

pip install -r requirements.txt

# Optional: Whisper support

pip install openai-whisper
brew install ffmpeg  # macOS

```text

# Build image

docker build -t podcast-scraper -f podcast_scraper/Dockerfile .

# Run with volume mount

docker run --rm \
  -v "$(pwd)/output:/app/output" \
  podcast-scraper \
  https://example.com/feed.xml --output-dir /app/output
```text

- Single-file architecture limits modularity (addressed in v2.0.0)
- Whisper processing is sequential (not parallel) for thread safety
- No automatic speaker name detection (added in v2.0.0)
- Limited documentation (comprehensive docs added in v2.0.0)

## Looking Ahead

Version 1.0.0 provides a solid foundation for podcast transcript collection. Future versions will focus on:

- Modular architecture for better maintainability
- Automatic speaker detection using NER
- Comprehensive documentation with RFCs and PRDs
- Enhanced testing coverage
- API stability for programmatic use

## Contributors

- Marko Dragoljevic (@chipi)

## License

MIT License - see LICENSE file for details

---

**Note:** This is the baseline release that established the core functionality. Version 2.0.0 introduced a major refactoring to a modular architecture while maintaining backward compatibility with v1.0.0 configurations and usage patterns.
````
