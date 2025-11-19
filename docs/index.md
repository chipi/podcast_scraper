# Podcast Scraper

**Download and process podcast transcripts with ease.**

Podcast Scraper is a Python tool that downloads transcripts for every episode in a podcast RSS feed. It understands Podcasting 2.0 transcript tags, resolves relative URLs, and can fall back to Whisper transcription when episodes lack published transcripts.

> **⚠️ Important:** This project is intended for **personal, non-commercial use only**. All downloaded content must remain local and not be shared or redistributed. See [Legal Notice & Appropriate Use](legal.md) for details.

---

## ✨ Key Features

- **Transcript Downloads** — Automatic detection and download of podcast transcripts from RSS feeds
- **Whisper Fallback** — Generate transcripts using OpenAI Whisper when none exist
- **Speaker Detection** — Automatic speaker name detection using Named Entity Recognition (NER)
- **Screenplay Formatting** — Format Whisper transcripts as dialogue with speaker labels
- **Episode Summarization** — Generate concise summaries using local transformer models (BART + LED)
- **Metadata Generation** — Create database-friendly JSON/YAML metadata documents per episode
- **Multi-threaded Downloads** — Concurrent processing with configurable worker pools
- **Resumable Operations** — Skip existing files, reuse media, and handle interruptions gracefully
- **Configuration Files** — JSON/YAML config support for repeatable workflows
- **Service Mode** — Non-interactive daemon mode for automation and process managers

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Install the package
pip install -e .

# For Whisper transcription, ensure ffmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### Basic Usage

```bash
# Download transcripts from a podcast RSS feed
python3 -m podcast_scraper.cli https://example.com/feed.xml

# Limit episodes and use custom output directory
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --max-episodes 10 \
  --output-dir ./my_transcripts

# Use Whisper when transcripts are missing
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --whisper-model base

# Generate metadata and summaries
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries
```

---

## ⚙️ Configuration

Use JSON or YAML configuration files for complex workflows:

**config.yaml**

```yaml
rss: https://example.com/feed.xml
output_dir: ./transcripts
max_episodes: 50
transcribe_missing: true
whisper_model: base
screenplay: true
num_speakers: 2
auto_speakers: true
generate_metadata: true
generate_summaries: true
workers: 4
skip_existing: true
```

```bash
python3 -m podcast_scraper.cli --config config.yaml
```

See [examples/config.example.yaml](https://github.com/chipi/podcast_scraper/blob/main/examples/config.example.yaml) for complete configuration options.

---

## 📦 Service Mode

Run as a daemon or service for automation:

```bash
# Service mode (non-interactive, config file only)
python3 -m podcast_scraper.service --config config.yaml
```

**Use cases:**
- systemd services
- supervisor managed processes
- cron jobs
- CI/CD pipelines

See [examples/systemd.service.example](https://github.com/chipi/podcast_scraper/blob/main/examples/systemd.service.example) and [examples/supervisor.conf.example](https://github.com/chipi/podcast_scraper/blob/main/examples/supervisor.conf.example) for configuration examples.

---

## 🐍 Python API

```python
import podcast_scraper

# Configure and run
cfg = podcast_scraper.Config(
    rss="https://example.com/feed.xml",
    output_dir="./out",
    transcribe_missing=True,
    generate_metadata=True
)

podcast_scraper.run_pipeline(cfg)
```

See [API Reference](api/API_REFERENCE.md) for complete API documentation.

---

## 📚 Documentation

### For Users

- **[Quick Start](#-quick-start)** — Installation and basic usage (above)
- **[Configuration Guide](api/configuration.md)** — Complete configuration options and examples
- **[CLI Reference](api/cli.md)** — Command-line interface documentation
- **[Service API](api/service.md)** — Service mode and daemon usage
- **[Legal Notice](legal.md)** — Important usage restrictions and fair use

### For Developers

- **[Architecture](ARCHITECTURE.md)** — High-level system design and module responsibilities
- **[API Reference](api/API_REFERENCE.md)** — Complete public API documentation
- **[API Boundaries](api/API_BOUNDARIES.md)** — API design principles and stability guarantees
- **[Testing Strategy](TESTING_STRATEGY.md)** — Test coverage and quality assurance
- **[Contributing](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)** — Contribution guidelines

### Technical Specifications

- **[Product Requirements (PRDs)](prd/index.md)** — Intent and functional expectations for major features
- **[Requests for Comment (RFCs)](rfc/index.md)** — Technical specifications for module implementations
- **[Release Notes](releases/RELEASE_v2.3.0.md)** — Version history and changelog

---

## 🛠️ Advanced Features

### Automatic Speaker Detection

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --screenplay \
  --num-speakers 2 \
  --auto-speakers
```

Speaker names are automatically extracted from episode metadata using Named Entity Recognition. Manual override available via `--speaker-names`.

### Episode Summarization

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --generate-metadata \
  --generate-summaries \
  --summary-model bart-large \
  --summary-device mps
```

Generates concise summaries using a hybrid map-reduce strategy with local transformer models. Supports GPU acceleration (CUDA/MPS).

### Language Support

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --language fr \
  --whisper-model base
```

Automatic language-aware Whisper model selection and NER model matching.

---

## 📖 Local Development

Run the documentation site locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Visit [http://localhost:8000](http://localhost:8000) to preview. The site updates automatically as you edit Markdown files.

---

## ⚖️ Legal & Fair Use

This project is intended for **personal, non-commercial use only**. All downloaded content must remain local to your device and must not be shared, uploaded, or redistributed.

**You are responsible for ensuring compliance with:**
- Copyright law
- RSS feed terms of service
- Podcast platform policies

See [Legal Notice & Appropriate Use](legal.md) for complete details.

---

## 📄 License

MIT License - See [LICENSE](https://github.com/chipi/podcast_scraper/blob/main/LICENSE) for details.

**Note:** The MIT license applies only to the source code. It does not grant any rights to redistribute third-party podcast content.

---

## 🔗 Links

- **Repository:** [github.com/chipi/podcast_scraper](https://github.com/chipi/podcast_scraper)
- **Issues:** [Report bugs or request features](https://github.com/chipi/podcast_scraper/issues)
- **Documentation:** [chipi.github.io/podcast_scraper](https://chipi.github.io/podcast_scraper/)
