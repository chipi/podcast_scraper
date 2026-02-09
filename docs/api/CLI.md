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

- `--transcription-provider PROVIDER` - Provider for transcription (`whisper`, `openai`, `gemini`, `mistral`)
- `--speaker-detector-provider PROVIDER` - Provider for speaker detection (`spacy`, `openai`, `gemini`, `anthropic`, `mistral`, `deepseek`, `grok`, `ollama`)
- `--summary-provider PROVIDER` - Provider for summarization (`transformers`, `openai`, `gemini`, `anthropic`, `mistral`, `deepseek`, `grok`, `ollama`)

### Transcription Options

- `--transcribe-missing` - Use Whisper for episodes without transcripts (now default)
- `--no-transcribe-missing` - Disable automatic transcription
- `--whisper-model MODEL` - Whisper model to use (tiny, base, small, medium, large)
- `--whisper-device DEVICE` - Device for Whisper (cuda/mps/cpu/auto, default: auto-detect)
- `--mps-exclusive` - Serialize GPU work on MPS to prevent memory contention (default: enabled)
- `--no-mps-exclusive` - Allow concurrent GPU operations on MPS (for systems with sufficient GPU memory)
- `--screenplay` - Format Whisper output as screenplay
- `--num-speakers N` - Number of speakers (default: 2)
- `--speaker-names NAMES` - Comma-separated speaker names

### Audio Preprocessing Options (RFC-040)

- `--enable-preprocessing` - Enable audio preprocessing before transcription (experimental)
- `--preprocessing-cache-dir DIR` - Custom cache directory for preprocessed audio (default: `.cache/preprocessing`)
- `--preprocessing-sample-rate RATE` - Target sample rate in Hz (default: 16000, must be Opus-supported: 8000, 12000, 16000, 24000, 48000)
- `--preprocessing-silence-threshold THRESHOLD` - Silence detection threshold (default: `-50dB`)
- `--preprocessing-silence-duration DURATION` - Minimum silence duration to remove in seconds (default: `2.0`)
- `--preprocessing-target-loudness LOUDNESS` - Target loudness in LUFS for normalization (default: `-16`)

**Note**: Preprocessing requires `ffmpeg` to be installed. If `ffmpeg` is not available, preprocessing is automatically disabled with a warning.

### OpenAI Provider Options

- `--openai-api-key KEY` - OpenAI API key (can also use `OPENAI_API_KEY` env var)
- `--openai-api-base URL` - Custom OpenAI API base URL (for E2E testing or custom endpoints)
- `--openai-transcription-model MODEL` - OpenAI model for transcription (default: `whisper-1`)
- `--openai-speaker-model MODEL` - OpenAI model for speaker detection (default: `gpt-4o-mini`)
- `--openai-summary-model MODEL` - OpenAI model for summarization (default: `gpt-4o-mini`)
- `--openai-temperature TEMP` - Temperature for OpenAI generation (0.0-2.0, default: 0.3)
- `--openai-max-tokens N` - Maximum tokens for OpenAI responses (default: model-specific)
- `--openai-cleaning-model MODEL` - OpenAI model for transcript cleaning (default: `gpt-4o-mini`, cheaper than summary model)
- `--openai-cleaning-temperature TEMP` - Temperature for OpenAI cleaning (0.0-2.0, default: 0.2, lower = more deterministic)

### Gemini Provider Options

- `--gemini-api-key KEY` - Gemini API key (can also use `GEMINI_API_KEY` env var)
- `--gemini-api-base URL` - Custom Gemini API base URL (for E2E testing or custom endpoints)
- `--gemini-transcription-model MODEL` - Gemini model for transcription (default: environment-based, `gemini-1.5-flash` for test, `gemini-1.5-pro` for prod)
- `--gemini-speaker-model MODEL` - Gemini model for speaker detection (default: environment-based)
- `--gemini-summary-model MODEL` - Gemini model for summarization (default: environment-based)
- `--gemini-temperature TEMP` - Temperature for Gemini generation (0.0-2.0, default: 0.3)
- `--gemini-max-tokens N` - Max tokens for Gemini generation (default: model default)
- `--gemini-cleaning-model MODEL` - Gemini model for transcript cleaning (default: `gemini-1.5-flash`, cheaper than summary model)
- `--gemini-cleaning-temperature TEMP` - Temperature for Gemini cleaning (0.0-2.0, default: 0.2, lower = more deterministic)

### Anthropic Provider Options

- `--anthropic-api-key KEY` - Anthropic API key (can also use `ANTHROPIC_API_KEY` env var)
- `--anthropic-api-base URL` - Custom Anthropic API base URL (for E2E testing or custom endpoints)
- `--anthropic-speaker-model MODEL` - Anthropic model for speaker detection (default: environment-based)
- `--anthropic-summary-model MODEL` - Anthropic model for summarization (default: environment-based)
- `--anthropic-temperature TEMP` - Temperature for Anthropic generation (0.0-1.0, default: 0.3)
- `--anthropic-max-tokens N` - Max tokens for Anthropic generation (default: model default)
- `--anthropic-cleaning-model MODEL` - Anthropic model for transcript cleaning (default: `claude-3-5-haiku-latest`, cheaper than summary model)
- `--anthropic-cleaning-temperature TEMP` - Temperature for Anthropic cleaning (0.0-1.0, default: 0.2, lower = more deterministic)

### Mistral Provider Options

- `--mistral-api-key KEY` - Mistral API key (can also use `MISTRAL_API_KEY` env var)
- `--mistral-api-base URL` - Custom Mistral API base URL (for E2E testing or custom endpoints)
- `--mistral-transcription-model MODEL` - Mistral model for transcription (default: `voxtral-mini-latest`)
- `--mistral-speaker-model MODEL` - Mistral model for speaker detection (default: environment-based)
- `--mistral-summary-model MODEL` - Mistral model for summarization (default: environment-based)
- `--mistral-temperature TEMP` - Temperature for Mistral generation (0.0-1.0, default: 0.3)
- `--mistral-max-tokens N` - Max tokens for Mistral generation (default: model default)
- `--mistral-cleaning-model MODEL` - Mistral model for transcript cleaning (default: `mistral-small-latest`, cheaper than summary model)
- `--mistral-cleaning-temperature TEMP` - Temperature for Mistral cleaning (0.0-1.0, default: 0.2, lower = more deterministic)

### DeepSeek Provider Options

- `--deepseek-api-key KEY` - DeepSeek API key (can also use `DEEPSEEK_API_KEY` env var)
- `--deepseek-api-base URL` - Custom DeepSeek API base URL (for E2E testing or custom endpoints)
- `--deepseek-speaker-model MODEL` - DeepSeek model for speaker detection (default: `deepseek-chat`)
- `--deepseek-summary-model MODEL` - DeepSeek model for summarization (default: `deepseek-chat`)
- `--deepseek-temperature TEMP` - Temperature for DeepSeek generation (0.0-2.0, default: 0.3)
- `--deepseek-max-tokens N` - Max tokens for DeepSeek generation (default: model default)
- `--deepseek-cleaning-model MODEL` - DeepSeek model for transcript cleaning (default: `deepseek-chat`, cheaper than summary model)
- `--deepseek-cleaning-temperature TEMP` - Temperature for DeepSeek cleaning (0.0-2.0, default: 0.2, lower = more deterministic)

### Grok Provider Options

- `--grok-api-key KEY` - Grok API key (can also use `GROK_API_KEY` env var)
- `--grok-api-base URL` - Custom Grok API base URL (for E2E testing or custom endpoints)
- `--grok-speaker-model MODEL` - Grok model for speaker detection (default: `grok-2`)
- `--grok-summary-model MODEL` - Grok model for summarization (default: `grok-2`)
- `--grok-temperature TEMP` - Temperature for Grok generation (0.0-2.0, default: 0.3)
- `--grok-max-tokens N` - Max tokens for Grok generation (default: model default)
- `--grok-cleaning-model MODEL` - Grok model for transcript cleaning (default: `grok-3-mini`, cheaper than summary model)
- `--grok-cleaning-temperature TEMP` - Temperature for Grok cleaning (0.0-2.0, default: 0.2, lower = more deterministic)

### Ollama Provider Options

- `--ollama-api-base URL` - Custom Ollama API base URL (for E2E testing or custom endpoints, default: `http://localhost:11434/v1`)
- `--ollama-speaker-model MODEL` - Ollama model for speaker detection (default: environment-based)
- `--ollama-summary-model MODEL` - Ollama model for summarization (default: environment-based)
- `--ollama-temperature TEMP` - Temperature for Ollama generation (0.0-2.0, default: 0.3)
- `--ollama-max-tokens N` - Max tokens for Ollama generation (default: model default)
- `--ollama-timeout SECONDS` - Timeout in seconds for Ollama API calls (default: 120, local inference can be slow)
- `--ollama-cleaning-model MODEL` - Ollama model for transcript cleaning (default: `llama3.2:latest`, smaller/faster than summary model)
- `--ollama-cleaning-temperature TEMP` - Temperature for Ollama cleaning (0.0-2.0, default: 0.2, lower = more deterministic)

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

- `--dry-run` - Preview without writing files (includes cost projection for OpenAI providers) (includes cost projection for OpenAI providers)
- `--skip-existing` - Skip episodes with existing output
- `--clean-output` - Remove output directory before processing
- `--fail-fast` - Stop on first episode failure (Issue #379)
- `--max-failures N` - Stop after N episode failures (Issue #379)

### Logging Options

- `--json-logs` - Output structured JSON logs for monitoring/alerting (Issue #379)

## Cost Projection in Dry-Run Mode

When using `--dry-run` with OpenAI providers configured, the pipeline displays a cost projection before execution. This helps you estimate API costs before running expensive operations.

The cost projection includes:

- **Transcription costs** - Based on estimated audio duration (from RSS feed metadata or 30-minute fallback)
- **Speaker detection costs** - Based on estimated token usage (transcript length + prompt overhead)
- **Summarization costs** - Based on estimated token usage (transcript length + prompt overhead)

**Example output:**

```text
Dry run complete. transcripts_planned=5
  - Direct downloads planned: 3
  - Whisper transcriptions planned: 2
  - Output directory: /path/to/output

Cost Projection (Dry Run):
==============================
Transcription (whisper-1):
  - Episodes: 5
  - Estimated audio: 150.0 minutes
  - Estimated cost: $0.9000

Speaker Detection (gpt-4o-mini):
  - Episodes: 5
  - Estimated tokens: ~29,750 input + ~250 output
  - Estimated cost: $0.0045

Summarization (gpt-4o):
  - Episodes: 5
  - Estimated tokens: ~29,950 input + ~750 output
  - Estimated cost: $0.1625

Total Estimated Cost: $1.0670

Note: Estimates are approximate and based on average episode duration. Actual costs may vary based on actual audio length and transcript complexity.
```

**Note:** Cost projection appears when OpenAI or Gemini providers are configured. Estimates use episode durations from RSS feed metadata when available, or a conservative 30-minute average per episode as a fallback.

## Configuration Files

The CLI supports JSON and YAML configuration files:

```bash
python -m podcast_scraper.cli --config config.json
```

## Diagnostic Commands (Issue #379)

### `doctor` Command

The `doctor` command validates your environment and dependencies:

```bash
python -m podcast_scraper.cli doctor
```

**Checks performed:**

- Python version (must be 3.10+)
- `ffmpeg` availability (required for Whisper transcription)
- Write permissions (output directory)
- ML model cache status (Whisper, Transformers, spaCy)
- Network connectivity (optional, with `--verbose`)

**Example output:**

```text
✓ Checking Python version...
  ✓ Python 3.11.9 (required: 3.10+)

✓ Checking ffmpeg...
  ✓ ffmpeg version 6.0

✓ Checking write permissions...
  ✓ Output directory is writable

✓ Checking ML model caches...
  ✓ Whisper: 2 models cached (245 MB)
  ✓ Transformers: 3 models cached (1.2 GB)
  ✓ spaCy: 1 model cached (45 MB)

✓ All checks passed!
```

**Exit codes:**

- `0` - All checks passed
- `1` - Some checks failed

## Exit Codes (Issue #379)

The CLI uses standard exit codes for automation and scripting:

- **0**: Success (pipeline completed successfully, even if some episodes failed)
- **1**: Failure (run-level error: invalid config, missing dependencies, fatal errors)

**Exit Code Policy:**

- Exit code 0 is returned when the pipeline completes, even if individual episodes fail
- Exit code 1 is only returned for run-level failures (invalid configuration, missing dependencies, fatal errors)
- Episode-level failures are tracked in metrics and do not affect exit code unless `--fail-fast` or `--max-failures` is used

**Examples:**

```bash
# Success: pipeline completed, 5 episodes processed, 2 failed
podcast_scraper --rss https://example.com/feed.xml
echo $?  # 0

# Failure: invalid RSS URL
podcast_scraper --rss invalid-url
echo $?  # 1

# Failure: missing ffmpeg
podcast_scraper --rss https://example.com/feed.xml
echo $?  # 1 (dependency check failed)
```

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

# Use Gemini for everything
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider gemini \
  --speaker-detector-provider gemini \
  --summary-provider gemini

# Mix and match providers
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider whisper \
  --speaker-detector-provider gemini \
  --summary-provider openai
```

## See Also

- [Core API](CORE.md) - Programmatic usage
- [Service API](SERVICE.md) - Non-interactive daemon usage
- [Configuration](CONFIGURATION.md) - All configuration options
- [Home](../index.md) - Complete documentation
