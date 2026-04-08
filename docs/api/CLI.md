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
- `--summary-provider PROVIDER` - Provider for summarization (`transformers`, `hybrid_ml`, `openai`, `gemini`, `anthropic`, `mistral`, `deepseek`, `grok`, `ollama`)

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

<a id="transcript-cleaning-hybrid-ml-preprocessing-issue-419"></a>

### Transcript cleaning and hybrid ML preprocessing (Issue #419)

- `--transcript-cleaning-strategy {pattern,llm,hybrid}` - How transcripts are cleaned before summarization (`pattern` = regex/rules; `llm` = LLM-only; `hybrid` = pattern then conditional LLM when using LLM-oriented cleaners). Applies to **LLM summarization providers** and **`hybrid_ml`** (same `cleaning_processor` wiring as API providers).
- `--hybrid-internal-preprocessing-after-pattern PROFILE_ID` - When `--summary-provider hybrid_ml` and `--transcript-cleaning-strategy pattern`, selects the **registered preprocessing profile** applied inside `HybridMLProvider.summarize` after workflow pattern cleaning (default in config: `cleaning_hybrid_after_pattern`). Omit to use the Config default; YAML/config file field: `hybrid_internal_preprocessing_after_pattern`.

See [RFC-042 — Layered transcript cleaning](../rfc/RFC-042-hybrid-summarization-pipeline.md#layered-transcript-cleaning-issue-419), [CONFIGURATION.md](CONFIGURATION.md#transcript-cleaning-configuration-issue-418), and [Preprocessing Profiles Guide](../guides/PREPROCESSING_PROFILES_GUIDE.md).

### OpenAI Provider Options

- `--openai-api-key KEY` - OpenAI API key (can also use `OPENAI_API_KEY` env var)
- `--openai-api-base URL` - Custom OpenAI API base URL (for E2E testing or custom endpoints)
- `--openai-transcription-model MODEL` - OpenAI model for transcription (default: `whisper-1`)
- `--openai-speaker-model MODEL` - OpenAI model for speaker detection (default: `gpt-4o-mini`)
- `--openai-summary-model MODEL` - OpenAI model for summarization (default: `gpt-4o-mini`)
- `--openai-insight-model MODEL` - OpenAI model for GIL `generate_insights` when `gi_insight_source=provider` (default: same as summary model)
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
- `--anthropic-cleaning-model MODEL` - Anthropic model for transcript cleaning (default: `claude-haiku-4-5`, cheaper than summary model)
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
- `--summary-model MODEL` - Summary model to use (MAP-phase, `transformers` provider)
- `--summary-reduce-model MODEL` - Summary reduce model to use (REDUCE-phase, `transformers` provider)
- `--hybrid-map-model MODEL` - Hybrid MAP model when `--summary-provider hybrid_ml` (e.g. `longt5-base`)
- `--hybrid-reduce-model MODEL` - Hybrid REDUCE model (HF id, Ollama tag, or GGUF path for `llama_cpp`)
- `--hybrid-reduce-backend {transformers,ollama,llama_cpp}` - Hybrid REDUCE backend
- `--hybrid-map-device` / `--hybrid-reduce-device` - Devices for hybrid MAP/REDUCE (`cuda`, `mps`, `cpu`, `auto`)
- `--save-cleaned-transcript` / `--no-save-cleaned-transcript` - Persist `.cleaned` transcript alongside source (default: save)

### Cache Management (v2.4.0+)

- `cache --status` - View cache status for all ML models
- `cache --clean [TYPE]` - Clean ML caches (types: `whisper`, `transformers`, `spacy`, `all`)

### Control Options

- `--dry-run` - Preview without writing files (includes LLM cost projection when API providers are configured)
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

**Note:** Cost projection appears when billable LLM providers are configured (OpenAI, Gemini, Mistral, etc.). Estimates use episode durations from RSS feed metadata when available, or a conservative 30-minute average per episode as a fallback.

Optional **YAML pricing overrides** (`pricing_assumptions_file` / `PRICING_ASSUMPTIONS_FILE`) apply to the same formulas; see [LLM cost estimate assumptions](CONFIGURATION.md#llm-cost-estimate-assumptions-optional-yaml) in the configuration guide.

## Configuration Files

The CLI supports JSON and YAML configuration files:

```bash
python -m podcast_scraper.cli --config config.json
```

## Diagnostic Commands (Issue #379, #429)

**Subcommands:** The first argument can be `doctor`, `cache`, or `pricing-assumptions` (e.g. `podcast-scraper doctor`). Startup checks (Python 3.10+, ffmpeg) run only for the main pipeline; they are skipped for these subcommands so you can run doctor even when ffmpeg is missing.

### `pricing-assumptions` Command

Reports whether the pricing YAML resolves, prints `schema_version` and **metadata** (`last_reviewed`, `pricing_effective_date`, `stale_review_after_days`, `source_urls`). Use this after editing rates or on a schedule to see if assumptions need a human refresh.

```bash
python -m podcast_scraper.cli pricing-assumptions
python -m podcast_scraper.cli pricing-assumptions --file config/pricing_assumptions.yaml
make check-pricing-assumptions
```

#### Staleness in the report

The command loads `metadata` from the YAML and evaluates **staleness** only when both of the following are set and valid:

- `last_reviewed` — ISO date (`YYYY-MM-DD`)
- `stale_review_after_days` — positive integer

**Rule:** let `age_days` be the number of whole calendar days from `last_reviewed` to **today** (local date of the machine running the CLI). If `age_days > stale_review_after_days`, the file is **stale**. The command then appends a **Staleness:** section with a short explanation (verify vendor sites and update the YAML).

If either field is missing, unparsable, or `stale_review_after_days` is zero or negative, **no staleness line is printed** (the tool does not guess).

#### `--strict` exit code

With **`--strict`**, the command still prints the full report, but exits with code **1** when the stale condition above is true. Exit code **0** means either not stale, or staleness could not be computed (same cases as “no staleness line”). Use this in CI or release scripts so an expired review date fails the step until someone bumps `last_reviewed` after verifying prices.

```bash
python -m podcast_scraper.cli pricing-assumptions --strict
make check-pricing-assumptions ARGS='--strict'
```

Full field semantics and merge behavior: [Configuration: LLM cost estimate assumptions](CONFIGURATION.md#llm-cost-estimate-assumptions-optional-yaml).

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

- Exit code 0 is returned when the pipeline completes, even if individual episodes fail.
- Exit code 1 is only returned for run-level failures (invalid configuration, missing dependencies, fatal errors).
- Episode-level failures are tracked in metrics and in `index.json` (per-episode `status`, `error_type`, `error_message`, `error_stage`). They do **not** change the exit code: the run still exits 0 when it finishes. Use `--fail-fast` or `--max-failures` to stop processing earlier; those flags do not make the exit code 1.

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

## Grounded insights (`gi`) subcommands

Inspect and explore **Grounded Insight Layer** artifacts (`*.gi.json`) after a run with `generate_gi` enabled (see [Grounded Insights Guide](../guides/GROUNDED_INSIGHTS_GUIDE.md), RFC-050). **Shallow v1 scope** (ML vs `stub` vs bullets, topic explore semantics, deterministic `gi query`, Postgres deferral): [Recorded product decisions (v1, issue 460)](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460). **With `generate_kg` too:** [KG shallow v1 record](../guides/KNOWLEDGE_GRAPH_GUIDE.md#recorded-product-decisions-v1-kg).

```bash
# Validate artifacts (symmetric with `kg validate`; use --strict for full JSON Schema)
python -m podcast_scraper.cli gi validate ./output/metadata --strict

# Export corpus: NDJSON (one artifact per line) or merged bundle (symmetric with `kg export`)
python -m podcast_scraper.cli gi export --output-dir ./output --format ndjson --out gi.ndjson
python -m podcast_scraper.cli gi export --output-dir ./output --format merged --out gi-bundle.json

# One episode: stats, optional full text and quotes (--show)
python -m podcast_scraper.cli gi inspect --episode-path ./output/metadata/ep1.gi.json
python -m podcast_scraper.cli gi inspect --output-dir ./output --episode-id 'sha256:...'

# One insight by id (with evidence spans)
python -m podcast_scraper.cli gi show-insight --id 'insight:<id-from-gi.json>' --episode-path ./output/metadata/ep1.gi.json

# Cross-episode: topic / speaker filters, sort, RFC-style JSON
# When ./output/search/vectors.faiss exists, --topic uses semantic ranking first (RFC-061).
python -m podcast_scraper.cli gi explore --output-dir ./output --topic 'AI regulation' --format json
python -m podcast_scraper.cli gi explore --output-dir ./output --speaker Host --sort time --strict

# UC4: fixed English question patterns → explore JSON or topic leaderboard (RFC-050)
python -m podcast_scraper.cli gi query --output-dir ./output --question 'What insights about inflation?'
python -m podcast_scraper.cli gi query --output-dir ./output --question 'What insights are there about trade?'
python -m podcast_scraper.cli gi query --output-dir ./output --question 'What did Sam say?' --limit 10
python -m podcast_scraper.cli gi query --output-dir ./output --question 'What did Sam say about inflation?'
python -m podcast_scraper.cli gi query --output-dir ./output --question 'Which topics have the most insights?'
```

## Semantic corpus search (`search`, `index`)

Meaning-based retrieval over indexed GIL, summaries, and transcripts (RFC-061). See
[Semantic Search Guide](../guides/SEMANTIC_SEARCH_GUIDE.md). Enable indexing with
`vector_search: true` in config or run `index` manually.

```bash
python -m podcast_scraper.cli search "supply chain disruptions" --output-dir ./output
python -m podcast_scraper.cli search "quantum computing" --output-dir ./output --type insight --format json
python -m podcast_scraper.cli index --output-dir ./output
python -m podcast_scraper.cli index --output-dir ./output --rebuild
python -m podcast_scraper.cli index --output-dir ./output --stats
```

**PRD-017 quality metrics** (grounding rate, quote validity, density) over a run directory:

```bash
make gil-quality-metrics DIR=./output
make gil-quality-metrics DIR=./output ARGS='--enforce --min-avg-insights 3 --min-avg-quotes 5'
make compare-gil-runs REF=./output/run_ref CAND=./output/run_ml
make kg-quality-metrics DIR=./output ARGS='--enforce --json'
make quality-metrics-ci
```

`compare-gil-runs` expects each path to be a **pipeline run root** with `metadata/*.gi.json`
(see `docs/wip/gil-ml-vs-openai-outcome-benchmark.md`).

## Knowledge Graph (`kg`) subcommands

Inspect and export **Knowledge Graph** artifacts (`*.kg.json`) after a run with `generate_kg`
enabled. Symmetric to the `gi` subcommand for GIL.

```bash
# Validate all kg.json under a directory (strict schema)
python -m podcast_scraper.cli kg validate ./output/metadata --strict

# Inspect one episode (by file or by episode id under output dir)
python -m podcast_scraper.cli kg inspect --episode-path ./output/metadata/1_ep.kg.json
python -m podcast_scraper.cli kg inspect --output-dir ./output --episode-id 'sha256:...'

# Export corpus: NDJSON (one artifact per line) or single merged JSON bundle
python -m podcast_scraper.cli kg export --output-dir ./output --format ndjson --out kg.ndjson
python -m podcast_scraper.cli kg export --output-dir ./output --format merged --out kg-bundle.json

# Aggregate entities across episodes; topic pairs that co-occur in the same episode
python -m podcast_scraper.cli kg entities --output-dir ./output --min-episodes 2 --format json
python -m podcast_scraper.cli kg topics --output-dir ./output --min-support 2 --format json
```

Details: [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md), [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md). **Shallow v1 scope** (extraction + ML, no `kg query` IR, Postgres deferral): [Recorded product decisions (v1, KG shallow)](../guides/KNOWLEDGE_GRAPH_GUIDE.md#recorded-product-decisions-v1-kg). **GIL companion:** [Recorded product decisions (v1, issue 460)](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460).

## See Also

- [Core API](CORE.md) - Programmatic usage
- [Service API](SERVICE.md) - Non-interactive daemon usage
- [Configuration](CONFIGURATION.md) - All configuration options
- [Home](../index.md) - Complete documentation
