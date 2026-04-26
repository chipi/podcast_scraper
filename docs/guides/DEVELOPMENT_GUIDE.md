# Development Guide

> **Maintenance Note**: This document should be kept up-to-date as linting rules, Makefile
> targets, pre-commit hooks, CI/CD workflows, or development setup procedures evolve. When
> adding new checks, tools, workflows, or environment setup steps, update this document
> accordingly.

This guide provides detailed implementation instructions for developing the podcast scraper.
For high-level architectural decisions and design principles, see [Architecture](../architecture/ARCHITECTURE.md).

## Polyglot repository (Python + web viewer)

The **Python** toolchain (`Makefile`, `pytest`, `pip install -e …`) is anchored at the **repo
root**. The **GI/KG viewer** is a **Node** project in **`web/gi-kg-viewer/`** (Vite, Vitest,
Playwright). Environment templates differ on purpose (**`config/examples/.env.example`** vs
**`web/gi-kg-viewer/.env.example`**). For a single onboarding story and command cheat sheet, see
the **[Polyglot repository guide](POLYGLOT_REPO_GUIDE.md)**.

## Testing

For comprehensive testing information, see the dedicated testing documentation:

- **[Testing Strategy](../architecture/TESTING_STRATEGY.md)** - Testing philosophy, test pyramid, decision criteria
- **[Testing Guide](TESTING_GUIDE.md)** - Quick reference, test execution commands
- **[Experiment Guide](EXPERIMENT_GUIDE.md)** — Complete guide: datasets, baselines, experiments, and evaluation
- **[Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md)** — Frozen release profiles (RSS, CPU%, wall time per stage; RFC-064)
- **[Unit Testing Guide](UNIT_TESTING_GUIDE.md)** - Unit test mocking patterns and isolation
- **[Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md)** - Integration test guidelines
- **[E2E Testing Guide](E2E_TESTING_GUIDE.md)** - E2E server, real ML models
- **[Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)** - What to test, prioritization

### Quick Reference

| Layer | Directory | Speed | Mocking |
| ------- | ----------- | ------- | --------- |
| Unit | `tests/unit/` | < 100ms | All mocked |
| Integration | `tests/integration/` | < 5s | External mocked |
| E2E | `tests/e2e/` | < 60s | No mocks |

**FastAPI viewer:** unit tests in **`tests/unit/podcast_scraper/server/`**; wired HTTP integration in
**`tests/integration/server/`**. Playwright UI tests are separate — [Testing Guide — Browser E2E](TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2).

### Running Tests

```bash
make check-unit-imports        # Verify modules can import without ML dependencies
make check-test-policy         # Enforce 3-tier ML/AI testing policy (importorskip, ml_models, empty files)
make deps-analyze              # Analyze module dependencies (with report)
make deps-check                # Check dependencies (exits on error)
make analyze-test-memory       # Analyze test memory usage (default: test-unit)
make test-unit                 # Unit tests (parallel)
make test-integration          # Integration tests (parallel, reruns)
make test-e2e                  # E2E tests (parallel, with reruns)
make test                      # All tests
make test-fast                 # Unit + critical path integration + critical path E2E

# Manual validation (not CI — run from laptop when needed)
make pipeline-validate                              # All providers × full pipeline
make pipeline-validate PROVIDER=gemini MODEL=gemini-2.5-flash-lite  # Single provider
make pipeline-validate PV_ARGS="--all-cloud"        # 6 cloud providers
make pipeline-validate PV_ARGS="--all-local"        # Core 5 Ollama (ADR-077)
make transcription-sweep                            # Local Whisper model comparison
```

### Fast Validation for Changed Files

When fixing a few files to stabilize a failing PR, use `make validate-files` to run only impacted tests. This is much faster than running the entire test suite.

**Usage:**

```bash
# Validate specific files (runs all test types by default)
make validate-files FILES="src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py"

# Unit tests only (fastest, < 1 minute typically)
make validate-files-unit FILES="src/podcast_scraper/config.py"

# Include integration/E2E tests
make validate-files FILES="..." TEST_TYPE=all

# Fast mode (critical_path tests only)
make validate-files-fast FILES="src/podcast_scraper/config.py"
```

**What it does:**

1. **Linting/formatting** on changed files (black, isort, flake8, mypy)
2. **Discovery** of impacted tests via module markers
3. **Execution** of only those tests (unit/integration/e2e based on TEST_TYPE)

**Performance:**

- **Unit tests only**: < 1 minute for typical changes
- **With integration**: < 2 minutes
- **Full suite (all types)**: < 5 minutes (still faster than `make ci-fast` which takes 6-10 minutes)

**How it works:**

Tests are tagged with module markers (e.g., `module_config`, `module_workflow`) that map to source modules. When you specify changed files, the system:

- Maps files to modules (e.g., `config.py` → `module_config`)
- Finds tests tagged with those module markers
- Runs only those tests

**Note:** This is a **development tool** for fast iteration. For full validation before PR, still use `make ci-fast` or `make ci`.

### ML Dependencies in Tests

Modules importing ML dependencies at **module level** will fail unit tests in CI.

**Solutions:**

1. **Mock before import** (recommended):

   ```python
   from unittest.mock import MagicMock, patch

   with patch.dict("sys.modules", {"spacy": MagicMock()}):
       from podcast_scraper import speaker_detection
   ```

2. **Use lazy imports**: Import inside functions, not at module level

3. **Verify imports work without ML deps**: Run `make check-unit-imports` before pushing
   - This verifies modules can be imported without ML dependencies installed
   - Runs automatically in CI before unit tests
   - Use when: adding new modules, refactoring imports, or debugging CI failures

4. **Enforce testing policy**: Run `make check-test-policy` before pushing
   - Checks: no `pytest.importorskip()` in unit tests, no `*_AVAILABLE` skip guards
     in unit tests, no `@pytest.mark.ml_models` in integration tests, no empty test files
   - Script: `scripts/tools/check_test_policy.py` (pass `--fix-hint` for remediation tips)
   - Runs automatically in `make ci` and `make ci-fast`
   - Use when: adding, moving, or deleting test files

5. **Run unit tests**: Run `make test-unit` before pushing

### Module Dependency Analysis

Analyze module dependencies to detect architectural issues like circular imports and excessive coupling.

**When to use:**

- After refactoring modules or moving code between modules
- When adding new imports or dependencies
- Before major refactoring to understand current architecture
- When debugging circular import errors
- Before committing if you changed module structure

**Usage:**

```bash
make deps-analyze    # Full analysis with JSON report (reports/deps-analysis.json)
make deps-check      # Quick check (exits with error if issues found, CI-friendly)
```

**What it checks:**

- **Circular imports**: Detects cycles in the import graph (should be 0)
- **Import thresholds**: Flags modules with >15 imports (suggests refactoring)
- **Import patterns**: Analyzes import structure across all modules

**Output:**

- Console output with issues and summary
- JSON report (with `--report` flag) saved to `reports/deps-analysis.json`
- Visual dependency graphs (generated separately via `make deps-graph`)

**Runs automatically in CI:** In nightly workflow (`nightly-deps-analysis` job) with 90-day artifact retention for tracking architecture changes over time.

**See also:** [Module Dependency Analysis](../architecture/ARCHITECTURE.md#module-dependency-analysis) for detailed documentation.

### Test Memory Analysis

Analyze memory usage during test execution to identify memory leaks, excessive resource usage, and optimization opportunities.

**When to use:**

- Debugging memory issues (tests crash with OOM errors, system becomes unresponsive)
- Optimizing test performance (finding optimal worker count, understanding resource usage)
- Investigating memory leaks (memory growth over time, system memory decreases after tests)
- Capacity planning (determining required RAM for CI, understanding resource needs)
- Before major changes (after adding ML model tests, changing parallelism settings)

**Usage:**

```bash
# Analyze default test target (test-unit)
make analyze-test-memory

# Analyze specific test target
make analyze-test-memory TARGET=test-unit
make analyze-test-memory TARGET=test-integration
make analyze-test-memory TARGET=test-e2e

# Analyze with limited workers (to test memory impact)
make analyze-test-memory TARGET=test-integration WORKERS=4
```

**What it monitors:**

- **Peak memory usage**: Maximum memory consumed during test execution
- **Average memory usage**: Average memory over test duration
- **Worker processes**: Number of parallel test workers spawned
- **Memory growth**: Detects potential memory leaks (memory increasing over time)
- **System resources**: CPU cores, total/available memory (before/after)

**Output:**

- Memory usage statistics (peak, average, worker count)
- Memory usage over time (sample points every 2 seconds)
- Recommendations (warnings if thresholds exceeded)
- System resource changes (before/after comparison)

**Recommendations provided:**

- Warns if peak memory > 80% of total RAM
- Warns if worker count > CPU cores
- Warns if peak memory > 8 GB
- Suggests optimal worker count (CPU cores - 2)
- Detects memory growth (potential leaks)

**Dependencies:** Requires `psutil` package (`pip install psutil`)

**See also:** [Troubleshooting Guide](TROUBLESHOOTING.md#memory-issues-with-ml-models) for memory issue debugging.

### Quality Evaluation

Evaluation is handled automatically by the experiment runner. When you run an experiment with `--baseline` and/or `--reference` flags, the system automatically computes metrics and comparisons.

**When to use:**

- After modifying cleaning logic in `preprocessing.py`
- When testing new summarization models or chunking strategies
- Before major releases to ensure no regression in output quality

**Usage:**

```bash
# Run experiment with automatic evaluation
make experiment-run \
  CONFIG=data/eval/configs/my_config.yaml \
  BASELINE=baseline_prod_authority_v1 \
  REFERENCE=silver_gpt52_v1
```

For details, see the **[Experiment Guide](EXPERIMENT_GUIDE.md)** (Step 4: Evaluate Results).

## Environment Setup

### Virtual Environment

**Quick setup:**

```bash

bash scripts/setup_venv.sh
source .venv/bin/activate

```

**Note:** The `setup_venv.sh` script automatically installs the package in editable mode
(`pip install -e .`), which is required for:

- Running CLI commands: `python3 -m podcast_scraper.cli` (typical argv: **`--profile`**, **`--config`**, **`--feeds-spec`** — [CLI.md — Quick Start](../api/CLI.md#quick-start))
- Importing the package in Python: `from podcast_scraper import ...`
- Running tests that import the package

**Manual setup (if not using setup_venv.sh):**

If you create a virtual environment manually, you **must** install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,llm]"  # Editable mode with dev, ML, and LLM extras
```

**Optional — spaCy wheels on disk:** Run `make download-spacy-wheels`, then `make init`; if
`wheels/spacy/*.whl` exists, the Makefile sets `PIP_FIND_LINKS` for you (see
[Dependencies Guide — Optional local wheel cache](DEPENDENCIES_GUIDE.md#optional-local-wheel-cache-for-spacy-models)).

**Why editable mode (`-e`)?**

- Changes to source code are immediately available without reinstalling
- Required for development workflow
- Allows `python3 -m podcast_scraper.cli` to work

### Updating Virtual Environment Dependencies

**CRITICAL: Update venv when dependency ranges change**

When `pyproject.toml` dependency version ranges are modified (e.g., `black>=23.0.0,<27.0.0`), you **must** update your local virtual environment to match what CI installs.

**Why this matters:**

- CI installs fresh dependencies each run, getting the **latest version** in the range
- Your local venv may have an **older version** installed when the range was smaller
- Pip doesn't auto-upgrade packages that still satisfy the constraint
- This causes **version mismatches** between local and CI

**When to update:**

- After modifying dependency version ranges in `pyproject.toml`
- After pulling changes that modify `pyproject.toml` dependency ranges
- When CI fails with formatting/linting errors but local passes
- When you see "File would be reformatted" in CI but not locally

**How to update:**

```bash
# Update all dev dependencies to latest in their ranges
pip install --upgrade -e .[dev]

# Or update specific tool (e.g., black)
pip install --upgrade "black>=23.0.0,<27.0.0"

# Verify version matches CI
python -m black --version  # Should show latest in range (e.g., 26.1.0)
```

**Common symptoms of stale venv:**

- Local: `make format-check` passes
- CI: `make format-check` fails with "would reformat"
- Local: `make lint` passes
- CI: `make lint` fails with different errors
- Tool versions differ: `python -m black --version` shows older version than CI logs

**Prevention:**

After modifying `pyproject.toml` dependency ranges, always run:

```bash
pip install --upgrade -e .[dev]
```

### Environment Variables

**Note:** Setting up a `.env` file is **optional** but **recommended**, especially if you plan to
use OpenAI providers or want to customize logging, paths, or performance settings.

1. **Copy example `.env` file:**

   ```bash
   cp config/examples/.env.example .env
   ```

2. **Edit `.env` and add your settings:**

   ```bash

   # OpenAI API key (required for OpenAI providers)

   OPENAI_API_KEY=sk-your-actual-key-here

   # Logging

   LOG_LEVEL=DEBUG

   # Paths

   OUTPUT_DIR=/data/transcripts
   LOG_FILE=/var/log/podcast_scraper.log
   CACHE_DIR=/cache/models

   # Performance tuning

   WORKERS=4
   TRANSCRIPTION_PARALLELISM=3
   PROCESSING_PARALLELISM=4
   SUMMARY_BATCH_SIZE=2
   SUMMARY_CHUNK_PARALLELISM=2
   TIMEOUT=60
   SUMMARY_DEVICE=cpu
   ```

3. **The `.env` file is automatically loaded** via `python-dotenv` when `podcast_scraper.config` module is imported.

**Security notes:**

- `.env` is in `.gitignore` (never committed)
- `config/examples/.env.example` is safe to commit (template only)
- API keys are never logged or exposed
- Environment variables take precedence over `.env` file
- HuggingFace model loading uses `trust_remote_code=False`; only enable `trust_remote_code=True` if a model's documentation explicitly requires it and the source is trusted (Issue #429).

**Priority order** (for each configuration field):

1. **Config file field** (highest priority) - if the field is set in the config file and not `null`/empty, it takes precedence
2. **Environment variable** - only used if the config file field is `null`, not set, or empty
3. **Default value** - used if neither config file nor environment variable is set

**Exception**: `LOG_LEVEL` environment variable takes precedence over config file (allows easy runtime log level control).

**Note**: You can define the same field in both the config file and as an environment variable.
The config file value will be used if it's set.
This allows config files for project defaults and environment variables for deployment-specific overrides.

**See also:**

- `docs/api/CONFIGURATION.md` - Configuration API reference (includes environment variables and [Twelve-factor app alignment (config)](../api/CONFIGURATION.md#twelve-factor-app-alignment-config))
- `docs/rfc/RFC-013-openai-provider-implementation.md` - API key management details
- `docs/prd/PRD-006-openai-provider-integration.md` - OpenAI provider requirements

### ML Model Cache Management

The project uses a local `.cache/` directory for ML models (Whisper, HuggingFace Transformers,
spaCy). This cache can grow large (several GB) with both dev/test and production models.

#### Preloading Models

To download and cache all required ML models:

```bash
# Preload test models (small, fast models for local dev/testing)
make preload-ml-models

# Preload production models (large, quality models)
make preload-ml-models-production
```

**Cache locations:**

- Whisper: `.cache/whisper/` (e.g., `tiny.en.pt`, `base.en.pt`)
- HuggingFace: `.cache/huggingface/hub/` (e.g., `facebook/bart-base`, `allenai/led-base-16384`)
- spaCy: `.cache/spacy/` (if using local cache)

#### Transcript hash cache (JSON)

When `transcript_cache_enabled` is on, transcripts are keyed by a hash of the episode audio file
under `transcript_cache_dir` (default: `.cache/transcripts/`). Each entry is a JSON file containing
the transcript text plus metadata (`cached_at`, optional `provider` / `model`). If the transcription
provider returned timed **segments** (Whisper-style `start` / `end` / `text` per item), those are
also stored in the same JSON under an optional `segments` array. On a **cache hit**, the pipeline
writes the transcript file and, when segments are present, the sibling **`*.segments.json`** file
next to it so GI quote audio timestamps match a fresh transcription run. Cache files created before
segments were stored omit `segments`; re-transcribe or clear that cache entry to populate them.

**GI segment alignment:** Quote audio timestamps (`timestamp_*_ms`) map character offsets using
`*.segments.json` only when the concatenation of segment `text` fields matches the GI transcript
string within **50 characters** (`SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA` in
`src/podcast_scraper/gi/pipeline.py`). Screenplay formatting or edited transcripts without
regenerated segments can trip this guard; the pipeline logs **one warning per episode artifact**
when it skips segment-based timestamps and segment-derived speakers (GitHub issue #545).

**Direct RSS transcript download (`transcript_source=direct_download`):** Plain `.txt` or `.html`
transcript URLs do not carry timed cues, so GI quote **audio** timestamps stay at zero unless you add
a compatible `*.segments.json` by other means. When the feed serves **WebVTT (`.vtt`) or SubRip
(`.srt`)**, the downloader normalizes to **`transcripts/… .txt`** plus a sibling **`… .segments.json`**
(parsed cues; GitHub issue #544) so metadata and GI use plain text with the same segment shape as
Whisper. If parsing yields no cues, the raw caption file is stored as before. Whisper runs and
transcript cache behavior are separate (GitHub issue #540).

**`--skip-existing` and missing `.segments.json` (GitHub issue #542):** With
`skip_existing: true`, the pipeline treats an existing Whisper transcript **`.txt`** as “done” for
transcription even when there is **no** sibling **`basename.segments.json`**. Summaries and GI still
run, but **GI quote audio timestamps** (`timestamp_*_ms`) stay at **0** until a compatible segment
sidecar exists (same alignment rules as above). **Recovery without changing defaults:** delete or
rename the transcript `.txt` (or run once with `skip_existing: false`) so the episode is
re-transcribed with a segment-capable provider, or clear the relevant transcript cache entry if you
use `transcript_cache_enabled`. **Opt-in automation:** set **`backfill_transcript_segments: true`**
(YAML / env) or **`--backfill-transcript-segments`** (CLI) together with **`generate_gi`**: the
workflow will **not** skip transcription solely because the `.txt` exists when the sidecar is
missing, and **`append`** will not mark the episode complete until **`transcript_file_path.txt`** and
**`transcript_file_path.segments.json`** (sibling of the `.txt`) both exist. Default is **`false`** so
existing skip behavior and API cost stay unchanged.

**Scope:** **`backfill_transcript_segments`** hooks **`download_media_for_transcription`** for **Whisper-style** outputs under **`transcripts/`**. It does **not** override **`skip_existing`** on other transcript persistence paths (for example **direct RSS transcript** downloads that hit a separate save helper): those still skip when the target file already exists unless you remove or replace the file.

**Transcription provider vs GI quote audio timing (GitHub issue #543):** GI **`timestamp_*_ms`** on quotes
needs either a populated **`.segments.json`** (from the pipeline) or **non-empty `segments`** from
**`transcribe_with_segments`**. If the provider returns **text but an empty `segments` list**, behavior
matches missing sidecar: quotes keep **character offsets**, but **audio times stay 0**. This is
**independent of transcript cache** ([#540](https://github.com/chipi/podcast_scraper/issues/540)): the
cache cannot invent timings the API did not return.

| `transcription_provider` | Transcription | Timed segments for GI audio seek |
| --- | --- | --- |
| **`whisper`** (local **MLProvider**) | Yes | **Yes** when the model returns segments |
| **`openai`** | Yes | **Yes** when the API returns segment-style alignment (Whisper API path) |
| **`gemini`** | Yes (text) | **No** — integration returns empty `segments` (no native timed chunks in our path) |
| **`mistral`** | Yes (text) | **No for now** — returns empty `segments` until Voxtral (or API) exposes a mapped shape |
| **`anthropic`** | **Not supported** for audio (`NotImplementedError` on transcribe) | N/A — use **whisper** or **openai** for audio + GI timing |

For **GI + audio seek in the viewer**, prefer **`whisper`** or **`openai`** as `transcription_provider`.
Programmatic checks: **`ProviderCapabilities.supports_gi_segment_timing`** and per-episode metadata
**`processing.config_snapshot.ml_providers.transcription.gi_segment_timing_expected`** (written when
transcription provider info is recorded).

<a id="gi-quote-speaker-id"></a>

**GI quote `speaker_id` (GitHub issue [#541](https://github.com/chipi/podcast_scraper/issues/541)):** On each **Quote** node, **`speaker_id`** is set only when **`_speaker_id_for_char_range`** in `src/podcast_scraper/gi/pipeline.py` can map the quote’s character span to timed **segments** (from **`.segments.json`** or non-empty in-memory **`transcript_segments`**) that carry **`speaker`** or **`speaker_id`**, and the same **concatenated segment `text`** vs GI transcript alignment gate applies as for **`timestamp_*_ms`** (**`SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA`**; see **GI segment alignment** above — on failure, segment-based speakers and timestamps are skipped and the pipeline logs **one warning per episode**, [#545](https://github.com/chipi/podcast_scraper/issues/545)). **`speaker_id`** is **`null`** when segments are **time + text only** (common **Whisper** / **OpenAI** paths), when segments are missing ([#540](https://github.com/chipi/podcast_scraper/issues/540), [#542](https://github.com/chipi/podcast_scraper/issues/542)), when the provider returns empty **`segments`** ([#543](https://github.com/chipi/podcast_scraper/issues/543)), or for stub artifacts. Episode **NER** host/guest lists and **screenplay** rotation heuristics do **not** populate **`quote.speaker_id`**. When **`speaker_id`** is non-null, it is normally a **`person:{slug}`** id (RFC-072 **Person** node); legacy corpora may still use **`speaker:{slug}`** until migrated ([GI ontology — Person / Quote properties](../architecture/gi/ontology.md)).

**Future adapters:** When a vendor exposes **word- or chunk-level timestamps** in a stable API shape, a
normalizer can map them to `{start, end, text}` and write **`.segments.json`** like Whisper paths.
Ship only with fixtures and unit tests; no adapter is implied until that shape exists.

**See also:** `.cache/README.md` for detailed cache structure and usage.

#### Backup and Restore

The cache directory can be backed up and restored for easy management:

**Backup:**

```bash
# Create backup (saves to ~/podcast_scraper_cache_backups/)
make backup-cache

# Dry run to preview
make backup-cache-dry-run

# List existing backups
make backup-cache-list

# Clean up old backups (keep 5 most recent)
make backup-cache-cleanup
```

**Restore:**

```bash
# Interactive restore (lists backups, prompts for selection)
make restore-cache

# Restore specific backup
python scripts/cache/restore_cache.py --backup cache_backup_20250108-120000.tar.gz

# Force overwrite existing .cache
python scripts/cache/restore_cache.py --backup 20250108 --force
```

**What gets backed up:**

- All model files (Whisper, HuggingFace, spaCy)
- Cache directory structure
- Excludes: `.lock` files, `.incomplete` downloads, temporary files

**See also:**

- `scripts/cache/backup_cache.py` - Backup script documentation
- `scripts/cache/restore_cache.py` - Restore script documentation
- `.cache/README.md` - Cache directory documentation

#### Process Safety for ML Cache Operations (RFC-074)

ML model loading triggers heavy filesystem I/O (readdir, lstat, mmap on
multi-GB files). On macOS (APFS), this contends for a global kernel lock
and can cause processes to enter uninterruptible wait (`UE` state) where
`kill -9` has no effect.

**Key safeguards in the build system:**

- **No ML imports at Makefile parse time** -- `make help` completes in
  under 1 second with zero Python processes spawned.
- **Filesystem-only cache checks** -- `_is_transformers_model_cached()`
  checks for `config.json` + weight files instead of calling
  `AutoTokenizer.from_pretrained()`, avoiding heavy disk I/O.
- **Offline mode** -- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  are exported for all Makefile recipes, preventing accidental model
  downloads during builds.
- **Preload timeout** -- `preload_ml_models.py` has a 600-second
  `signal.alarm` hard timeout to prevent indefinite hangs.

**Diagnostic commands:**

```bash
make check-zombie      # Detect unkillable (UE state) Python processes
make check-spotlight   # Verify Spotlight indexing status (macOS)
make cleanup-processes # Kill leftover Python/test processes
```

If `make check-zombie` reports UE processes, reboot is the only option.
After reboot, run Disk Utility First Aid on the boot volume.

**Recommended macOS settings:** Disable Spotlight indexing on
`~/.cache/huggingface`, `~/.cache/whisper`, and `.venv/` directories
(System Settings, Spotlight, Privacy) to reduce APFS lock contention.

Full details: [RFC-074](../rfc/RFC-074-process-safety-ml-workloads-macos.md),
[Architecture -- Process Safety](../architecture/ARCHITECTURE.md#process-safety-for-ml-workloads-rfc-074).

#### Cleaning Cache

To remove cached models (useful for testing or freeing disk space):

```bash
# Clean all ML model caches (user cache locations)
make clean-cache

# Clean build artifacts and caches
make clean-all
```

**Note:** `make clean-cache` removes models from `~/.cache/` locations, not the project-local
`.cache/` directory. To remove the project-local cache, manually delete `.cache/` or use the
restore script to replace it.

## Semantic corpus search (RFC-061) {#semantic-corpus-search-rfc-061}

Optional **FAISS** vector index under `<output_dir>/search/` for meaning-based retrieval over
GIL, summaries, and transcripts. Enable with **`vector_search: true`** in config (YAML
keys mirror `Config`: `vector_index_path`, `vector_embedding_model`,
`vector_chunk_size_tokens`, `vector_chunk_overlap_tokens`). The pipeline runs
embed-and-index after finalize when enabled; you can also run **`podcast index`** /
**`podcast search`** and get **semantic `gi explore --topic`** when the index exists.
**Qdrant** and other platform backends are **Draft** [RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md).

**Full guide:** [Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md).

### Insight clustering and multi-quote extraction (#599, #600, #601) {#insight-clustering}

**Insight clustering** groups semantically similar GI insights across episodes using the same
average-linkage algorithm as topic clustering. CLI: `podcast insight-clusters --output-dir ./output`.
Writes `insight_clusters.json` to `<output-dir>/search/`. Module: `search/insight_clusters.py`.

**Multi-quote extraction** (#600) changed all 8 providers from single-quote to multi-quote per
insight (uncapped). Prompt: "Extract all short verbatim quotes." Parser handles both
`{"quotes": [...]}` (new) and `{"quote_text": "..."}` (backward compat). ML provider uses
`answer_candidates(top_k=3)`. Quote dedup by text prevents LLM repetition.

**Cluster context expansion** (#601) adds cross-episode evidence to `gi explore` results.
Flag: `gi explore --expand-clusters`. Loads `insight_clusters.json`, finds cluster members from
other episodes, and displays their quotes alongside the matched insight. Module:
`search/insight_cluster_context.py`.

**Tests:** Unit: `test_insight_clusters.py` (11 tests), `test_insight_cluster_context.py`
(12 tests). Integration: `tests/integration/search/test_insight_clusters_cli.py` (3 tests).

## GI / KG browser viewer {#gi-kg-browser-viewer-local-prototype}

**v2 (Vue + FastAPI, RFC-062)** is the supported viewer for graph, dashboard,
semantic search, and explore.

**Layout and env files (root vs `web/gi-kg-viewer/`):** [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md).

### Viewer v2 (RFC-062 / `#489`)

**FastAPI (all `/api/*` routes, `create_app`, OpenAPI):** [Server Guide](SERVER_GUIDE.md)
— endpoint table, Corpus Library, index rebuild, **`/docs`** and **`/openapi.json`**
when `serve` is running. Platform routes under `routes/platform/` are **not** mounted yet (stubs only).

- **Location:** `web/gi-kg-viewer/`
- **Python extra:** `[server]` (FastAPI + uvicorn) — not part of the default `make init`
  line (`.[dev,ml,llm]`); add `server` when you work on or run the viewer API.
  See [Dependencies Guide — Canonical optional extras](DEPENDENCIES_GUIDE.md#canonical-optional-extras) for the full list (`dev`, `ml`, `compare`, `llm`, `server`).
- **End-user flow:** Build `dist/` once (`npm install && npm run build` in
  `web/gi-kg-viewer`), then `python -m podcast_scraper.cli serve --output-dir <run>`;
  open **<http://127.0.0.1:8000>** and set **Corpus root** to that same directory.
  Full walkthrough:
  [web/gi-kg-viewer/README.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md)
  and
  [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md).
- **CIL pill to graph focus:** Digest Recent and the Episode subject rail use
  `web/gi-kg-viewer/src/utils/cilGraphFocus.ts` so clustered pills pass optional
  `tc:…` ids into `pendingFocusCameraIncludeRawIds` (same idea as Search hits). Further
  entry surfaces and audit notes: [Viewer graph spec — Graph focus entry points](../architecture/VIEWER_GRAPH_SPEC.md#graph-focus-entry-points).

**Makefile targets (repository root):**

| Target | Purpose |
| ------ | ------- |
| `make serve` | Runs **`serve-api`** and **`serve-ui`** in parallel (API + Vite dev on 5173). |
| `make serve-api SERVE_OUTPUT_DIR=…` | FastAPI only (default port **8000**). |
| `make serve-ui` | Vite dev server only (`web/gi-kg-viewer`, port **5173**, proxies `/api` → 8000). |
| `make test-ui` | Vitest unit tests for TS utility logic (parsing, merge, metrics, formatting). Fast (~150 ms), no browser. |
| `make test-ui-e2e` | Playwright browser tests: `npm install`, `playwright install firefox`, `npm run test:e2e` (Vite on **5174** inside Playwright config — no clash with 5173). |
| `make verify-gil-offsets-strict` | **Quote** vs **indexed transcript chunk** character alignment on a corpus (set **`GIL_OFFSET_VERIFY_DIR`**; optional **`GIL_OFFSET_MIN_RATE`**, default **0.95**). Supports **feed-nested** metadata. See [Semantic Search Guide — lift & verification](SEMANTIC_SEARCH_GUIDE.md#chunk-to-insight-lift-and-offset-verification-rfc-072--528). |

**Contributor notes:**

- **Viewer UX changes (order matters):** If you change **UI** that users or Playwright see
  (copy, labels, layout, routes, theme tokens, accessible names, list/load flows), update **(1)**
  [`e2e/E2E_SURFACE_MAP.md`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md),
  **(2)** `e2e/*.spec.ts` / `helpers.ts` / `fixtures.ts` and run **`make test-ui-e2e`**, **(3)**
  [VIEWER_IA.md](../uxs/VIEWER_IA.md) when **shell IA** changes, then [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) and/or the relevant [feature UXS](../uxs/index.md) when the **visual or experience spec** changes.
  Checklist: [E2E Testing Guide — When you change viewer UX](E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow).

### Debugging viewer UI

When something looks wrong in the GI/KG viewer (wrong panel, missing control, failed navigation,
Playwright or MCP clicking the wrong **Search**):

- Use the browser’s **Network** and **Console** (and Vue DevTools if you use them) for requests,
  errors, and component state.
- Use [`e2e/E2E_SURFACE_MAP.md`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
  as the **automation and accessibility contract**: stable roles and labels, entry paths, and notes
  on ambiguous names. It applies to **Playwright failures**, **manual reproduction**, and
  **agent-driven** Chrome / DevTools MCP sessions, not only to writing new specs.
- **Insight** graph node detail (`NodeDetail.vue`): meta line (**grounded**, **insight_type**, **position_hint**, optional **confidence**), **Prefill semantic search**, **Set Explore filters**, **Supporting quotes** (**SUPPORTED_BY**), **Episode on graph** when resolvable — see [UXS-004 Graph exploration](../uxs/UXS-004-graph-exploration.md).
- For the agent-assisted loop (live Chrome + MCP, headless Playwright, validation symmetry), see
  [Agent-Browser Closed Loop Guide](AGENT_BROWSER_LOOP_GUIDE.md). For the Playwright change checklist,
  see [E2E Testing Guide — Browser E2E](E2E_TESTING_GUIDE.md#browser-e2e-playwright).

**More contributor notes:**

- UI E2E uses **Firefox** (see `web/gi-kg-viewer/playwright.config.ts`).
- Pytest coverage for the same APIs lives under `tests/unit/podcast_scraper/server/`
  and `tests/integration/server/` (e.g. `test_server_api.py` — wired app + real filesystem;
  see Server Guide for other modules).
- **Full server reference:** [Server Guide](SERVER_GUIDE.md) — architecture, all endpoints,
  adding routes, testing, platform evolution.

**See also:** [GIL / KG / CIL cross-layer](GIL_KG_CIL_CROSS_LAYER.md) · [Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md) · [Grounded Insights
Guide](GROUNDED_INSIGHTS_GUIDE.md) · [Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md) ·
[CLI API](../api/CLI.md) (`gi` / `kg` / `search` / `index` subcommands).

## Run Comparison Tool (RFC-047 / RFC-066)

Streamlit UI for comparing ML evaluation runs and
performance profiles. Lives in `tools/run_compare/`
(outside `src/`).

- **Extra:** `pip install -e ".[compare]"`
- **Run:** `make run-compare`
- **Pages:** Home (ROUGE charts), KPIs, Delta
  (baseline vs candidates), Episodes (side-by-side
  diffs), Performance (frozen profiles, resource
  deltas, per-stage trends)
- **Details:**
  [tools/run_compare/README.md](https://github.com/chipi/podcast_scraper/blob/main/tools/run_compare/README.md),
  [RFC-047](../rfc/RFC-047-run-comparison-visual-tool.md),
  [RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)

## Evaluation artifacts (`data/eval/`)

The `data/eval/` tree is the project's ML quality
infrastructure. Contributors use it to validate new
models, compare provider outputs, and gate regressions
before merging.

### Layout

| Directory | Purpose | Mutable? |
| --------- | ------- | -------- |
| `sources/` | Immutable raw inputs (transcripts, RSS XML, metadata) | No |
| `datasets/` | Versioned dataset definitions (JSON) | No (once published) |
| `materialized/` | Regenerable copies of transcripts + metadata for a dataset | Yes (regenerate) |
| `configs/` | Experiment input YAML (task, backend, dataset, prompts, params) | Yes |
| `baselines/` | Frozen "known good" runs for regression comparison | No (once promoted) |
| `references/` | Quality targets: **silver** (LLM-generated) and **gold** (human-verified) | No (once promoted) |
| `runs/` | Disposable experiment outputs; promote to baseline/reference or delete | Yes |
| `schemas/` | JSON schemas for metrics, episode metadata, NER gold | No |

**Immutability rule:** `sources/`, `datasets/`,
`baselines/`, and `references/` are immutable once
published. `materialized/` and `runs/` can be
regenerated or discarded.

### Common workflows

```bash
# Run an experiment against a baseline
make experiment-run \
  CONFIG=data/eval/configs/my_config.yaml \
  BASELINE=baseline_prod_authority_v1

# Compare runs visually (Streamlit)
make run-compare

# Promote a successful run to baseline
make run-promote RUN_ID=run_xxx \
  --as baseline PROMOTED_ID=baseline_v2 \
  REASON="New production baseline"

# List baselines and runs
make baselines-list
make runs-list
```

**Full guide:**
[Experiment Guide](EXPERIMENT_GUIDE.md) and
`data/eval/README.md`.

## Performance profiles (`data/profiles/`)

RFC-064 frozen performance snapshots per
provider/release. Each YAML captures per-stage
wall time, peak RSS, CPU usage, and environment
metadata for a specific provider configuration.

### Workflow

```bash
# Capture a profile from a pipeline run
make profile-freeze VERSION=v2.6-openai \
  PIPELINE_CONFIG=config/profiles/freeze/openai.yaml

# Compare two profiles
make profile-diff FROM=v2.6-wip-openai TO=v2.6-wip-gemini
```

Profiles live in `data/profiles/<version>.yaml`.
Pipeline capture configs live in
`config/profiles/freeze/*.yaml`.

**Full guide:**
[Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md)
and `data/profiles/README.md`.

## Validation gates: `ci-fast` vs `ci-ui-fast` vs `ci`

| Target | What it runs | When to use |
| ------ | ------------ | ----------- |
| `make ci-fast` | `cleanup-processes`, format-check, lint, type, security, complexity, docstrings, spelling, quality-metrics-ci, `test-fast` (critical-path unit + integration + E2E), `test-ui` (Vitest), `build-viewer` (`vue-tsc -b && vite build`), docs, build | **Default pre-commit gate.** ~6-10 min. Skips Playwright and coverage enforcement. |
| `make ci-ui-fast` | Same chain as `ci-fast` but `test-fast-no-py-e2e` (skips Python e2e) + `test-ui-e2e` (Playwright firefox) for viewer-iteration runs. Includes `build-viewer`. | **Viewer-heavy work.** ~8-12 min. |
| `make ci` | Everything in `ci-fast` + full `test` suite, `test-ui-e2e` (Playwright), `build-viewer`, coverage-enforce, **and** `stack-test-ml-ci` (full Docker stack + ml pipeline + Playwright + always-teardown) | **True full local parity** with the GitHub Actions chain (Python application → Stack test). ~20-30 min. Run before merge for changes that touch pipeline / Docker / route handlers. |
| `make test-fast` | Critical-path unit + integration + E2E tests only (no lint/format/type) | Quick test-only feedback. |
| `make build-viewer` | `cd web/gi-kg-viewer && npm install && npm run build` (`vue-tsc -b && vite build`). Catches strict TypeScript regressions invisible to `vitest` / `playwright`. | Standalone — already wired into every `ci*` target above. |
| `make stack-test-ml` | One-shot ml pipeline path: build → up → seed → Playwright (airgapped/whisper-tiny). Stack stays up after. | Local Docker validation, no API keys needed. |
| `make stack-test-cloud-thin` | Same flow with the `pipeline-llm` image + `cloud_thin` profile. **Local-only** — public CI does not run it (recurring API cost). Requires `.env` with the cloud API keys the profile uses. | Local LLM-path validation before push. |

The pre-commit hook runs staged checks (format,
lint, mypy, markdownlint, JSON/YAML validation) —
**not** `make ci-fast`. Run `make ci-fast` manually
before pushing.

## Optional pip extras reference

| Extra | Purpose | When to install |
| ----- | ------- | --------------- |
| `[dev]` | Tooling (pytest, black, flake8, mypy, etc.) | Always for development |
| `[ml]` | Local ML (Whisper, spaCy, torch, FAISS, etc.) | When using local models |
| `[llm]` | LLM API SDKs (openai, google-genai, anthropic, mistralai, httpx) | When using cloud providers |
| `[server]` | FastAPI + uvicorn | When running `serve` or working on viewer API |
| `[compare]` | Streamlit | When using `make run-compare` |

`make init` installs `[dev,ml,llm]`. Add `[server]`
or `[compare]` manually when needed.

## Markdown Linting

For detailed information about markdown linting, including automated fixing, table
formatting solutions, pre-commit hooks, and CI/CD integration, see the [Markdown Linting
Guide](MARKDOWN_LINTING_GUIDE.md).

**Quick reference:**

- **Before committing:** Run `make fix-md` to auto-fix common issues
- **Format on save:** Prettier is configured to format markdown files automatically
- **Pre-commit hook:** Automatically checks markdown files before commits
- **CI/CD:** All markdown files are linted in CI - errors will fail the build

**Lessons learned:** See the [Lessons Learned section](MARKDOWN_LINTING_GUIDE.md#lessons-learned-from-large-scale-cleanup)
in the Markdown Linting Guide for best practices from our large-scale cleanup effort
(fixed ~1,016 errors across 91 files).

## AI Coding Guidelines

This project includes comprehensive AI coding guidelines to ensure consistent code quality and workflow when using AI assistants.

### Overview

**Primary reference:** `.ai-coding-guidelines.md` - This is the PRIMARY source of truth for all AI actions in this project.

**Purpose:**

- Provides project-specific context and patterns for AI assistants
- Ensures consistent code quality and workflow
- Prevents common mistakes (auto-committing, skipping CI, etc.)

### Entry Points by AI Tool

Different AI assistants load guidelines from different locations:

| Tool               | Entry Point                       | Auto-Loaded           |
| ------------------ | --------------------------------- | --------------------- |
| **Cursor**         | `.cursor/rules/ai-guidelines.mdc` | Yes (modern format)   |
| **Claude Desktop** | `CLAUDE.md` (root directory)      | Yes                   |
| **GitHub Copilot** | `.github/copilot-instructions.md` | Yes                   |

**All entry points reference `.ai-coding-guidelines.md` as the primary source.**

### Critical Workflow Rules

**BRANCH CREATION CHECKLIST - MANDATORY BEFORE CREATING ANY BRANCH:**

**CRITICAL: Always check for uncommitted changes before creating a new branch.**

**Step 1: Check Current State**

```bash
git status
```

**What to look for:**

- If you see "Changes not staged for commit" → You have uncommitted changes
- If you see "Untracked files" → You have new files
- If you see "nothing to commit, working tree clean" → You're good to go!

**Step 2: Handle Uncommitted Changes (if any)**

**Option A: Commit to Current Branch** (if changes belong to current work)

```bash
git add .
git commit -m "your message"
```

**Option B: Stash for Later** (if you want to save but not commit)

```bash
git stash

# Later: git stash pop

```

**Option C: Discard Changes** (if not needed)

```bash
git checkout .

# Or for specific files:

git checkout -- path/to/file
```

**Quick One-Liner Check:**

```bash
git status --porcelain
```

**If you see any output, handle it first!**

**What happens if you don't follow this:**

- Uncommitted changes from previous work get included in your new branch
- Your commit will show more files than you actually changed
- PR will show confusing diffs with unrelated changes
- Harder to review and understand what actually changed

**Example: Clean Branch Creation**

```bash

# 1. Check status

$ git status
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean

# 2. Pull latest

$ git pull origin main
Already up to date.

# 3. Create branch

$ git checkout -b issue-117-output-organization
Switched to a new branch 'issue-117-output-organization'

# 4. Verify clean state

$ git status
On branch issue-117-output-organization
nothing to commit, working tree clean
```

**NEVER commit without:**

- Showing user what files changed (`git status`)
- Showing user the actual changes (`git diff`)
- Getting explicit user approval
- User deciding commit message

**NEVER push to PR without:**

- Running `make ci` locally first (full validation)
- Ensuring `make ci` passes completely
- Fixing all failures before pushing

**Note:** Use `make ci-fast` for quick feedback during development, but always run
`make ci` before pushing to ensure full validation.

## What's in `.ai-coding-guidelines.md`

**Sections include:**

- **Git Workflow** - Commit approval, PR workflow, branch naming
- **Code Organization** - Module boundaries, when to create new files
- **Testing Requirements** - Mocking patterns, test structure
- **Documentation Standards** - PRDs, RFCs, docstrings
- **Common Patterns** - Configuration, error handling, logging
- **Decision Trees** - When to create modules, PRDs, RFCs
- **When to Ask** - When AI should ask vs. act autonomously

### For Developers

**If you're using Cursor AI:**

- The guidelines are automatically loaded (no setup needed)
- AI assistants will follow project patterns and workflows
- Guidelines ensure consistent code quality
- **See also:** [`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`](CURSOR_AI_BEST_PRACTICES_GUIDE.md) -
  Best practices for using Cursor AI effectively, including model selection, workflow

  optimization, prompt templates, and project-specific recommendations

**If you're using other AI assistants:**

- The guidelines are automatically loaded (no setup needed)
- AI assistants will follow project patterns and workflows
- Guidelines ensure consistent code quality

**If you're not using an AI assistant:**

- You don't need to read these files
- They're for AI tools, not human developers
- Human contributors should follow [CONTRIBUTING.md](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)

### Maintenance

**When to update `.ai-coding-guidelines.md`:**

- New patterns or conventions are established
- Workflow changes (e.g., new CI checks)
- Architecture decisions that affect code organization
- New tools or processes are added

**Keep entry points in sync:**

- When updating `.ai-coding-guidelines.md`, ensure entry points (`CLAUDE.md`,
  `.github/copilot-instructions.md`, `.cursor/rules/ai-guidelines.mdc`) still reference

  it correctly

**See:** `.ai-coding-guidelines.md` for complete guidelines.

## Code Style Guidelines

### Formatting Tools

The project uses automated formatting and quality tools:

- **Black**: Code formatting (line length: 100 characters)
- **isort**: Import statement organization
- **flake8**: Linting and style enforcement
- **mypy**: Static type checking
- **radon**: Cyclomatic complexity analysis
- **vulture**: Dead code detection
- **interrogate**: Docstring coverage
- **codespell**: Spell checking

**Apply formatting automatically:**

```bash
make format
```

**Run all quality checks:**

```bash
make quality  # complexity, deadcode, docstrings, spelling
```

### Naming Conventions

**Functions and Variables:** Use `snake_case` with descriptive names.

```python

# Good

def fetch_rss_feed(url: str) -> RssFeed:
    episode_count = len(feed.episodes)

# Bad

def fetchRSSFeed(url: str):  # camelCase
    x = len(feed.episodes)  # non-descriptive name
```

**Classes:** Use `PascalCase` with descriptive nouns.

```python

# Good

class RssFeed:
    pass

# Bad

class rss_feed:  # snake_case
    pass
```

**Constants:** Use `UPPER_SNAKE_CASE`.

```python
DEFAULT_TIMEOUT = 20
MAX_RETRIES = 3
```

**Private Members:** Prefix with underscore.

```python
class SummaryModel:
    def __init__(self):
        self._device = "cpu"  # Internal attribute

    def _load_model(self):  # Internal method
        pass
```

## Type Hints

All functions should have type hints:

```python
def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safe filesystem use."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def run_pipeline(cfg: Config) -> None:
    """Run the complete podcast scraping pipeline.

    Args:
        cfg: Configuration object containing RSS URL and processing options.

    Raises:
        ValueError: If configuration is invalid.
        HTTPError: If RSS feed cannot be fetched.
    """
    pass
```

### Import Order

Follow this order (enforced by isort):

1. Standard library imports
2. Third-party imports
3. Local application imports

```python

# Standard library

import os
import sys
from pathlib import Path

# Third-party

import requests
from pydantic import BaseModel

# Local

from podcast_scraper import config
from podcast_scraper.models import Episode
```

## Every New Function Needs

**Unit test with mocks for external dependencies:**

```python
@patch("podcast_scraper.rss.downloader.requests.Session")
def test_fetch_url_with_retry(self, mock_session):
    """Test that fetch_url retries on network failure."""
    mock_session.get.side_effect = [
        requests.ConnectionError("Network error"),
        MockHTTPResponse(content="Success", status_code=200)
    ]
    result = fetch_url("https://example.com/feed.xml")
    self.assertEqual(result, "Success")
```

**Descriptive test names:**

```python

# Good

def test_sanitize_filename_removes_invalid_characters(self):
    pass

def test_whisper_model_selection_prefers_en_variant_for_english(self):
    pass

# Bad

def test_config(self):
    pass

def test_whisper(self):
    pass
```

**Also consider:**

- **Integration test** (marked `@pytest.mark.integration`)
- **Documentation update** (README, API docs, or relevant guide)
- **Examples** if user-facing

## Mock External Dependencies

Always mock external dependencies in tests:

- **HTTP requests**: Mock `requests` module (unit/integration tests), use E2E server for E2E tests
- **Whisper models**:
  - **Unit Tests**: Mock `whisper.load_model()` and `whisper.transcribe()` (all dependencies mocked)
  - **Integration Tests**: Mock Whisper for speed (focus on component integration)
  - **E2E Tests**: Use real Whisper models (NO mocks - complete workflow validation)
- **File I/O**: Use `tempfile.TemporaryDirectory` for isolated tests
- **spaCy models**:
  - **Unit Tests**: Mock NER extraction (all dependencies mocked)
  - **Integration Tests**: Mock spaCy for speed (focus on component integration)
  - **E2E Tests**: Use real spaCy models (NO mocks - complete workflow validation)
- **API providers**: Mock API clients (unit/integration tests), use E2E server mock endpoints (E2E tests)

**Provider Testing Patterns:**

- **Unit Tests**: Mock all provider dependencies (API clients, ML models)
- **Integration Tests**: Use real provider implementations with mocked external services
  (HTTP APIs) and mocked ML models (Whisper, spaCy, Transformers)

- **E2E Tests**: Use real providers with E2E server mock endpoints (for API providers)
  or real implementations (for local providers). ML models are REAL - no mocks allowed.

```python
import tempfile
from unittest.mock import patch, Mock

class TestEpisodeProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.providers.ml.whisper_utils.whisper")
    def test_transcription(self, mock_whisper):
        mock_whisper.load_model.return_value = Mock()
        mock_whisper.transcribe.return_value = {"text": "Test transcript"}
        # ... test code ...
```

For detailed mocking patterns, see:

- [Unit Testing Guide](UNIT_TESTING_GUIDE.md) - Unit test mocking patterns
- [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) - Integration test guidelines
- [E2E Testing Guide](E2E_TESTING_GUIDE.md) - E2E testing with real ML

**Network isolation**: All tests use `--disable-socket --allow-hosts=127.0.0.1,localhost`.

**E2E test modes** (`E2E_TEST_MODE` env var):

- `fast`: 1 episode (quick)
- `multi_episode`: 5 episodes (full validation)
- `data_quality`: All mock data (nightly)

**Flaky Test Reruns** (integration/E2E only):

```bash

# Automatic in make targets, or manually:

pytest --reruns 2 --reruns-delay 1
```

## When to Create PRD (Product Requirements Document)

Create a PRD for:

- New user-facing features
- Changes that affect user workflows

**Template:** `docs/prd/PRD-XXX-feature-name.md`

**Examples:**

- PRD-004: Metadata Generation
- PRD-005: Episode Summarization

## When to Create RFC (Request for Comments)

Create an RFC for:

- Architectural changes
- Breaking API changes
- Design decisions that need discussion
- Technical implementation approaches

**Template:** `docs/rfc/RFC-XXX-feature-name.md`

**Examples:**

- RFC-010: Speaker Name Detection
- RFC-012: Episode Summarization

### When to Skip PRD/RFC

You can proceed without PRD/RFC for:

- Bug fixes
- Small enhancements (< 100 lines of code)
- Internal refactoring that doesn't affect API
- Documentation-only updates
- Test improvements

### Always Update

**README** if:

- CLI flags change
- New features are user-facing
- Installation requirements change
- Usage examples need updates

**`docs/architecture/ARCHITECTURE.md`** if:

- Module responsibilities change
- New modules are added
- Data flow changes
- Design decisions are made

**`docs/architecture/TESTING_STRATEGY.md`** if:

- Testing approach changes
- New test categories are added
- Test infrastructure is updated

**API docs** if:

- Public API changes (functions, classes, parameters)
- New public modules are added
- API contracts change

### Before Pushing Documentation Changes

**Always check `mkdocs.yml` and verify all links when adding, moving, or deleting documentation files:**

- [ ] **New files added?** → Add to `nav` configuration in `mkdocs.yml`
- [ ] **Files moved?** → Update path in `nav` configuration
- [ ] **Files deleted?** → Remove from `nav` configuration
- [ ] **Links updated?** → Use relative paths (e.g., `rfc/RFC-019.md` not `docs/rfc/RFC-019.md`)
- [ ] **All links verified?** → Check that all internal links point to existing files
- [ ] **No broken links?** → Run `make docs` to catch broken links before CI
- [ ] **Test locally?** → Run `make docs` to verify build succeeds

**Common issues:**

- Missing files in `nav` → Build will warn about pages not in nav
- Broken links → Build will fail if links point to non-existent files
- Wrong path format → Use relative paths from `docs/` directory

**Why this matters:**

- Broken links waste CI build time (~3-5 min per failed build)
- Fixing locally with `make docs` takes seconds vs. waiting for CI
- Prevents unnecessary CI failures and re-runs

**Example:** When adding a new RFC:

```yaml

# mkdocs.yml

nav:

  - RFCs:
      - RFC-023 README Acceptance Tests: rfc/RFC-023-readme-acceptance-tests.md
```

## CI/CD Integration

> **See also:** [CI/CD Documentation](../ci/index.md) for complete CI/CD pipeline documentation with visualizations.

### What Runs in CI

The GitHub Actions workflows use **intelligent path-based filtering** to run only when necessary. This means:

- **Documentation-only changes:** Only the docs workflow runs (~3-5 min)
- **Python code changes:** All workflows run for full validation (~15-20 min)
- **README changes:** Only the docs workflow runs (~3-5 min)

**Python Application Workflow** (4 parallel jobs) - **Runs only when Python/config files change:**

1. **Lint Job** (2-3 min, no ML deps):
   - Black/isort formatting checks
   - Flake8 linting
   - Markdownlint for docs
   - Mypy type checking
   - Bandit + pip-audit security scanning
   - Code quality analysis (complexity, dead code, docstrings, spelling)

2. **Test Job** (10-15 min, full ML stack):
   - Full pytest suite with coverage
   - Integration tests (mocked)

3. **Docs Job** (3-5 min):
   - MkDocs build (strict mode)
   - API documentation generation

4. **Build Job** (2-3 min):
   - Build source distribution
   - Build wheel distribution

**Documentation Deployment** (sequential) - **Runs when docs or Python files change:**

- Build MkDocs site
- Deploy to GitHub Pages (on push to main)

**CodeQL Security** (parallel language analysis) - **Runs only when code/workflow files change:**

- Python security scanning
- GitHub Actions security scanning

### Path-Based CI Optimization

Workflows are configured to skip when irrelevant files change:

| Files Changed | Python App | Docs | CodeQL | Time Savings |
| ------------- | ---------- | ---- | ------ | ------------ |
| Only `docs/` | Skip | Run | Skip | ~18 minutes |
| Only `.py` | Run | Run | Run | - |
| Only `README.md` | Skip | Run | Skip | ~18 minutes |
| `pyproject.toml` | Run | Skip | Skip | ~5 minutes |
| `docker/pipeline/Dockerfile` | Run | Skip | Skip | ~5 minutes |

This optimization provides fast feedback for documentation updates while maintaining full validation for code changes.

### CI Failure Response

If CI fails on your PR:

1. **Check the CI logs** to identify the failure
2. **Reproduce locally:** Run `make ci` to see the same failure
3. **Fix the issue** and test locally
4. **Push the fix** - CI will re-run automatically

**CI Command Differences:**

- **`make ci`**: Full CI suite
  - Runs `test` (unit + integration + e2e tests, excludes slow/ml_models)
  - Full validation matching GitHub Actions
  - Use before commits/PRs

- **`make ci-fast`**: Fast CI checks
  - Runs `test-fast` (unit + critical path integration + critical path e2e, no coverage)
  - Skips `coverage-enforce` (the main difference from `make ci`)
  - Quick feedback during development
  - Use for rapid iteration, but always run `make ci` before pushing

- **`make ci-clean`**: Complete CI suite (clean start)
  - Runs `clean-all format-check lint lint-markdown type security preload-ml-models test docs build`
  - Starts fresh by removing build artifacts + ML caches, then runs the full validation pipeline
  - Use before releases or when you need full test coverage from a clean state

**Common failures:**

| Issue | Solution |
| ----- | -------- |
| Formatting issues | Run `make format` to auto-fix |
| Linting errors | Fix code style issues or run `make format` |
| Type errors | Add missing type hints |
| Test failures | Fix or update tests |
| Coverage drop | Add tests for new code |
| Markdown linting | Run `make fix-md` to auto-fix markdown issues |

**Process safety (RFC-074):**

`cleanup-processes` runs automatically before every `ci` and `ci-fast`
invocation. If you see stuck Python processes after a failed `make`
run, use `make cleanup-processes` manually. If `make check-zombie`
reports UE-state processes, reboot is required. See
[CI Local Development -- Process Safety](../ci/LOCAL_DEVELOPMENT.md).

**Prevent failures with pre-commit hooks:**

```bash

# Install once

make install-hooks

# Now linting failures are caught before commit!
```

### Release checklist

**Standing plan (policy, eval/perf expectations, doc validation order):** see
[Release Playbook](RELEASE_PLAYBOOK.md).

Use this checklist before tagging a release (e.g. v2.6.0). For the full gate, run **`make pre-release`**
(see [ADR-031](../adr/ADR-031-mandatory-pre-release-validation.md) and [Release Playbook](RELEASE_PLAYBOOK.md) Phase 3); then complete tagging and GitHub Release below.

#### 1. Pre-flight

- **Branch & tree**: Work from `main` (or your release branch). Ensure a clean working tree: `git status --porcelain` should be empty, or only include files you intend to commit for the release.
- **Version**: Decide the release version using [Semantic Versioning](https://semver.org/) (see [Releases index](../releases/index.md)): major (breaking), minor (new features), patch (fixes).

#### 2. Version bump

- **`pyproject.toml`**: Set `version = "X.Y.Z"` in the `[project]` section.
- **`src/podcast_scraper/__init__.py`**: Set `__version__ = "X.Y.Z"` so the package and CLI report the same version. Keep both in sync.

#### 3. Release docs prep

**Why this matters:** Architecture diagrams are not generated in CI. The docs site and all CI jobs use the committed `docs/architecture/diagrams/*.svg` files. If you release without updating them, the published docs will show outdated architecture, and subsequent PRs may fail checks until you run `make visualize` and commit updated SVGs. Running release-docs-prep before every release keeps diagrams in sync with the code you are releasing.

- Run **`make release-docs-prep`**. This:
  - Regenerates architecture diagrams (`docs/architecture/diagrams/*.svg`).
  - Creates a draft `docs/releases/RELEASE_vX.Y.Z.md` for the current version (from `pyproject.toml`) if it does not exist.
- Review and commit:
  - `git add docs/architecture/diagrams/*.svg docs/releases/RELEASE_*.md`
  - `git commit -m "docs: release docs prep (visualizations and release notes)"`

#### 4. Release notes

- Edit **`docs/releases/RELEASE_vX.Y.Z.md`**: fill in Summary, Key Features, Upgrade Notes (if any), and Full Changelog link (e.g. `https://github.com/chipi/podcast_scraper/compare/vPREVIOUS...vX.Y.Z`).
- Update **`docs/releases/index.md`**: add the new version to the table and update the "Latest Release" section (remove any "upcoming" wording once the version is tagged and published).

#### 5. Quality and validation

Run all of the following in each release cycle before releasing so the codebase meets project standards.

- **Format & lint**: `make format` then `make lint` and `make type`. Fix any issues.
- **Markdown**: `make fix-md` (or `make lint-markdown`) so docs and markdown pass.
- **Docs build**: `make docs` (MkDocs build must succeed).
- **Code hygiene**: Run **`make quality`** (complexity, dead code, docstrings, spelling). Resolve or document any findings so that:
  - Complexity (radon) and maintainability index are acceptable or exceptions documented.
  - Docstring coverage meets the configured `fail-under` (see `[tool.interrogate]` in `pyproject.toml`).
  - Dead code (vulture) and spelling (codespell) findings are triaged (fixed or whitelisted/ignored).
  - Test coverage meets the combined threshold (see [Issue #432](https://github.com/chipi/podcast_scraper/issues/432) for background and targets).
- **Tests**: Run the full CI gate: **`make ci`** (format-check, lint, type, security, complexity, docstrings, spelling, tests, coverage-enforce, docs, build). For maximum confidence (e.g. major release), run **`make ci-clean`** or run **`make test`** then **`make coverage-enforce`**, **`make docs`**, **`make build`**.
- **Diagrams (required for release):** If diagrams are stale, run `make visualize` and commit `docs/architecture/diagrams/*.svg`. Before release, **`make release-docs-prep`** regenerates diagrams and drafts release notes—do not skip it or the published docs site will ship with outdated architecture.
- **Build**: Ensure **`make build`** succeeds (sdist/wheel in `.build/dist/` or `dist/`).

#### 6. Commit and push

- Commit all release changes (version bumps, release notes, index, diagram updates) with a clear message, e.g. `chore: release vX.Y.Z`.
- Push the branch: `git push origin <branch>` (never push to `main` without a reviewed PR unless your workflow allows it).

#### 7. Tag and GitHub release

- Create an annotated tag: **`git tag -a vX.Y.Z -m "Release vX.Y.Z"`** (use the same version as in `pyproject.toml` and `__init__.py`).
- Push the tag: **`git push origin vX.Y.Z`**.
- On GitHub: open **Releases** → **Draft a new release**, choose tag `vX.Y.Z`, paste the contents of `docs/releases/RELEASE_vX.Y.Z.md` as the release description, and publish.

#### 8. Post-release (optional)

- If you use a "next dev" version, bump to it (e.g. `X.Y.(Z+1)` or `X.Y.Z-dev`) in `pyproject.toml` and `__init__.py` and commit so the next build is not stuck on the release version.

**See also:** [ADR-031: Mandatory Pre-Release Validation](../adr/ADR-031-mandatory-pre-release-validation.md), [Architecture visualizations](../architecture/diagrams/README.md), [Releases index](../releases/index.md).

## Modularity

- **Single Responsibility:** Each module should have one clear purpose
- **Loose Coupling:** Modules should depend on abstractions, not concrete implementations
- **High Cohesion:** Related functionality should be grouped together

### Configuration

**All runtime options flow through the `Config` model:**

```python
from podcast_scraper import Config

# Good - centralized configuration

cfg = Config(
    rss="https://example.com/feed.xml",
    output_dir="./output",
    transcribe_missing=True
)
run_pipeline(cfg)

# Bad - scattered configuration

fetch_rss(url, timeout=30)
download_transcripts(episodes, workers=8)
transcribe_missing(jobs, model="base")
```

**Adding new configuration options:**

1. Add to `Config` model in `config.py`
2. Add CLI argument in `cli.py`
3. Document in README options section
4. Update config examples in `config/examples/`

## Error Handling

**Follow these patterns:**

```python

# Recoverable errors - log warnings, continue

try:
    transcript = download_transcript(url)
except requests.RequestException as e:
    logger.warning(f"Failed to download transcript: {e}")
    return None

# Unrecoverable errors - raise specific exceptions

if not cfg.rss:
    raise ValueError("RSS URL is required")

# Validation errors - use ValueError with clear message

if cfg.workers < 1:
    raise ValueError(f"Workers must be >= 1, got: {cfg.workers}")

# Graceful degradation for optional features

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available, transcription disabled")
```

## CLI exit codes (Issue #429)

The main pipeline command uses the following exit code policy:

- **0** – Run completed. The pipeline ran to the end (config valid, no run-level exception). Some episodes may have failed; partial results and run index still reflect failures.
- **1** – Run-level failure. Configuration error, dependency missing (e.g. ffmpeg), or an unhandled exception during the run.

So exit code 0 means "the run finished", not "every episode succeeded". Use the run index (`index.json`) or `run.json` to see per-episode status. Flags `--fail-fast` and `--max-failures` stop processing after the first or after N episode failures but **still exit 0** if the run completed without a run-level error.

## CLI subcommands and startup (Issue #429)

- **Subcommands:** The first positional argument
  selects a subcommand. When omitted, the CLI runs
  the default pipeline. All invocations use
  `python -m podcast_scraper.cli <subcommand>`.

| Subcommand | Purpose |
| ---------- | ------- |
| *(default)* | Run the transcript pipeline for one or more RSS feeds |
| `doctor` | Environment checks (Python, ffmpeg, permissions, models) |
| `cache` | ML model cache management (`--status`, `--clean`) |
| `serve` | Start the FastAPI viewer server (`--output-dir`) |
| `search` | Semantic search over a corpus index |
| `index` | Build or rebuild the FAISS vector index |
| `gi` | GIL subcommands (`inspect`, `show-insight`, `explore`) |
| `kg` | KG subcommands (`inspect`) |
| `corpus-status` | Show multi-feed corpus status |
| `pricing-assumptions` | Display pricing model assumptions |

- **Startup validation:** Before the main pipeline
  runs, the CLI checks Python version (3.10+) and
  that `ffmpeg` is on PATH. These checks are
  **skipped** for utility subcommands (`doctor`,
  `cache`, `serve`, `search`, `index`, `gi`, `kg`,
  `corpus-status`, `pricing-assumptions`).
- **Full CLI reference:**
  [CLI.md](../api/CLI.md).
- **Live pipeline monitor (RFC-065):** On the default pipeline command, **`--monitor`**
  spawns a subprocess with a live RSS/CPU/stage dashboard (or appends **`.monitor.log`** when
  the monitor’s stderr is not a TTY) and writes **`.pipeline_status.json`** under the output
  directory. Optional **`.[monitor]`**: **`--memray`** / **`memray:`** for heap captures; with
  monitor + TTY, **`f`** in the parent triggers **py-spy** to **`debug/flamegraph_*.svg`**. See
  [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md).

## Log Level Guidelines

**Use `logger.info()` for:**

- High-level operations that users care about
- Important state changes and milestones
- User-facing progress updates
- Important results (e.g., "Summary generated", "saved transcript")
- Episode processing start/completion
- Major pipeline stages (e.g., "Starting Whisper transcription", "Processing summarization")

**Use `logger.debug()` for:**

- Detailed internal operations
- Model loading/unloading details
- Configuration details and parameter values
- Per-item processing details
- Technical implementation details
- Validation metrics and statistics
- Chunking, mapping, and reduction details
- File handle management and cleanup
- Fallback attempts and retries

**Use `logger.warning()` for:**

- Recoverable errors
- Degraded functionality
- Missing optional dependencies
- Non-critical failures

**Use `logger.error()` for:**

- Unrecoverable errors
- Critical failures
- Validation failures

### Examples

```python

# Good - INFO for high-level operation

logger.info("Processing summarization for %d episodes in parallel", len(episodes))

# Good - DEBUG for detailed technical info

logger.debug("Pre-loading %d model instances for thread safety", max_workers)

# Good - INFO for important results

logger.info("Summary generated in %.1fs (length: %d chars)", elapsed, len(summary))

# Bad - INFO for technical details (should be DEBUG)

logger.info("Loading summarization model: %s on %s", model_name, device)
```

**Module-Specific Guidelines:**

- **Workflow:** INFO for episode counts, major stages; DEBUG for cleanup
- **Summarization:** INFO for generation start/completion; DEBUG for model loading
- **Whisper:** INFO for "transcribing with Whisper"; DEBUG for model loading
- **Episode Processing:** INFO for file saves; DEBUG for download details
- **Speaker Detection:** INFO for results; DEBUG for model download

## Rationale

This approach ensures:

- **Service/daemon logs** remain focused and readable
- **Production monitoring** shows high-level progress without noise
- **Debugging** still has access to detailed information when needed
- **Log file sizes** stay manageable during long runs

When in doubt, prefer DEBUG over INFO - it's easier to promote a log level than to demote it.

### Progress Reporting

**Use the `progress.py` abstraction:**

```python
from podcast_scraper.utils.progress import progress_context

# Good - uses progress abstraction

with progress_context(
    total=len(episodes),
    description="Downloading transcripts"
) as reporter:
    for episode in episodes:
        process_episode(episode)
        reporter.update(1)

# Bad - direct tqdm usage

from tqdm import tqdm
for episode in tqdm(episodes):
    process_episode(episode)
```

## Lazy Loading Pattern

**For optional dependencies:**

```python

# At module level

_whisper = None

def load_whisper():
    """Lazy load Whisper library."""
    global _whisper
    if _whisper is None:
        try:
            import whisper
            _whisper = whisper
        except ImportError:
            raise ImportError(
                "Whisper not installed. "
                "Install with: pip install openai-whisper"
            )
    return _whisper
```

## Module Responsibilities

The full module map with dependency diagrams is in
[Architecture](../architecture/ARCHITECTURE.md).
Detailed boundaries are in
`.cursor/rules/module-boundaries.mdc`. Below is a
compact package-level overview.

### Public API and entry points

- **`cli.py`** — CLI only, no business logic
- **`service.py`** — Service API, structured results
  for daemon use
- **`config.py`** — Configuration models and validation

### Pipeline and workflow

- **`workflow/orchestration.py`** — Orchestration only,
  no HTTP/IO details
- **`workflow/stages/`** — Stage modules (setup,
  scraping, processing, transcription, metadata,
  summarization)
- **`workflow/episode_processor.py`** — Episode-level
  processing logic
- **`workflow/corpus_operations.py`** — Multi-feed
  manifest and summary artifacts
- **`workflow/append_resume.py`** — Append/resume logic
- **`workflow/degradation.py`** — Graceful degradation
  for non-critical stages
- **`workflow/run_manifest.py`** /
  **`workflow/run_summary.py`** — Run tracking
- **`workflow/jsonl_emitter.py`** — Streaming metrics
- **`workflow/metadata_generation.py`** — Metadata
  document generation

### RSS and downloads

- **`rss/parser.py`** — RSS parsing, episode creation
- **`rss/downloader.py`** — HTTP operations only
- **`rss/feed_cache.py`** — Optional on-disk RSS cache

### Providers (9 total)

- **`providers/ml/`** — Local ML (`MLProvider`,
  `HybridMLProvider`, Whisper, spaCy, summarizer,
  model registry)
- **`providers/{openai,gemini,anthropic,mistral,deepseek,grok,ollama}/`** — LLM provider packages
- **`providers/capabilities.py`** — Capability flags
- **`transcription/`**, **`speaker_detectors/`**,
  **`summarization/`** — Protocol interfaces and
  factory functions
- **`prompts/store.py`** — Versioned Jinja2 prompt
  templates

### Knowledge extraction

- **`gi/`** — Grounded Insight Layer (pipeline,
  schema, grounding, explore, corpus, quality metrics)
- **`kg/`** — Knowledge Graph (pipeline, schema,
  LLM extraction, CLI handlers, quality metrics)

### Search

- **`search/`** — FAISS vector indexing, transcript
  chunking, corpus search/similarity, protocols,
  CLI handlers

### Server (FastAPI viewer API)

- **`server/app.py`** — App factory, CORS, static
  mounting
- **`server/routes/`** — 10 route modules (health,
  artifacts, search, explore, index_stats,
  index_rebuild, corpus_library, corpus_binary,
  corpus_metrics, corpus_digest)
- **`server/corpus_catalog.py`** — Filesystem-backed
  episode catalog
- **`server/corpus_digest.py`** — Digest selection
- **`server/index_rebuild.py`** /
  **`server/index_staleness.py`** — Background FAISS
  rebuild and freshness
- **`server/pathutil.py`** — Safe corpus path
  resolution

### Support

- **`models/`** — Shared data models (RssFeed,
  Episode, TranscriptionJob)
- **`schemas/`** — Summary schema validation
- **`cache/`** — Cache directories and management
- **`cleaning/`** — Transcript cleaning (pattern,
  LLM, hybrid)
- **`evaluation/`** — Experiment config, scorers,
  regression, fingerprinting
- **`preprocessing/`** — Audio preprocessing (FFmpeg,
  Opus, VAD)
- **`utils/`** — Filesystem, progress, timeouts,
  retries, redaction, corpus paths, provider metrics

**Keep concerns separated** — don't mix HTTP calls in
CLI, don't put business logic in config, etc.

## When to Create New Files

**Create new modules when:**

- Implementing a new major feature (e.g., new provider implementation)
- A module has distinct responsibility following Single Responsibility Principle
- An existing module exceeds ~1000 lines and can be logically split

**Modify existing files when:**

- Fixing bugs
- Enhancing existing functionality
- Refactoring within the same module

### Provider Implementation Patterns

The project uses a **protocol-based provider system** for transcription, speaker detection,
and summarization. When implementing new providers:

1. **Understand the Protocol**: Read the protocol definition in `{capability}/base.py`
2. **Implement Provider Class**: Create `{capability}/{provider}_provider.py`
3. **Register in Factory**: Update `{capability}/factory.py` to include new provider
4. **Add Configuration**: Update `config.py` to support provider selection
5. **Add CLI Support**: Update `cli.py` with provider arguments (if needed)
6. **Add E2E Server Mocking**: For API providers, add mock endpoints
7. **Write Tests**: Create unit, integration, and E2E tests

**For complete implementation guide**, see [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md).

**Choosing a provider:**
[AI Provider Comparison](AI_PROVIDER_COMPARISON_GUIDE.md)
(decision-oriented: cost, quality, speed, privacy) and
[Provider Deep Dives](PROVIDER_DEEP_DIVES.md)
(per-provider reference cards, benchmarks, magic
quadrant).

**Validating provider quality:** Run experiments
against `data/eval/` baselines and capture
performance profiles in `data/profiles/`. See
[Experiment Guide](EXPERIMENT_GUIDE.md) and
[Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md).

## Third-Party Dependencies

For detailed information about third-party dependencies, see the
[Dependencies Guide](DEPENDENCIES_GUIDE.md).

## Summarization Implementation

For detailed information about the summarization system, see the
[ML Provider Reference](ML_PROVIDER_REFERENCE.md).
