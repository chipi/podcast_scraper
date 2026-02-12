# Development Guide

> **Maintenance Note**: This document should be kept up-to-date as linting rules, Makefile
> targets, pre-commit hooks, CI/CD workflows, or development setup procedures evolve. When
> adding new checks, tools, workflows, or environment setup steps, update this document
> accordingly.

This guide provides detailed implementation instructions for developing the podcast scraper.
For high-level architectural decisions and design principles, see [Architecture](../ARCHITECTURE.md).

## Testing

For comprehensive testing information, see the dedicated testing documentation:

- **[Testing Strategy](../TESTING_STRATEGY.md)** - Testing philosophy, test pyramid, decision criteria
- **[Testing Guide](TESTING_GUIDE.md)** - Quick reference, test execution commands
- **[Experiment Guide](EXPERIMENT_GUIDE.md)** — Complete guide: datasets, baselines, experiments, and evaluation
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

### Running Tests

```bash
make check-unit-imports        # Verify modules can import without ML dependencies
make deps-analyze              # Analyze module dependencies (with report)
make deps-check                # Check dependencies (exits on error)
make analyze-test-memory       # Analyze test memory usage (default: test-unit)
make test-unit                 # Unit tests (parallel)
make test-integration          # Integration tests (parallel, reruns)
make test-e2e                  # E2E tests (serial first, then parallel)
make test                      # All tests
make test-fast                 # Unit + critical path only
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

1. **Use lazy imports**: Import inside functions, not at module level

1. **Verify imports work without ML deps**: Run `make check-unit-imports` before pushing
   - This verifies modules can be imported without ML dependencies installed
   - Runs automatically in CI before unit tests
   - Use when: adding new modules, refactoring imports, or debugging CI failures

1. **Run unit tests**: Run `make test-unit` before pushing

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

**See also:** [Module Dependency Analysis](../ARCHITECTURE.md#module-dependency-analysis) for detailed documentation.

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
  CONFIG=config/experiments/my_experiment.yaml \
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

- Running CLI commands: `python3 -m podcast_scraper.cli`
- Importing the package in Python: `from podcast_scraper import ...`
- Running tests that import the package

**Manual setup (if not using setup_venv.sh):**

If you create a virtual environment manually, you **must** install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,ml]  # Install package in editable mode with dev and ML dependencies
```

**Why editable mode (`-e`)?**

- Changes to source code are immediately available without reinstalling
- Required for development workflow
- Allows `python3 -m podcast_scraper.cli` to work

### Updating Virtual Environment Dependencies

**⚠️ CRITICAL: Update venv when dependency ranges change**

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

- ✅ Local: `make format-check` passes
- ❌ CI: `make format-check` fails with "would reformat"
- ✅ Local: `make lint` passes
- ❌ CI: `make lint` fails with different errors
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

- ✅ `.env` is in `.gitignore` (never committed)
- ✅ `config/examples/.env.example` is safe to commit (template only)
- ✅ API keys are never logged or exposed
- ✅ Environment variables take precedence over `.env` file
- ✅ HuggingFace model loading uses `trust_remote_code=False`; only enable `trust_remote_code=True` if a model's documentation explicitly requires it and the source is trusted (Issue #429).

**Priority order** (for each configuration field):

1. **Config file field** (highest priority) - if the field is set in the config file and not `null`/empty, it takes precedence
2. **Environment variable** - only used if the config file field is `null`, not set, or empty
3. **Default value** - used if neither config file nor environment variable is set

**Exception**: `LOG_LEVEL` environment variable takes precedence over config file (allows easy runtime log level control).

**Note**: You can define the same field in both the config file and as an environment variable.
The config file value will be used if it's set.
This allows config files for project defaults and environment variables for deployment-specific overrides.

**See also:**

- `docs/api/CONFIGURATION.md` - Configuration API reference (includes environment variables)
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

| Tool               | Entry Point                       | Auto-Loaded            |
| ------------------ | --------------------------------- | ---------------------- |
| **Cursor**         | `.cursor/rules/ai-guidelines.mdc` | ✅ Yes (modern format) |
| **Claude Desktop** | `CLAUDE.md` (root directory)      | ✅ Yes                 |
| **GitHub Copilot** | `.github/copilot-instructions.md` | ✅ Yes                 |

**All entry points reference `.ai-coding-guidelines.md` as the primary source.**

### Critical Workflow Rules

**🚨 BRANCH CREATION CHECKLIST - MANDATORY BEFORE CREATING ANY BRANCH:**

**CRITICAL: Always check for uncommitted changes before creating a new branch.**

**Step 1: Check Current State**

```bash
git status
```

**What to look for:**

- ❌ If you see "Changes not staged for commit" → You have uncommitted changes
- ❌ If you see "Untracked files" → You have new files
- ✅ If you see "nothing to commit, working tree clean" → You're good to go!

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
```python

**If you see any output, handle it first!**

**What happens if you don't follow this:**
- ❌ Uncommitted changes from previous work get included in your new branch
- ❌ Your commit will show more files than you actually changed
- ❌ PR will show confusing diffs with unrelated changes
- ❌ Harder to review and understand what actually changed

**Example: Clean Branch Creation**

```bash

# 1. Check status

$ git status
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean  ✅

# 2. Pull latest

$ git pull origin main
Already up to date.

# 3. Create branch

$ git checkout -b issue-117-output-organization
Switched to a new branch 'issue-117-output-organization'

# 4. Verify clean state

$ git status
On branch issue-117-output-organization
nothing to commit, working tree clean  ✅
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
- **Prompt templates available:** `.cursor/prompts/` contains reusable prompt templates for
  CI debugging, RFC design, code reviews, and implementation planning

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

✅ **Unit test with mocks for external dependencies:**

```python
@patch("podcast_scraper.downloader.requests.Session")
def test_fetch_url_with_retry(self, mock_session):
    """Test that fetch_url retries on network failure."""
    mock_session.get.side_effect = [
        requests.ConnectionError("Network error"),
        MockHTTPResponse(content="Success", status_code=200)
    ]
    result = fetch_url("https://example.com/feed.xml")
    self.assertEqual(result, "Success")
```

✅ **Descriptive test names:**

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

    @patch("podcast_scraper.whisper_integration.whisper")
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

**`docs/ARCHITECTURE.md`** if:

- Module responsibilities change
- New modules are added
- Data flow changes
- Design decisions are made

**`TESTING_STRATEGY.md`** if:

- Testing approach changes
- New test categories are added
- Test infrastructure is updated

**API docs** if:

- Public API changes (functions, classes, parameters)
- New public modules are added
- API contracts change

### ⚠️ Before Pushing Documentation Changes

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
| Only `docs/` | ❌ Skip | ✅ Run | ❌ Skip | ~18 minutes |
| Only `.py` | ✅ Run | ✅ Run | ✅ Run | - |
| Only `README.md` | ❌ Skip | ✅ Run | ❌ Skip | ~18 minutes |
| `pyproject.toml` | ✅ Run | ❌ Skip | ❌ Skip | ~5 minutes |
| `Dockerfile` | ✅ Run | ❌ Skip | ❌ Skip | ~5 minutes |

This optimization provides fast feedback for documentation updates while maintaining full validation for code changes.

### CI Failure Response

If CI fails on your PR:

1. **Check the CI logs** to identify the failure
2. **Reproduce locally:** Run `make ci` to see the same failure
3. **Fix the issue** and test locally
4. **Push the fix** - CI will re-run automatically

**CI Command Differences:**

- **`make ci`**: Full CI suite
  - Runs `test-ci` (unit + integration + e2e-fast tests, excludes slow/ml_models)
  - Full validation matching GitHub Actions
  - Use before commits/PRs

- **`make ci-fast`**: Fast CI checks
  - Skips cleanup step (faster startup)
  - Runs `test-ci-fast` (unit + fast integration + fast e2e, excludes slow/ml_models, no coverage)
  - Quick feedback during development
  - Use for rapid iteration, but always run `make ci` before pushing

- **`make ci-full`**: Complete CI suite
  - Runs `clean-all` first (removes build artifacts + ML caches)
  - Runs `test` (all tests: unit + integration + e2e, all slow/fast variants)
  - Complete validation including slow/ml_models tests
  - Use before releases or when you need full test coverage

**Common failures:**

| Issue | Solution |
| ----- | -------- |
| Formatting issues | Run `make format` to auto-fix |
| Linting errors | Fix code style issues or run `make format` |
| Type errors | Add missing type hints |
| Test failures | Fix or update tests |
| Coverage drop | Add tests for new code |
| Markdown linting | Run `python scripts/tools/fix_markdown.py` or `markdownlint --fix` |

**Prevent failures with pre-commit hooks:**

```bash

# Install once

make install-hooks

# Now linting failures are caught before commit!
```

### Release checklist

Use this checklist before tagging a release (e.g. v2.6.0). Until `make pre-release` exists (see [ADR-041](../adr/ADR-041-mandatory-pre-release-validation.md)), follow these steps manually.

#### 1. Pre-flight

- **Branch & tree**: Work from `main` (or your release branch). Ensure a clean working tree: `git status --porcelain` should be empty, or only include files you intend to commit for the release.
- **Version**: Decide the release version using [Semantic Versioning](https://semver.org/) (see [Releases index](../releases/index.md)): major (breaking), minor (new features), patch (fixes).

#### 2. Version bump

- **`pyproject.toml`**: Set `version = "X.Y.Z"` in the `[project]` section.
- **`src/podcast_scraper/__init__.py`**: Set `__version__ = "X.Y.Z"` so the package and CLI report the same version. Keep both in sync.

#### 3. Release docs prep

- Run **`make release-docs-prep`**. This:
  - Regenerates architecture diagrams (`docs/architecture/*.svg`).
  - Creates a draft `docs/releases/RELEASE_vX.Y.Z.md` for the current version (from `pyproject.toml`) if it does not exist.
- Review and commit:
  - `git add docs/architecture/*.svg docs/releases/RELEASE_*.md`
  - `git commit -m "docs: release docs prep (visualizations and release notes)"`

#### 4. Release notes

- Edit **`docs/releases/RELEASE_vX.Y.Z.md`**: fill in Summary, Key Features, Upgrade Notes (if any), and Full Changelog link (e.g. `https://github.com/chipi/podcast_scraper/compare/vPREVIOUS...vX.Y.Z`).
- Update **`docs/releases/index.md`**: add the new version to the table and, if appropriate, update the "Latest Release" / "Upcoming" section.

#### 5. Quality and validation

- **Format & lint**: `make format` then `make lint` and `make type`. Fix any issues.
- **Markdown**: `make fix-md` (or `make lint-markdown`) so docs and markdown pass.
- **Docs build**: `make docs` (MkDocs build must succeed).
- **Tests**: Run the full CI gate: **`make ci`** (format-check, lint, type, security, complexity, docstrings, spelling, tests, coverage-enforce, docs, build). For maximum confidence (e.g. major release), run **`make ci-clean`** or run **`make test`** then **`make coverage-enforce`**, **`make docs`**, **`make build`**.
- **Diagrams**: `make check-visualizations` (optional; already covered by `release-docs-prep`).
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

**See also:** [ADR-041: Mandatory Pre-Release Validation](../adr/ADR-041-mandatory-pre-release-validation.md), [Architecture visualizations](../architecture/README.md), [Releases index](../releases/index.md).

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

- **Subcommands:** The first argument can be `doctor` or `cache`. When you run `python -m podcast_scraper.cli doctor` (or `podcast-scraper doctor`), the rest of the arguments are passed to that subcommand. If you omit arguments, the CLI uses `sys.argv[1:]` so subcommands work when invoked from the shell.
- **Startup validation:** Before the main pipeline runs, the CLI checks Python version (3.10+) and that `ffmpeg` is on PATH. These checks are **skipped** for `doctor` and `cache` so you can run doctor even if ffmpeg is missing.
- **Doctor** (`podcast-scraper doctor`): Runs environment checks (Python, ffmpeg, write permissions, model cache, ML imports). Use `--check-network` to test connectivity and `--check-models` to load default Whisper and summarizer once (slow). See [Troubleshooting - Doctor command](TROUBLESHOOTING.md#doctor-command-issue-379-429).
- **Cache** (`podcast-scraper cache --status` / `--clean`): Manages ML model caches. No Python/ffmpeg validation.

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
from podcast_scraper import progress

# Good - uses progress abstraction

with progress.make_progress(
    total=len(episodes),
    desc="Downloading transcripts"
) as pbar:
    for episode in episodes:
        process_episode(episode)
        pbar.update(1)

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

- **`cli.py`**: CLI only, no business logic
- **`service.py`**: Service API, structured results for daemon use
- **`workflow.orchestration`**: Orchestration only, no HTTP/IO details
- **`config.py`**: Configuration models and validation
- **`rss.downloader`**: HTTP operations only
- **`utils.filesystem`**: File system utilities only
- **`rss.parser`**: RSS parsing, episode creation
- **`workflow.episode_processor`**: Episode-level processing logic
- **`providers.ml.whisper_utils`**: Whisper transcription utilities
- **`providers.ml.speaker_detection`**: NER-based speaker extraction
- **`providers.ml.summarizer`**: Transcript summarization
- **`workflow.metadata_generation`**: Metadata document generation
- **`utils.progress`**: Progress reporting abstraction
- **`models.py`**: Shared data models

**Keep concerns separated** - don't mix HTTP calls in CLI, don't put business logic in config, etc.

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
1. **Implement Provider Class**: Create `{capability}/{provider}_provider.py`
1. **Register in Factory**: Update `{capability}/factory.py` to include new provider
1. **Add Configuration**: Update `config.py` to support provider selection
1. **Add CLI Support**: Update `cli.py` with provider arguments (if needed)
1. **Add E2E Server Mocking**: For API providers, add mock endpoints
1. **Write Tests**: Create unit, integration, and E2E tests

**For complete implementation guide**, see [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md).

## Third-Party Dependencies

For detailed information about third-party dependencies, see the
[Dependencies Guide](DEPENDENCIES_GUIDE.md).

## Summarization Implementation

For detailed information about the summarization system, see the
[ML Provider Reference](ML_PROVIDER_REFERENCE.md).
