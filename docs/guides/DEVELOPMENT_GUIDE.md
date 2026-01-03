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
make test-unit            # Unit tests (parallel)
make test-integration     # Integration tests (parallel, reruns)
make test-e2e             # E2E tests (serial first, then parallel)
make test                 # All tests
make test-fast            # Unit + critical path only
```

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

1. **Verify locally**: Run `make test-unit-no-ml` before pushing

## Environment Setup

### Virtual Environment

**Quick setup:**

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

### Environment Variables

1. **Copy example `.env` file:**

```bash
cp examples/.env.example .env
```

1. **Edit `.env` and add your settings:**

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

1. **The `.env` file is automatically loaded** via `python-dotenv` when `podcast_scraper.config` module is imported.

**Security notes:**

- ✅ `.env` is in `.gitignore` (never committed)
- ✅ `examples/.env.example` is safe to commit (template only)
- ✅ API keys are never logged or exposed
- ✅ Environment variables take precedence over `.env` file

**Priority order:**

- Config file field (highest priority)
- Environment variable
- Default value
- Exception: `LOG_LEVEL` (env var takes precedence)

**See also:**

- `docs/api/CONFIGURATION.md` - Configuration API reference (includes environment variables)
- `docs/rfc/RFC-013-openai-provider-implementation.md` - API key management details
- `docs/prd/PRD-006-openai-provider-integration.md` - OpenAI provider requirements

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

### What's in `.ai-coding-guidelines.md`

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

The project uses automated formatting tools to ensure consistency:

- **Black**: Code formatting (line length: 100 characters)
- **isort**: Import statement organization
- **flake8**: Linting and style enforcement
- **mypy**: Static type checking

**Apply formatting automatically:**

```bash
make format
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

> **See also:** [`CI_CD.md`](../CI_CD.md) for complete CI/CD pipeline documentation with visualizations.

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
| Markdown linting | Run `python scripts/fix_markdown.py` or `markdownlint --fix` |

**Prevent failures with pre-commit hooks:**

```bash

# Install once

make install-hooks

# Now linting failures are caught before commit!

```

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
4. Update config examples in `examples/`

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

```python
    import whisper
    WHISPER_AVAILABLE = True
```

except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available, transcription disabled")

```python

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
- **`workflow.py`**: Orchestration only, no HTTP/IO details
- **`config.py`**: Configuration models and validation
- **`downloader.py`**: HTTP operations only
- **`filesystem.py`**: File system utilities only
- **`rss_parser.py`**: RSS parsing, episode creation
- **`episode_processor.py`**: Episode-level processing logic
- **`whisper_integration.py`**: Whisper transcription interface
- **`speaker_detection.py`**: NER-based speaker extraction
- **`summarizer.py`**: Transcript summarization
- **`metadata.py`**: Metadata document generation
- **`progress.py`**: Progress reporting abstraction
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
[Summarization Guide](SUMMARIZATION_GUIDE.md).
