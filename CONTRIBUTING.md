# Contributing Guide

Thanks for taking the time to contribute! This project mirrors its CI pipeline locally so you can catch issues before opening a pull request.

**Table of Contents**

- [Quick Start](#quick-start)
- [Code Style Guidelines](#code-style-guidelines)
- [Development Workflow](#development-workflow)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [CI/CD Integration](#cicd-integration)
- [Architecture Principles](#architecture-principles)
- [Logging Guidelines](#logging-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Quick Start

### 1. Set up your environment

```bash
# Clone the repository
git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Option 1: Use setup script (recommended)
bash scripts/setup_venv.sh
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Option 2: Manual setup
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies and the package itself
make init
```

> `make init` upgrades pip, installs lint/test/type/security tooling, installs runtime requirements (if `requirements.txt` exists), and then installs `podcast_scraper` in editable mode with `[dev,ml]` extras. It matches the dependencies used in CI.

**Optional: Set up environment variables** (if testing OpenAI providers or configuring deployment settings):

```bash
# Copy example .env file
cp examples/.env.example .env

# Edit .env and add your settings
# OPENAI_API_KEY=sk-your-actual-key-here
# LOG_LEVEL=DEBUG
# OUTPUT_DIR=/tmp/test_output
# WORKERS=2
```

The `.env` file is automatically loaded via `python-dotenv` when the package is imported. See `docs/ENVIRONMENT_VARIABLES.md` for complete list of supported environment variables.

### 2. Run the full check suite (matches CI)

```bash
make ci
```

This command executes the same steps as the GitHub Actions workflow:

- `black`/`isort` formatting checks
- `flake8` linting
- `markdownlint` for markdown files
- `mypy` type checking
- `bandit` + `pip-audit` security scans
- `pytest` with coverage report
- `mkdocs build` documentation build (outputs to `.build/site/`)
- `python -m build` packaging sanity check (outputs to `.build/dist/`)

> **Note:** Build artifacts (distributions, documentation site, coverage reports) are organized in `.build/` directory. Test outputs are stored in `.test_outputs/`. Use `make clean` to remove all build artifacts.

### 3. Common commands

Use the Makefile targets to work faster:

```bash
make help          # List all targets
make format        # Auto-format with black + isort
make format-check  # Formatting check without modifying files
make lint          # Run flake8 linting
make lint-markdown # Run markdownlint on markdown files
make type          # Run mypy type checks
make security      # Run bandit + pip-audit security scans
make test          # Run pytest with coverage
make docs          # Build MkDocs documentation
make build         # Build sdist & wheel
make ci            # Run the full CI suite locally
make clean         # Remove build artifacts
```

---

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

**Check formatting without modifying:**

```bash
make format-check
```

### Naming Conventions

#### Variables and Functions

- Use `snake_case` for variables, functions, and methods
- Use descriptive names that indicate purpose
- Avoid single-letter names except for common iterators (`i`, `j`, `k`)

```python
# Good
def fetch_rss_feed(url: str) -> RssFeed:
    episode_count = len(feed.episodes)
    
# Bad
def fetchRSSFeed(url: str):  # camelCase
    x = len(feed.episodes)  # non-descriptive name
```

#### Classes

- Use `PascalCase` for class names
- Use descriptive nouns that represent entities

```python
# Good
class RssFeed:
    pass

class TranscriptionJob:
    pass

# Bad
class rss_feed:  # snake_case
    pass
```

#### Constants

- Use `UPPER_SNAKE_CASE` for module-level constants

```python
# Good
DEFAULT_TIMEOUT = 20
MAX_RETRIES = 3

# Bad
default_timeout = 20  # lowercase
maxRetries = 3  # camelCase
```

#### Private Members

- Prefix with single underscore for internal use

```python
class SummaryModel:
    def __init__(self):
        self._device = "cpu"  # Internal attribute
    
    def _load_model(self):  # Internal method
        pass
```

### Type Hints

**All public functions and methods must have type hints:**

```python
# Good
def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safe filesystem use."""
    pass

# Bad
def sanitize_filename(filename, max_length=255):
    """Sanitize filename for safe filesystem use."""
    pass
```

**Use Optional, Union, List, Dict from typing:**

```python
from typing import Optional, List, Dict, Union

def process_episode(
    episode: Episode,
    cfg: Config,
    progress: Optional[ProgressBar] = None
) -> Dict[str, Union[str, int]]:
    pass
```

### Docstrings

**Use Google-style or NumPy-style docstrings for all public functions:**

```python
def run_pipeline(cfg: Config, progress_factory=None) -> None:
    """Run the complete podcast scraping pipeline.
    
    Args:
        cfg: Configuration object containing RSS URL and processing options.
        progress_factory: Optional custom progress reporter (default: tqdm).
        
    Raises:
        ValueError: If configuration is invalid.
        HTTPError: If RSS feed cannot be fetched.
        
    Example:
        >>> from podcast_scraper import Config, run_pipeline
        >>> cfg = Config(rss="https://example.com/feed.xml")
        >>> run_pipeline(cfg)
    """
    pass
```

**Module-level docstrings:**

```python
"""Podcast transcript downloading and processing.

This module provides functionality for downloading podcast transcripts
from RSS feeds with optional Whisper fallback for missing transcripts.
"""
```

### Import Organization

**Order imports in three groups (isort handles this automatically):**

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, List

# Third-party
import requests
from pydantic import BaseModel

# Local
from podcast_scraper import config
from podcast_scraper.models import Episode
```

---

## Development Workflow

### Git Workflow

#### Branch Naming

Create descriptive branches for all changes:

```bash
# Feature branches
git checkout -b feature/add-postgresql-export
git checkout -b feature/issue-40-etl-loading

# Bug fix branches
git checkout -b fix/whisper-progress-indicator
git checkout -b fix/issue-19-progress-bars

# Documentation branches
git checkout -b docs/update-api-reference
git checkout -b docs/contributing-guide
```

#### Commit Messages

Follow conventional commit format:

```text
<type>: <short description>

<detailed description if needed>

Fixes #<issue-number>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `ci`: CI/CD changes
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**

```text
feat: add PostgreSQL export adapter

Implement export functionality to generate PostgreSQL-compatible SQL
dumps from episode metadata. Includes schema templates and CLI flags.

Fixes #40
```

```text
fix: resolve double progress bar in Whisper transcription

Remove duplicate progress indicators when transcribing with Whisper.
Now shows single consolidated progress bar.

Fixes #19
```

### When to Create New Files

**Create new modules when:**

- Implementing a new major feature (e.g., new summarization provider)
- A module has distinct responsibility following Single Responsibility Principle
- An existing module exceeds ~1000 lines and can be logically split

**Modify existing files when:**

- Fixing bugs
- Enhancing existing functionality
- Refactoring within the same module

### Module Boundaries

Respect established module boundaries (see `docs/ARCHITECTURE.md`):

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

---

## Testing Requirements

> **See also:** [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md) for comprehensive testing guidelines.

### Every New Function Needs

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

✅ **Test both happy path and error cases:**

```python
def test_sanitize_filename_valid(self):
    """Test filename sanitization with valid input."""
    result = sanitize_filename("Episode 1: Great Content")
    self.assertEqual(result, "Episode 1 Great Content")

def test_sanitize_filename_invalid_chars(self):
    """Test filename sanitization removes invalid characters."""
    result = sanitize_filename("Episode<>:\"/\\|?*")
    self.assertEqual(result, "Episode")
```

✅ **Use descriptive test names:**

```python
# Good
def test_config_validation_raises_error_for_negative_workers(self):
    pass

def test_whisper_model_selection_prefers_en_variant_for_english(self):
    pass

# Bad
def test_config(self):
    pass

def test_whisper(self):
    pass
```

### Every New Feature Needs

- **Integration test** (can be marked `@pytest.mark.slow` or `@pytest.mark.integration`)
- **Documentation update** (README, API docs, or relevant guide)
- **Examples** if user-facing

### Mock External Dependencies

Always mock external dependencies in tests:

- **HTTP requests**: Mock `requests` module
- **Whisper models**: Mock `whisper.load_model()` and `whisper.transcribe()`
- **File I/O**: Use `tempfile.TemporaryDirectory` for isolated tests
- **spaCy models**: Mock NER extraction for unit tests

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

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_summarizer.py

# Run specific test
pytest tests/test_summarizer.py::TestModelSelection::test_select_model_with_explicit_model

# Run with verbose output
pytest -v

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

---

## Documentation Standards

### When to Create PRD (Product Requirements Document)

Create a PRD for:

- New user-facing features
- Significant functionality additions
- Changes that affect user workflows

**Template:** `docs/prd/PRD-XXX-feature-name.md`

**Examples:**

- PRD-004: Metadata Generation
- PRD-005: Episode Summarization

### When to Create RFC (Request for Comments)

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

**`docs/TESTING_STRATEGY.md`** if:

- Testing approach changes
- New test categories are added
- Test infrastructure is updated

**API docs** if:

- Public API changes (functions, classes, parameters)
- New public modules are added
- API contracts change

---

## CI/CD Integration

> **See also:** [`docs/CI_CD.md`](docs/CI_CD.md) for complete CI/CD pipeline documentation with visualizations.

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
| `docker/Dockerfile` | ✅ Run | ❌ Skip | ❌ Skip | ~5 minutes |

This optimization provides fast feedback for documentation updates while maintaining full validation for code changes.

### Before Pushing

**Recommended workflow:**

```bash
# 1. Install pre-commit hook (one-time setup)
make install-hooks

# 2. Make your changes
# ... edit files ...

# 3. Commit (pre-commit hook runs automatically)
git commit -m "your message"
# Hook checks formatting, linting, types, markdown

# 4. Full CI check before pushing
make ci

# 5. Push to remote
git push
```

**If you haven't installed the hook:**

```bash
# Quick check before commit
make format
make test

# Full CI check before push
make ci
```

**Expected outcome:**

- All formatting checks pass
- All lints pass
- All type checks pass
- All security scans pass
- All tests pass (>80% coverage)
- Documentation builds successfully
- Package builds successfully

### CI Failure Response

If CI fails on your PR:

1. **Check the CI logs** to identify the failure
2. **Reproduce locally:** Run `make ci` to see the same failure
3. **Fix the issue** and test locally
4. **Push the fix** - CI will re-run automatically

**Common failures:**

| Issue | Solution |
| ----- | -------- |
| Formatting issues | Run `make format` to auto-fix |
| Linting errors | Fix code style issues or run `make format` |
| Type errors | Add missing type hints |
| Test failures | Fix or update tests |
| Coverage drop | Add tests for new code |
| Markdown linting | Fix markdown syntax or run `markdownlint --fix` |

**Prevent failures with pre-commit hooks:**

```bash
# Install once
make install-hooks

# Now linting failures are caught before commit!
```

---

## Architecture Principles

### Modularity

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

**Adding new configuration:**

1. Add to `Config` model in `config.py`
2. Add CLI argument in `cli.py`
3. Document in README options section
4. Update config examples in `examples/`

### Error Handling

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

### Logging Guidelines

**Use appropriate log levels to keep service/daemon logs manageable:**

The project follows a principle of **minimal INFO verbosity** - INFO logs should focus on high-level operations and important user-facing events, while detailed technical information belongs in DEBUG.

#### Log Level Guidelines

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

#### Examples

```python
# Good - INFO for high-level operation
logger.info("Processing summarization for %d episodes in parallel", len(episodes))

# Good - DEBUG for detailed technical info
logger.debug("Pre-loading %d model instances for thread safety", max_workers)
logger.debug("Successfully pre-loaded %d model instances", len(worker_models))

# Good - INFO for important results
logger.info("Summary generated in %.1fs (length: %d chars)", elapsed, len(summary))
logger.info("saved transcript: %s (transcribed in %.1fs)", rel_path, elapsed)

# Bad - INFO for technical details
logger.info("Loading summarization model: %s on %s", model_name, device)  # Should be DEBUG
logger.info("Model loaded successfully (cached for future runs)")  # Should be DEBUG
logger.info("[MAP-REDUCE VALIDATION] Input text: %d chars, %d words", ...)  # Should be DEBUG

# Good - DEBUG for technical details
logger.debug("Loading summarization model: %s on %s", model_name, device)
logger.debug("Model loaded successfully (cached for future runs)")
logger.debug("[MAP-REDUCE VALIDATION] Input text: %d chars, %d words", ...)
```

#### Module-Specific Patterns

**Workflow (`workflow.py`):**

- INFO: Episode titles, episode counts, major stages, progress milestones
- DEBUG: Model loading/unloading, host detection details, cleanup operations

**Summarization (`summarizer.py`, `metadata.py`):**

- INFO: Summary generation start/completion, important results
- DEBUG: Model selection, loading details, chunking stats, validation metrics, config details

**Whisper (`whisper_integration.py`):**

- INFO: Transcription start ("transcribing with Whisper")
- DEBUG: Model loading, fallback attempts, device details

**Episode Processing (`episode_processor.py`):**

- INFO: File save operations ("saved transcript", "saved")
- DEBUG: Download details, file reuse, speaker names

**Speaker Detection (`speaker_detection.py`):**

- INFO: Detection results ("→ Guest: %s")
- DEBUG: Model download attempts, detection failures

#### Rationale

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

### Lazy Loading for Optional Dependencies

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

---

## Pull Request Process

### Before Creating PR

**1. Ensure all checks pass:**

```bash
make ci
```

**2. Update documentation:**

- [ ] README if user-facing changes
- [ ] API docs if public API changes
- [ ] Architecture docs if design changes
- [ ] Add/update tests

**3. Commit with clear messages:**

- Follow conventional commit format
- Reference issue numbers
- Explain "why" not just "what"

### Creating the PR

**1. Create feature branch:**

```bash
git checkout -b feature/my-feature
```

**2. Push branch:**

```bash
git push -u origin feature/my-feature
```

**3. Open PR with:**

- **Clear title** (e.g., "Add PostgreSQL export adapter")
- **Description** that includes:
  - Problem being solved
  - Solution approach
  - Testing performed
  - Related issues (Fixes #XX)
  - Breaking changes (if any)

**PR Template Example:**

```markdown
## Summary

Add PostgreSQL export functionality to generate SQL dumps from episode metadata.

## Changes

- Added `export.py` module with SQL generation
- Added `--export-format sql` CLI flag
- Created SQL schema templates
- Updated documentation

## Testing

- Unit tests for SQL generation
- Integration test for full export
- Manual testing with PostgreSQL 14

## Related

Fixes #40

## Breaking Changes

None
```

### PR Review Process

1. **Automated checks run** (CI, linting, tests)
2. **Maintainer reviews code**
3. **Address feedback** if requested
4. **CI re-runs** after changes
5. **Approval and merge** once all checks pass

### After PR Merge

- Your branch will be deleted (automatic)
- Changes will be included in next release
- Release notes will be updated

---

## Getting Help

- **Documentation**: Check `docs/` directory first
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions (if enabled)
- **Architecture**: See `docs/ARCHITECTURE.md` for design principles
- **Testing**: See `docs/TESTING_STRATEGY.md` for testing guidelines

---

## Pre-commit Hooks (Recommended)

**Prevent CI failures before they happen!**

Install the pre-commit hook to automatically check your code before every commit:

```bash
# One-time setup
make install-hooks
```

The pre-commit hook automatically runs before each commit and **only checks staged files** (files you're committing), making it fast and efficient:

- ✅ **Black** formatting check (Python files)
- ✅ **isort** import sorting check (Python files)
- ✅ **flake8** linting (Python files)
- ✅ **markdownlint** (markdown files - **required** when markdown files are staged)
- ✅ **JSON syntax validation** (JSON files - uses Python's json.tool)
- ✅ **YAML syntax validation** (YAML/YML files - uses yamllint if available, otherwise Python yaml module)
- ✅ **mypy** type checking (Python files)

> **Note:** If you're committing markdown files, `markdownlint` must be installed. Install it with: `npm install -g markdownlint-cli`  
> **Note:** For better YAML validation, install `yamllint` with: `pip install yamllint` (optional - Python yaml module is used as fallback)  
> **Note:** The hook only checks files that are staged for commit, not the entire codebase. This makes it much faster and ensures you're only checking what you're actually committing.

**If any check fails, the commit is blocked** until you fix the issues.

### Skip Hook (Not Recommended)

```bash
# Skip pre-commit checks for a specific commit
git commit --no-verify -m "your message"
```

### Auto-fix Issues

```bash
# Auto-fix formatting issues
make format

# Then try committing again
git commit -m "your message"
```

**Why use hooks?**

- Catch issues locally before pushing
- Prevent CI failures from linting
- Get immediate feedback on code quality
- Save time waiting for CI

See [`docs/CI_CD.md`](docs/CI_CD.md#automatic-pre-commit-checks) for more details.

**Additional Resources:**

- [`docs/DEVELOPMENT_NOTES.md`](docs/DEVELOPMENT_NOTES.md) - Detailed notes on linting, formatting, and development workflows (updated as tooling evolves)

---

## AI Coding Guidelines

This project includes comprehensive AI coding guidelines for developers using AI assistants (Cursor, Claude Desktop, GitHub Copilot, etc.).

### For AI Assistants

**Primary reference:** `.ai-coding-guidelines.md` - This is the PRIMARY source of truth for all AI actions.

**Entry points by tool:**

- **Cursor:** `.cursor/rules/ai-guidelines.mdc` - Automatically loaded by Cursor
- **Claude Desktop:** `CLAUDE.md` - Automatically loaded when Claude runs in this directory
- **GitHub Copilot:** `.github/copilot-instructions.md` - Read by GitHub Copilot

**Critical workflow rules (from `.ai-coding-guidelines.md`):**

- ❌ **NEVER commit without showing changes and getting user approval**
- ❌ **NEVER push to PR without running `make ci` first**
- ✅ Always show `git status` and `git diff` before committing
- ✅ Always wait for explicit user approval before committing
- ✅ Always run `make ci` before pushing to PR (new or updated)

**What `.ai-coding-guidelines.md` contains:**

- Git Workflow (commit approval, PR workflow)
- Code Organization (module boundaries, patterns)
- Testing Requirements (mocking, test structure)
- Documentation Standards (PRDs, RFCs, docstrings)
- Common Patterns (configuration, error handling, logging)
- Decision Trees (when to create modules, PRDs, RFCs)

**For human contributors:** These guidelines help ensure AI assistants follow project standards. You don't need to read them unless you're configuring AI tools.

**See:** `.ai-coding-guidelines.md` for complete guidelines.

---

Thanks again for contributing! If you have questions, please open an issue or discussion.
