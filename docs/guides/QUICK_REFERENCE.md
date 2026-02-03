# Quick Reference

One-page cheat sheet for common podcast_scraper commands.

---

## Setup

```bash

# Initial setup

git clone https://github.com/chipi/podcast_scraper.git && cd podcast_scraper
python3 -m venv .venv && source .venv/bin/activate
make init                    # Install all dependencies

# ML model preloading (for tests)

make preload-ml-models       # Download Whisper, spaCy, transformers models

# Cache management

make backup-cache            # Backup .cache directory (saves to ~/podcast_scraper_cache_backups/)
make restore-cache           # Restore cache from backup (interactive)
make backup-cache-list       # List available backups
```

---

## Diagnostic Commands (Issue #379)

```bash
# Run diagnostic checks
python -m podcast_scraper.cli doctor

# Include network connectivity check
python -m podcast_scraper.cli doctor --verbose
```

## Daily Development

```bash

# Before coding

source .venv/bin/activate

# After making changes

make format                  # Auto-format code (black + isort)
make lint                    # Check style (flake8)
make type                    # Type check (mypy)
make quality                 # Code quality (complexity, docstrings, dead code, spelling)

# Before committing

make ci                      # Run full CI suite locally
```

---

## Testing

| Command | What it does | Time |
| ------- | ------------ | ---- |
| `make test-unit` | Unit tests (parallel, network blocked) | ~30s |
| `make test-integration` | Integration tests (serial) | ~2min |
| `make test-e2e` | E2E tests (serial, real ML) | ~5min |
| `make test-fast` | Critical path only | ~1min |
| `make test` | All tests | ~8min |
| `make test-nightly` | Nightly tests (production models) | ~4hrs |

```bash

# Run specific test file

pytest tests/unit/test_config.py -v

# Run specific test

pytest tests/unit/test_config.py::test_config_validation -v

# Run with output visible

pytest tests/ -v --no-header

# Debug failing test

pytest tests/path/to/test.py -x -v --tb=short
```

---

## Documentation

```bash
make docs                    # Build docs site
make fix-md                  # Auto-fix markdown issues
make lint-markdown           # Check markdown style

# Local preview

mkdocs serve                 # http://localhost:8000
```

---

## CLI Usage

```bash

# Basic transcript download

python3 -m podcast_scraper.cli https://example.com/feed.xml

# With Whisper transcription

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing --whisper-model base

# Full processing

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --generate-metadata \
  --generate-summaries \
  --output-dir ./output

# From config file

python3 -m podcast_scraper.cli --config config.yaml
```

---

## Git Workflow

```bash

# Create feature branch

git checkout -b feature/my-feature

# Make changes, then:

make format && make ci       # Format and verify

# Commit

git add -A
git commit -m "feat: add my feature"

# Push and create PR

git push -u origin feature/my-feature
```

---

## Debugging

```bash

# Enable debug logging

export LOG_LEVEL=DEBUG

# Check test coverage

make test-unit
open htmlcov/index.html

# Check for linting issues

make lint 2>&1 | head -50

# Validate config file

python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `pyproject.toml` | Dependencies, pytest config |
| `Makefile` | Development commands |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `tests/conftest.py` | Shared test fixtures |
| `.markdownlint.json` | Markdown linting rules |

---

## Key Directories

| Directory | Purpose |
| --------- | ------- |
| `src/podcast_scraper/` | Main source code |
| `tests/unit/` | Unit tests |
| `tests/integration/` | Integration tests |
| `tests/e2e/` | End-to-end tests |
| `docs/` | Documentation |
| `examples/` | Config examples |

---

## Common Issues

| Issue | Fix |
| ----- | --- |
| Tests skip "model not cached" | `make preload-ml-models` |
| Import errors | `pip install -e ".[dev,ml]"` |
| Whisper fails | `brew install ffmpeg` |
| CI fails locally | `make ci` |

**See:** [Troubleshooting Guide](TROUBLESHOOTING.md)

---

## Links

- [Development Guide](DEVELOPMENT_GUIDE.md) - Full development workflow
- [Testing Guide](TESTING_GUIDE.md) - Detailed test information
- [CLI Reference](../api/CLI.md) - All CLI options
- [Configuration](../api/CONFIGURATION.md) - Config file options
