# Troubleshooting Guide

Common issues and solutions for podcast_scraper development and usage.

---

## Quick Diagnosis

| Symptom | Likely Cause | Solution |
| ------- | ------------ | -------- |
| Tests skip with "model not cached" | ML models not preloaded | `make preload-ml-models` |
| `ModuleNotFoundError: transformers` | Missing ML dependencies | `pip install -e ".[ml]"` |
| Whisper fails silently | ffmpeg not installed | Install ffmpeg (see below) |
| CI fails but local passes | Different Python/dependency versions | `make ci` locally |
| Memory errors in tests | ML models loading repeatedly | Use `@pytest.mark.serial` |
| Import errors after pull | Dependencies changed | `pip install -e ".[dev,ml]"` |
| Tests hang with `-s` flag | tqdm + parallel execution deadlock | Use `-v` instead, or `-n 0` |
| OpenAI episodes skipped | Audio file size > 25MB | Use local Whisper or compress audio |

---

## ML Dependencies

### "Model not cached" Test Skips

**Symptom:** Tests skip with messages like "Whisper model not cached" or
"spaCy model not available".

**Solution:**

```bash
# Preload all ML models (requires network)
make preload-ml-models

# Verify models are cached (project-local cache)
ls -la .cache/whisper/          # Whisper models (tiny.en.pt, etc.)
ls -la .cache/huggingface/hub/  # Transformers models (bart, led)
python -c "import spacy; spacy.load('en_core_web_sm')"
```

**Note:** Models are cached in the project-local `.cache/` directory, not `~/.cache/`.
See `.cache/README.md` for cache structure details.

**Backup/Restore:** If you need to backup or restore your cache (e.g., when switching machines or after cleanup):

```bash
# Backup cache
make backup-cache

# Restore cache (interactive)
make restore-cache
```

See `.cache/README.md` for detailed backup/restore instructions.

### Whisper Model Download Fails

**Symptom:** Network errors when loading Whisper models.

**Solution:**

```bash
# Preload models using make target (recommended)
make preload-ml-models

# Or download model manually
python -c "import whisper; whisper.load_model('tiny.en')"

# Check project-local cache
ls .cache/whisper/

# Use smaller model for testing
python3 -m podcast_scraper.cli feed.xml --whisper-model tiny
```

### transformers/torch Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**

```bash

# Install ML dependencies

pip install -e ".[ml]"

# Or for development

pip install -e ".[dev,ml]"
```

### Memory Issues with ML Models

**Symptom:** Tests crash with memory errors, or system becomes unresponsive.

**Causes:**

- Multiple tests loading same models in parallel
- Large models (LED, BART) consuming GPU/CPU memory
- Too many parallel workers for available RAM

**Memory estimates per test type:**

| Test Type | Per Worker | 8 Workers | Recommended RAM |
| --------- | ---------- | --------- | --------------- |
| Unit | ~100 MB | ~1 GB | 4 GB |
| Integration | ~1-2 GB | ~8-16 GB | 16 GB |
| E2E | ~1.5-3 GB | ~12-24 GB | 32 GB |

**Solutions:**

```bash
# Reduce parallel workers (default is 8)
PYTEST_WORKERS=4 make test-integration

# Set smaller batch sizes
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**Parallelism configuration:**

The Makefile uses a dynamic formula: `min(max(1, cpu_count - 2), 8)`

- Reserves 2 cores for system
- Caps at 8 to prevent memory exhaustion
- Override with `PYTEST_WORKERS=N` environment variable

**Memory analysis script:**

```bash
# Analyze memory usage during tests
python scripts/tools/analyze_test_memory.py --test-target test-unit

# With limited workers
python scripts/tools/analyze_test_memory.py --test-target test-integration --max-workers 4
```

---

## Whisper / Transcription

### ffmpeg Not Found

**Symptom:** Whisper transcription fails silently or with "ffmpeg not found".

**Solution:**

```bash

# macOS

brew install ffmpeg

# Ubuntu/Debian

sudo apt install ffmpeg

# Verify installation

ffmpeg -version
```

### Episodes Skipped with OpenAI Provider

**Symptom:** Some episodes are skipped when using the OpenAI transcription provider, with a log message about "exceeds OpenAI API limit (25 MB)".

**Cause:** The OpenAI Whisper API has a hard limit of 25MB per audio file. To avoid API errors and wasting bandwidth, the system checks the file size before downloading.

**Solutions:**

1. **Use Local Whisper**: The local Whisper provider does not have this file size limit (it's only limited by your system's RAM).
2. **Compress Audio**: Use a tool like `ffmpeg` to reduce the bitrate or convert to a more efficient format (like Mono instead of Stereo) before processing if you must use the OpenAI API.
3. **Wait for Future Feature**: Plan for automatic audio preprocessing (downsampling/mono conversion) is in the roadmap.

---

## Test Failures

### Tests Pass Locally but Fail in CI

**Common causes:**

1. **Different Python version** - CI uses Python 3.10+
2. **Missing dependencies** - CI installs fresh each time
3. **Network calls** - CI blocks external network in unit tests
4. **File paths** - Hardcoded paths that don't exist in CI

**Debug steps:**

```bash

# Run full CI suite locally

make ci

# Run with same isolation as CI

make test-unit  # Network blocked for unit tests

# Check Python version

python --version
```

## Flaky Tests

**Symptom:** Tests pass sometimes, fail other times.

**Common causes:**

- Race conditions in parallel execution
- Shared state between tests
- Network timeouts

**Solutions:**

```bash

# Run serially to identify race conditions

pytest tests/integration/ -x -v --no-header

# Check for shared fixtures

grep -r "scope=" tests/conftest.py

# Add serial marker for problematic tests
# @pytest.mark.serial

```

### Test Hangs with `-s` Flag

**Symptom:** Tests hang indefinitely when using `-s` (no capture) with parallel execution.

**Root cause:** The `-s` flag disables pytest's output capturing. When combined with
pytest-xdist parallel execution (`-n auto`), this causes deadlocks because:

1. Multiple worker processes write to stdout/stderr simultaneously
2. No buffering means writes can interleave and block
3. **tqdm progress bars** (used by Whisper) compete for terminal control
4. Terminal locking causes processes to wait indefinitely

**Files using tqdm:**

| File | Usage |
| ---- | ----- |
| `src/podcast_scraper/whisper_integration.py` | `InterceptedTqdm` class |
| `src/podcast_scraper/transcription/whisper_provider.py` | `InterceptedTqdm` class |
| `src/podcast_scraper/ml/ml_provider.py` | `InterceptedTqdm` class |
| `src/podcast_scraper/cli.py` | `_TqdmProgress` class |

**Structural Fix:** Tests set `TQDM_DISABLE=1` environment variable in `tests/conftest.py`
to disable all tqdm progress bars during test execution.

**Workarounds:**

```bash
# Use -v instead of -s (provides verbose output without hang)
pytest tests/unit/ -v

# Disable parallelism when using -s
pytest tests/unit/ -s -n 0

# Use sequential Makefile targets
make test-unit-sequential

# Use --tb=short for better error output
pytest tests/unit/ --tb=short

# For debugging, use --pdb instead
pytest tests/unit/ --pdb
```yaml

---

## CI/CD Issues

### Pre-commit Hooks Failing

**Symptom:** Commits rejected by pre-commit hooks.

**Solution:**

```bash

# Run formatters

make format

# Fix markdown

make fix-md

# Run all checks

make lint
```python

## Documentation Build Fails

**Symptom:** `mkdocs build` fails with import errors.

**Solution:**

```bash

# Install docs dependencies

pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings mkdocstrings-python

# For API docs that import the package

pip install -e ".[ml]"
```

## Coverage Below Threshold

**Symptom:** CI fails with "Coverage below 80%".

**Solution:**

```bash

# Check current coverage

make test-unit
open htmlcov/index.html

# Identify uncovered code

coverage report --show-missing
```yaml

---

## Development Environment

### Virtual Environment Issues

**Symptom:** Wrong Python version or packages not found.

**Solution:**

```bash

# Create fresh venv

rm -rf .venv
python3.10 -m venv .venv
source .venv/bin/activate

# Reinstall everything

make init
```

## Import Errors After Git Pull

**Symptom:** `ImportError` or `ModuleNotFoundError` after pulling changes.

**Solution:**

```bash

# Reinstall package in editable mode

pip install -e ".[dev,ml]"

# Or use make target

make init
```

## mypy Type Errors

**Symptom:** `make type` fails with type errors.

**Common fixes:**

```bash

# Update type stubs

pip install --upgrade types-requests types-PyYAML

# Check specific file

mypy src/podcast_scraper/your_file.py --show-error-codes
```yaml

---

## Runtime Issues

### Configuration Not Loading

**Symptom:** CLI ignores config file settings.

**Debug steps:**

```bash

# Validate config file

python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for typos in keys

cat config.yaml

# Use verbose mode

python3 -m podcast_scraper.cli --config config.yaml -v
```

## Output Directory Errors

**Symptom:** "Permission denied" or "Directory not found".

**Solution:**

```bash

# Check directory exists and is writable

ls -la /path/to/output/
mkdir -p /path/to/output/

# Use absolute path

python3 -m podcast_scraper.cli feed.xml --output-dir /absolute/path/
```

## RSS Feed Parsing Errors

**Symptom:** "Invalid feed" or no episodes found.

**Debug steps:**

```bash

# Check feed is accessible

curl -I "https://example.com/feed.xml"

# Validate RSS format

python -c "import feedparser; print(feedparser.parse('https://example.com/feed.xml'))"
```yaml

---

## Getting Help

If your issue isn't covered here:

1. **Search existing issues:**
   [GitHub Issues](https://github.com/chipi/podcast_scraper/issues)

2. **Check logs:**

   ```bash
   # Enable debug logging
   export LOG_LEVEL=DEBUG
   python3 -m podcast_scraper.cli ...
   ```

1. **Open a new issue** with:
   - Python version (`python --version`)
   - OS and version
   - Full error message/traceback
   - Steps to reproduce

---

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md) - Test execution and debugging
- [Development Guide](DEVELOPMENT_GUIDE.md) - Environment setup
- [CI/CD](../ci/index.md) - Pipeline configuration
- [Dependencies Guide](DEPENDENCIES_GUIDE.md) - Package management
