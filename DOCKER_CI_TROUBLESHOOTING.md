# Docker CI Troubleshooting Guide

## Current Status

The Docker build CI workflow has been implemented and committed:

- Commit `a247f10`: feat: Add Docker build and test CI workflow  
- Commit `eac71eb`: Fix: Load Docker images for CI testing

## Common Failure Points & Fixes

### 1. Dockerfile Trailing Newline (Fixed)

**Issue:** hadolint validator fails on trailing newlines at end of Dockerfile

**Fix Applied:** Removed trailing blank line from `docker/Dockerfile`

```diff
 ENTRYPOINT ["python", "-m", "podcast_scraper.service"]
-

```

### 2. Python Linting May Trigger on Non-Python Files

**Issue:** The workflow trigger includes `'*.py'` which means ANY .py file change triggers Docker CI

**Current Trigger Paths:**
```yaml
paths:
  - 'docker/**'
  - 'Dockerfile'
  - 'requirements.txt'
  - 'pyproject.toml'
  - '*.py'                        # This triggers on ANY Python file
  - '.github/workflows/docker.yml'
```

**Impact:** Docker CI runs even when Python changes don't affect Docker, which may cause confusion

**Recommendation:** Consider narrowing the Python file trigger to only files that affect Docker:
```yaml
paths:
  - 'docker/**'
  - 'requirements.txt'
  - 'pyproject.toml'
  - 'service.py'           # Only the entry point
  - '.github/workflows/docker.yml'
```

### 3. hadolint Warnings vs Errors

**Current Setting:** `failure-threshold: warning`

This means hadolint will fail the build on warnings, not just errors. Common warnings:

- DL3008: Pin versions in apt-get install (already handled with `--no-install-recommends`)
- DL3013: Pin versions in pip install (already using requirements.txt)
- DL3059: Multiple consecutive RUN commands (acceptable for clarity)

**If hadolint is too strict:**
```yaml
- name: Validate Dockerfile with hadolint
  uses: hadolint/hadolint-action@v3.1.0
  with:
    dockerfile: docker/Dockerfile
    failure-threshold: error  # Only fail on errors, not warnings
```

### 4. Docker Build May Fail on Missing Dependencies

**Issue:** The Dockerfile copies the entire project and runs `pip install .`

**Requirements:**

- `pyproject.toml` must be present and valid
- `requirements.txt` must be present
- All Python modules must be importable

**Verification:**
```bash
# Test if pyproject.toml is valid
python -m build --check

# Test if imports work
python -c "from podcast_scraper import config"
```

## Testing Locally

Before pushing, test the Docker build locally:

```bash
# Quick build test
make docker-build

# Full test suite
make docker-test

# Manual verification
docker build -t test-image -f docker/Dockerfile .
docker run --rm test-image --help
docker run --rm test-image --version
```

## Workflow Structure

The Docker CI workflow (`./github/workflows/docker.yml`) includes:

1. **Build default image** - Tests standard single-model build
2. **Build multi-model image** - Tests build-args functionality  
3. **Smoke tests** - Validates basic functionality (--help, --version)
4. **Error handling** - Ensures proper arg validation
5. **hadolint** - Validates Dockerfile best practices

## Files Modified

- `.github/workflows/docker.yml` - Docker CI workflow (committed)
- `docker/Dockerfile` - Removed trailing newline (this change)
- `Makefile` - Added docker-build, docker-test, docker-clean targets (committed)
- `README.md` - Added Docker testing documentation (committed)

## Next Steps

If CI is still failing:

1. Check the actual CI logs for specific error messages
2. Look for hadolint warnings that might be treated as errors
3. Verify all Python imports work correctly
4. Check if the workflow trigger is too broad (consider narrowing `*.py`)

## Quick Fix Checklist

- [x] Remove trailing newlines from Dockerfile
- [x] Add `load: true` to build-push-action
- [ ] Consider narrowing workflow trigger paths
- [ ] Consider relaxing hadolint to `failure-threshold: error`
