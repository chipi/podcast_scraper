# Docker Build Fix

## Problem

The Docker CI workflow was failing because images built with `docker/build-push-action` were not being loaded into the Docker daemon, making them unavailable for smoke tests.

## Root Cause

When using Docker Buildx (via `docker/build-push-action`), images are built in BuildKit but not automatically imported into the local Docker daemon. This means:

```bash
# Image is built successfully
docker build -t myimage .  ✓

# But docker run can't find it
docker run myimage  ✗ (image not found)
```

## Solution

Added `load: true` parameter to both Docker build steps in `.github/workflows/docker.yml`:

```yaml
- name: Build Docker image (default)
  uses: docker/build-push-action@v5
  with:
    context: .
    file: docker/Dockerfile
    push: false
    load: true          # ← FIX: Load image into Docker daemon
    tags: podcast-scraper:test
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

## What Changed

### 1. GitHub Actions Workflow (`.github/workflows/docker.yml`)
- ✅ Added `load: true` to default build
- ✅ Added `load: true` to multi-model build
- ✅ Improved smoke tests with better error handling
- ✅ Added test for error conditions

### 2. Makefile (`Makefile`)
- ✅ Enhanced `docker-test` target with 4 comprehensive tests
- ✅ Quieter output for cleaner test results
- ✅ Better error messages

## Testing

### Automated Tests (CI)
1. **Help command** - `docker run --rm podcast-scraper:test --help`
2. **Version check** - `docker run --rm podcast-scraper:test --version`
3. **Error handling** - Verify required argument validation
4. **Multi-model build** - Test with `WHISPER_PRELOAD_MODELS=tiny.en,base.en`
5. **Dockerfile linting** - hadolint validation

### Manual Testing

```bash
# Run all tests locally
make docker-test

# Or test individually
make docker-build
docker run --rm podcast-scraper:test --help
docker run --rm podcast-scraper:test --version
```

## Verification

To verify the fix works, check:

1. ✅ Images are built successfully
2. ✅ Images are available in Docker daemon (`docker images`)
3. ✅ Smoke tests pass (--help, --version)
4. ✅ Error handling works (no args shows required error)
5. ✅ Multi-model build works

## Files Modified

1. `.github/workflows/docker.yml` - Added `load: true`, improved tests
2. `Makefile` - Enhanced `docker-test` with comprehensive checks
3. `DOCKER_BUILD_SUMMARY.md` - Added troubleshooting section

## Status

✅ **FIXED** - Docker builds now work correctly in CI and locally
