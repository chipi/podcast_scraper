# Docker CI Optimization Options

## Problem
Docker build and test takes >10 minutes on GitHub Actions for PRs, slowing down the PR review process.

## Current State
- **Workflow**: `.github/workflows/docker.yml`
- **Triggers**: Runs on every PR when `Dockerfile`, `.dockerignore`, `pyproject.toml`, or `*.py` files change
- **Builds**: 2 Docker images (default + multi-model)
- **Time**: ~10+ minutes per PR
- **Bottlenecks**:
  - Installing heavy ML dependencies (torch, transformers, whisper)
  - Preloading Whisper models during build
  - Building two images

## Options

### Option 1: Move Docker Build to Main Only (RECOMMENDED) ⭐
**Status**: ✅ Implemented (Hybrid: Option 1 + Option 2)

**Changes**:
- Docker build only runs on:
  - Push to `main` branch (full build with both images)
  - PRs that modify `Dockerfile` or `.dockerignore` (fast build, skip model preloading)
- Python code changes no longer trigger Docker build on PRs

**Pros**:
- ✅ Fastest PR feedback (no Docker build for most PRs)
- ✅ Docker issues still caught when Dockerfile changes
- ✅ Full Docker test suite runs on merge to main
- ✅ Simple change, easy to maintain

**Cons**:
- ⚠️ Docker issues from Python code changes only caught after merge
- ⚠️ Requires separate PR to fix Docker issues if they occur

**Impact**:
- **PR time saved**: ~10 minutes per PR (for non-Dockerfile changes)
- **Risk**: Low - Docker issues are rare and can be fixed quickly

---

### Option 2: Fast Docker Build on PRs, Full Build on Main
**Status**: Not implemented

**Changes**:
- PRs: Build only default image (skip multi-model), skip model preloading
- Main: Full build with both images and model preloading

**Implementation**:
```yaml
jobs:
  docker-build-fast:
    if: github.event_name == 'pull_request'
    # Build only default image, skip model preloading
    # Use build-arg to skip model preload: SKIP_MODEL_PRELOAD=true

  docker-build-full:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    # Full build with both images
```

- ✅ Faster PR feedback (~3-5 minutes vs 10+)
- ✅ Still validates Docker build works
- ✅ Full validation on main

**Cons**:
- ⚠️ Multi-model build not tested on PRs
- ⚠️ Model preloading not tested on PRs
- ⚠️ More complex workflow

**Impact**:
- **PR time saved**: ~5-7 minutes per PR
- **Risk**: Medium - some Docker features not tested on PRs

---

### Option 3: Conditional Docker Build (Only on Dockerfile Changes)
**Status**: Not implemented (similar to Option 1)

**Changes**:
- Only run Docker build when `Dockerfile` or `.dockerignore` actually change
- Skip for Python-only changes

**Pros**:
- ✅ Fast PR feedback for Python changes
- ✅ Docker tested when Dockerfile changes

**Cons**:
- ⚠️ Same as Option 1 (Docker issues from Python code only caught after merge)

**Impact**:
- **PR time saved**: ~10 minutes per PR (for non-Dockerfile changes)
- **Risk**: Low

---

### Option 4: Optimize Docker Build with Better Caching
**Status**: Not implemented

**Changes**:
- Improve Docker layer caching
- Use BuildKit cache mounts
- Cache pip dependencies separately
- Cache model downloads

**Implementation**:
```dockerfile
# Use BuildKit cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir .[ml]

RUN --mount=type=cache,target=/opt/whisper-cache \
    python -c "import whisper; whisper.load_model('base.en')"
```

- ✅ Faster builds when cache hits
- ✅ Still runs on every PR
- ✅ No reduction in test coverage

**Cons**:
- ⚠️ Still slow on cache misses
- ⚠️ Requires Dockerfile changes
- ⚠️ More complex caching setup

**Impact**:
- **PR time saved**: ~3-5 minutes per PR (when cache hits)
- **Risk**: Low - improves performance without reducing coverage

---

### Option 5: Parallel Docker Builds
**Status**: Not implemented

**Changes**:
- Build both images in parallel using matrix strategy
- Use separate jobs for each image

**Pros**:
- ✅ Faster overall (parallel execution)
- ✅ Still full coverage

**Cons**:
- ⚠️ Still takes ~10 minutes (just parallelized)
- ⚠️ More complex workflow
- ⚠️ Uses more CI resources

**Impact**:
- **PR time saved**: ~5 minutes per PR (parallel execution)
- **Risk**: Low - same coverage, just faster

---

### Option 6: Skip Multi-Model Build on PRs
**Status**: Not implemented

**Changes**:
- Only build default image on PRs
- Build both images on main

**Pros**:
- ✅ Faster PR feedback (~5 minutes saved)
- ✅ Full validation on main

**Cons**:
- ⚠️ Multi-model build not tested on PRs
- ⚠️ Less coverage on PRs

**Impact**:
- **PR time saved**: ~5 minutes per PR
- **Risk**: Medium - multi-model build not tested on PRs

---

## Recommendation

**Option 1 (Move to Main Only)** is recommended because:

1. **Fastest PR feedback**: No Docker build for most PRs
2. **Low risk**: Docker issues are rare and can be fixed quickly
3. **Simple**: Easy to understand and maintain
4. **Docker changes still tested**: When Dockerfile changes, Docker build still runs

**Alternative**: Combine Option 1 + Option 4:
- Move Docker build to main only (Option 1)
- Optimize Docker build with better caching (Option 4)
- Best of both worlds: fast PRs + faster Docker builds when they do run

## Implementation

**Hybrid Approach (Option 1 + Option 2 + Option 5)** has been implemented in `.github/workflows/docker.yml`:

**PR Triggers (Option 1):**
- Docker build runs on PRs only when `Dockerfile` or `.dockerignore` change
- Python code changes (`pyproject.toml`, `*.py`) no longer trigger Docker build on PRs
- **Result**: ~10 minutes saved per PR for Python-only changes

**Lightweight Build on PRs (Option 2):**
- When PRs do trigger (Dockerfile changes), they use lightweight build:
  - Build only default image (skip multi-model build)
  - Skip model preloading (`WHISPER_PRELOAD_MODELS=`)
  - **Result**: ~3-5 minutes per PR (vs 10+ minutes for full build)

**Optimized Full Build on Main (Option 5 - Parallel Builds):**
- Push to `main` triggers optimized full build:
  - **Parallel builds**: Both images (default + multi-model) build simultaneously using matrix strategy
  - **Scoped caching**: Separate cache scopes for each image type for better cache hits
  - **BuildKit cache mounts**: Already implemented in Dockerfile for pip and model caches
  - **Result**: ~5 minutes saved on main (parallel execution vs sequential)

**Performance Improvements:**
- **PRs**: ~10 minutes saved (no build for Python-only changes) + ~5-7 minutes saved (lightweight build when triggered)
- **Main**: ~5 minutes saved (parallel builds) + faster rebuilds (better caching)

## Metrics to Track

After implementing Option 1, track:
- Average PR CI time (should decrease by ~10 minutes)
- Number of Docker issues caught after merge (should be minimal)
- Developer feedback on PR speed

