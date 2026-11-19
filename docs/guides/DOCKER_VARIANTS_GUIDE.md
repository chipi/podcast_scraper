# Docker Variants Guide

This guide explains how to use different Docker image variants based on your needs: LLM-only (small, fast) or ML-enabled (full features).

## Overview

`docker/pipeline/Dockerfile` builds two image variants, selected by the `INSTALL_EXTRAS` build argument. There is no `Dockerfile` at the repository root, so every `docker build` below passes `-f docker/pipeline/Dockerfile` and runs from the repository root. The tags are names you choose for your own builds.

| Variant | `INSTALL_EXTRAS` | Example tag | Size | Use Case | Dependencies |
| ------- | ---------------- | ----------- | ---- | -------- | ------------ |
| **LLM-only** | `llm` | `:llm` | ~200-300MB | Cloud API providers only | Core + the `[llm]` extra (OpenAI, Gemini, Anthropic, Mistral, Deepgram SDKs) |
| **ML-enabled** | `ml` (default) | `:ml` | ~1-3GB | Local ML models (Whisper, spaCy, Transformers) | Core + the `[ml]` extra |

An empty `INSTALL_EXTRAS` installs the core only: no cloud SDKs and no local ML. It is a valid build (CI's `docker-build-fast` job uses it for smoke tests), but it cannot run any provider that needs a third-party SDK.

## Quick Start

### LLM-Only Variant (Recommended for API Users)

For users who only use cloud API providers (OpenAI, Gemini, Anthropic, Mistral, Deepgram, ...):

```bash
# Build LLM-only image
docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=llm -t podcast-scraper:llm .

# Run LLM-only image
docker run -v /host/config.yaml:/app/config.yaml \
           -e OPENAI_API_KEY=sk-your-key \
           podcast-scraper:llm
```

**Benefits:**

- **Smaller image**: ~200-300MB vs ~1-3GB (90%+ size reduction)
- **Faster builds**: No ML dependencies to download/compile
- **Faster startup**: No model loading overhead
- **Lower resource usage**: No GPU/CPU-intensive ML libraries

**Limitations:**

- Cannot use local Whisper transcription
- Cannot use local spaCy speaker detection
- Cannot use local Transformers summarization
- Requires API keys for the cloud providers you select

### ML-Enabled Variant (Full Features)

For users who want local ML models:

```bash
# Build ML-enabled image (default)
docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .

# Or use default (ML is the Dockerfile default)
docker build -f docker/pipeline/Dockerfile -t podcast-scraper:ml .

# Run ML-enabled image
docker run -v /host/config.yaml:/app/config.yaml \
           podcast-scraper:ml
```

**Benefits:**

- **Full features**: All providers available (local + API)
- **Privacy**: Local processing, no API calls needed
- **Cost-effective**: No API costs for transcription/summarization
- **Offline capable**: Works without internet (after models downloaded)

**Limitations:**

- **Larger image**: ~1-3GB (includes ML models)
- **Slower builds**: ML dependencies take time to download/compile
- **Higher resource usage**: Requires CPU/GPU for ML inference

## Build Arguments

### `INSTALL_EXTRAS`

Controls which optional dependencies to install:

- `""` (empty) = Core only (no cloud SDKs, no local ML)
- `"llm"` = Core + cloud API SDKs (`pip install '.[llm]'`; no torch, spaCy or lancedb)
- `"ml"` = Core + ML dependencies (ML-enabled variant, default)

**Example:**

```bash
# LLM-only
docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=llm -t podcast-scraper:llm .

# ML-enabled
docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .
```

### `PRELOAD_ML_MODELS`

Only applies when `INSTALL_EXTRAS=ml`. Controls ML model preloading:

- `"true"` = Preload models during build (default)
- `"false"` = Skip model preloading (faster builds, models loaded at runtime)

**Example:**

```bash
# Build ML variant without preloading models (faster build)
docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=ml --build-arg PRELOAD_ML_MODELS=false -t podcast-scraper:ml .
```

### ML preloading details

When `INSTALL_EXTRAS=ml` and `PRELOAD_ML_MODELS=true`, the image runs
`scripts/cache/preload_ml_models.py --$PRELOAD_VARIANT`; the default `PRELOAD_VARIANT=production` is the same bundle as `make preload-ml-models-production`:
Whisper (`tiny.en` + `base.en`), production and test Transformers models, hybrid LongT5/FLAN, and GIL
evidence models. There are no separate `WHISPER_MODELS` / `SKIP_TRANSFORMERS` Docker build args; use
`PRELOAD_ML_MODELS=false` for a smaller image that downloads models at runtime. `--build-arg PRELOAD_VARIANT=airgapped-thin` bakes only what `config/profiles/airgapped_thin.yaml` needs (the compose stack-test build uses it).

> **No diarization model is baked into the image, and the diarization stage is not in
> this repository.** This repository registers no built-in stages; diarization comes
> from an installed extension. A diarization stage loads the default `community-1` (v4) or
> the `3.1` fallback (`pyannote/speaker-diarization-3.1` + `pyannote/segmentation-3.0`)
> models that are both gated on Hugging Face (401 without a token, checked 2026-11-03), and
> the image build passes no token, so the preload skips them gracefully (the build still
> succeeds). The 3.1 licence also restricts redistribution. To use neural diarization
> from a container you must **supply your own Hugging Face token at runtime**:
> accept the model terms on Hugging Face, then pass `-e HF_TOKEN=hf_...` (and a
> persistent `~/.cache/huggingface` volume so the one-time download survives
> restarts). Without a token the pipeline falls back to gap-based screenplay
> formatting, or you can run with `--no-diarize`.

## Tagging Strategy

### Recommended Tags

This repository's CI builds these variants as `podcast-scraper:test` and `podcast-scraper:test-llm` and does not publish them (`.github/workflows/docker.yml`, `push: false`). If you tag and push your own builds, a scheme that works, with `<version>` the package version (`pyproject.toml`):

**LLM-only variant:**

- `podcast-scraper:llm` - LLM-only variant
- `podcast-scraper:<version>-llm` - Versioned LLM-only variant

**ML-enabled variant:**

- `podcast-scraper:ml` - ML-enabled variant
- `podcast-scraper:latest` - Latest ML-enabled variant (default)
- `podcast-scraper:<version>` - Versioned ML-enabled variant

### Industry Best Practices

This follows common Docker tagging patterns:

1. **`:latest`** points to the most common variant (ML-enabled in this case)
2. **Variant-specific tags** (`:llm`, `:ml`) for explicit selection
3. **Versioned tags** (`:<version>`, `:<version>-llm`) for reproducibility

## Where the variants are used

The variants live as build arguments on `docker/pipeline/Dockerfile`. They show up in two services of the compose stack:

| Compose service | `INSTALL_EXTRAS` | Built by |
| --------------- | ----------------- | -------- |
| `pipeline` | `ml` | `make stack-test-build` (or `docker compose --profile pipeline build pipeline`) |
| `pipeline-llm` | `llm` | `make stack-test-build-cloud` (or `docker compose --profile pipeline-llm build pipeline-llm`) |

When the API spawns a pipeline job, `viewer_operator.yaml`'s `pipeline_install_extras` field selects which service the factory targets — `ml` → `pipeline` (ML image), `llm` → `pipeline-llm` (LLM image). See [Docker Compose guide](DOCKER_COMPOSE_GUIDE.md) for the operator UI flow that drives this selection.

## Single-container `docker run` examples

If you want to run the pipeline as a one-shot or scheduler-driven container without the full compose stack:

**LLM-only:**

```bash
docker build --build-arg INSTALL_EXTRAS=llm -f docker/pipeline/Dockerfile -t podcast-scraper:llm .
docker run -v ./config.yaml:/app/config.yaml \
           -v ./output:/app/output \
           -e OPENAI_API_KEY=sk-your-key \
           podcast-scraper:llm
```

**ML-enabled:**

```bash
docker build --build-arg INSTALL_EXTRAS=ml -f docker/pipeline/Dockerfile -t podcast-scraper:ml .
docker run -v ./config.yaml:/app/config.yaml \
           -v ./output:/app/output \
           # Optional: model cache persistence (avoids redownload on rebuild)
           -v ./whisper-cache:/opt/whisper-cache \
           -v ./huggingface-cache:/home/podcast/.cache/huggingface \
           podcast-scraper:ml
```

The single-container flow uses `podcast_scraper.service` (entrypoint fallback) which reads `/app/config.yaml`. See [Docker Service guide](DOCKER_SERVICE_GUIDE.md) for env vars, supervisor mode, and security hardening.

## Configuration Compatibility

Both variants use the same configuration format. The difference is which providers are available:

### LLM-Only Configuration

```yaml
rss: https://example.com/feed.xml
output_dir: /app/output

# Only API providers available
transcription_provider: openai
speaker_detector_provider: openai
summary_provider: openai

# OpenAI API key required
openai_api_key: ${OPENAI_API_KEY}
```

### ML-Enabled Configuration

```yaml
rss: https://example.com/feed.xml
output_dir: /app/output

# All providers available (local + API)
transcription_provider: whisper  # or openai
speaker_detector_provider: spacy  # or openai
summary_provider: transformers  # or openai

# Optional: OpenAI API key for API providers
openai_api_key: ${OPENAI_API_KEY}
```

## Migration Guide

### Switching from ML to LLM-Only

1. **Rebuild image:**

   ```bash
   docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=llm -t podcast-scraper:llm .
   ```

2. **Update configuration:**
   - Change `transcription_provider: whisper` → `transcription_provider: openai`
   - Change `speaker_detector_provider: spacy` → `speaker_detector_provider: openai`
   - Change `summary_provider: transformers` → `summary_provider: openai`

3. **Add API keys:**
   - Set `OPENAI_API_KEY` environment variable
   - Or add `openai_api_key` to config file

4. **Update Docker Compose:**
   - Change `image: podcast-scraper:ml` → `image: podcast-scraper:llm`
   - Add `OPENAI_API_KEY` environment variable

### Switching from LLM-Only to ML

1. **Rebuild image:**

   ```bash
   docker build -f docker/pipeline/Dockerfile --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .
   ```

2. **Update configuration:**
   - Change providers to local ML providers (optional, can still use API)
   - Remove `openai_api_key` if using only local providers

3. **Update Docker Compose:**
   - Change `image: podcast-scraper:llm` → `image: podcast-scraper:ml`

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  build-variants:
    strategy:
      matrix:
        variant:
          - name: llm-only
            args: INSTALL_EXTRAS=llm
            tag: podcast-scraper:llm
          - name: ml
            args: INSTALL_EXTRAS=ml
            tag: podcast-scraper:ml
    steps:
      - name: Build ${{ matrix.variant.name }} variant
        run: |
          docker build -f docker/pipeline/Dockerfile \
            --build-arg ${{ matrix.variant.args }} \
            -t ${{ matrix.variant.tag }} .
```

## Size Comparison

| Variant | Base Image | Dependencies | Models | Total |
| ------- | ---------- | ------------ | ------ | ----- |
| LLM-only | ~150MB | ~50MB | 0MB | ~200MB |
| ML-enabled | ~150MB | ~500MB | ~1-2GB | ~1.5-2.5GB |

**Note:** Actual sizes vary based on:

- Base image version
- Dependency versions
- Model preloading settings
- Build optimizations

## Performance Comparison

| Metric | LLM-only | ML-enabled |
| ------ | -------- | ---------- |
| Build time | ~2-3 min | ~10-15 min |
| Startup | <1 sec | ~5-10 sec |
| Memory usage | ~100MB | ~500MB-2GB |
| CPU usage | Low | High (during inference) |
| Network required | Yes (API calls) | No (after models loaded) |

## Decision Guide

**Choose LLM-only if:**

- You only use OpenAI/API providers
- You want smaller images and faster builds
- You have reliable internet for API calls
- You prefer API-based processing

**Choose ML-enabled if:**

- You want local Whisper transcription
- You want local spaCy speaker detection
- You want local Transformers summarization
- You need offline capability
- You want to avoid API costs
- You prioritize privacy (local processing)

## Related Documentation

- [Docker Service Guide](DOCKER_SERVICE_GUIDE.md) - Service-oriented Docker usage
- [Development Guide](DEVELOPMENT_GUIDE.md) - Local installation and setup
- [Provider Configuration](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) - Provider setup
