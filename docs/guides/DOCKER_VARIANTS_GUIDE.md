# Docker Variants Guide

This guide explains how to use different Docker image variants based on your needs: LLM-only (small, fast) or ML-enabled (full features).

## Overview

Podcast Scraper provides two Docker image variants:

| Variant | Tag | Size | Use Case | Dependencies |
| ------- | --- | ---- | ------- | ------------ |
| **LLM-only** | `:llm-only` or `:latest-llm` | ~200-300MB | OpenAI/API providers only | Core + OpenAI SDK |
| **ML-enabled** | `:ml` or `:latest` | ~1-3GB | Local ML models (Whisper, spaCy, Transformers) | Core + ML dependencies |

## Quick Start

### LLM-Only Variant (Recommended for API Users)

For users who only use OpenAI or other LLM API providers:

```bash
# Build LLM-only image
docker build --build-arg INSTALL_EXTRAS="" -t podcast-scraper:llm-only .

# Run LLM-only image
docker run -v /host/config.yaml:/app/config.yaml \
           -e OPENAI_API_KEY=sk-your-key \
           podcast-scraper:llm-only
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
- Requires API keys for OpenAI providers

### ML-Enabled Variant (Full Features)

For users who want local ML models:

```bash
# Build ML-enabled image (default)
docker build --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .

# Or use default (ML is default for backwards compatibility)
docker build -t podcast-scraper:ml .

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

- `""` (empty) = Core only (LLM-only variant)
- `"ml"` = Core + ML dependencies (ML-enabled variant, default)

**Example:**

```bash
# LLM-only
docker build --build-arg INSTALL_EXTRAS="" -t podcast-scraper:llm-only .

# ML-enabled
docker build --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .
```

### `PRELOAD_ML_MODELS`

Only applies when `INSTALL_EXTRAS=ml`. Controls ML model preloading:

- `"true"` = Preload models during build (default)
- `"false"` = Skip model preloading (faster builds, models loaded at runtime)

**Example:**

```bash
# Build ML variant without preloading models (faster build)
docker build --build-arg INSTALL_EXTRAS=ml --build-arg PRELOAD_ML_MODELS=false -t podcast-scraper:ml .
```

### Other ML Build Arguments

When `INSTALL_EXTRAS=ml`, you can also control:

- `WHISPER_MODELS`: Comma-separated list of Whisper models to preload (default: `base.en`)
- `TRANSFORMERS_MODELS`: Comma-separated list of Transformers models to preload (default: all)
- `SKIP_TRANSFORMERS`: Set to `"1"` to skip Transformers preloading

## Tagging Strategy

### Recommended Tags

**LLM-only variant:**

- `podcast-scraper:llm-only` - LLM-only variant
- `podcast-scraper:latest-llm` - Latest LLM-only variant
- `podcast-scraper:2.5.0-llm` - Versioned LLM-only variant

**ML-enabled variant:**

- `podcast-scraper:ml` - ML-enabled variant
- `podcast-scraper:latest` - Latest ML-enabled variant (default)
- `podcast-scraper:2.5.0` - Versioned ML-enabled variant

### Industry Best Practices

This follows common Docker tagging patterns:

1. **`:latest`** points to the most common variant (ML-enabled in this case)
2. **Variant-specific tags** (`:llm-only`, `:ml`) for explicit selection
3. **Versioned tags** (`:2.5.0`, `:2.5.0-llm`) for reproducibility

## Docker Compose Examples

### Quick Start with Provided Files

The repository includes ready-to-use Docker Compose files:

**ML-enabled variant:**

```bash
# Use the default docker-compose.yml
docker-compose up -d
```

**LLM-only variant:**

```bash
# Use the LLM-only specific compose file
docker-compose -f docker-compose.llm-only.yml up -d
```

See `docker-compose.yml` and `docker-compose.llm-only.yml` in the repository root for complete examples with resource limits, volume mounts, and environment variables.

### LLM-Only Service

**Using provided file (`docker-compose.llm-only.yml`):**

```bash
docker-compose -f docker-compose.llm-only.yml up -d
```

**Or custom configuration:**

```yaml
version: '3.8'

services:
  podcast_scraper:
    image: podcast-scraper:llm-only
    build:
      context: .
      args:
        INSTALL_EXTRAS: ""
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./output:/app/output
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

### ML-Enabled Service

**Using provided file (`docker-compose.yml`):**

```bash
docker-compose up -d
```

**Or custom configuration:**

```yaml
version: '3.8'

services:
  podcast_scraper:
    image: podcast-scraper:ml
    build:
      context: .
      args:
        INSTALL_EXTRAS: ml
        PRELOAD_ML_MODELS: "true"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./output:/app/output
      # Model cache persistence (recommended)
      - ./whisper-cache:/opt/whisper-cache
      - ./huggingface-cache:/home/podcast/.cache/huggingface
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

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
   docker build --build-arg INSTALL_EXTRAS="" -t podcast-scraper:llm-only .
   ```

2. **Update configuration:**
   - Change `transcription_provider: whisper` → `transcription_provider: openai`
   - Change `speaker_detector_provider: spacy` → `speaker_detector_provider: openai`
   - Change `summary_provider: transformers` → `summary_provider: openai`

3. **Add API keys:**
   - Set `OPENAI_API_KEY` environment variable
   - Or add `openai_api_key` to config file

4. **Update Docker Compose:**
   - Change `image: podcast-scraper:ml` → `image: podcast-scraper:llm-only`
   - Add `OPENAI_API_KEY` environment variable

### Switching from LLM-Only to ML

1. **Rebuild image:**

   ```bash
   docker build --build-arg INSTALL_EXTRAS=ml -t podcast-scraper:ml .
   ```

2. **Update configuration:**
   - Change providers to local ML providers (optional, can still use API)
   - Remove `openai_api_key` if using only local providers

3. **Update Docker Compose:**
   - Change `image: podcast-scraper:llm-only` → `image: podcast-scraper:ml`

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  build-variants:
    strategy:
      matrix:
        variant:
          - name: llm-only
            args: INSTALL_EXTRAS=
            tag: podcast-scraper:llm-only
          - name: ml
            args: INSTALL_EXTRAS=ml
            tag: podcast-scraper:ml
    steps:
      - name: Build ${{ matrix.variant.name }} variant
        run: |
          docker build \
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

- ✅ You only use OpenAI/API providers
- ✅ You want smaller images and faster builds
- ✅ You have reliable internet for API calls
- ✅ You prefer API-based processing

**Choose ML-enabled if:**

- ✅ You want local Whisper transcription
- ✅ You want local spaCy speaker detection
- ✅ You want local Transformers summarization
- ✅ You need offline capability
- ✅ You want to avoid API costs
- ✅ You prioritize privacy (local processing)

## Related Documentation

- [Docker Service Guide](DOCKER_SERVICE_GUIDE.md) - Service-oriented Docker usage
- [Installation Guide](INSTALLATION_GUIDE.md) - Local installation variants
- [Provider Configuration](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) - Provider setup
- [AI Provider Comparison](AI_PROVIDER_COMPARISON_GUIDE.md) - Compare providers
