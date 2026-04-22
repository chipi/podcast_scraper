# Docker Service Guide

This guide explains how to use podcast_scraper as a service-oriented Docker container, suitable for daemon/service usage with supervisor, systemd, or as a long-running container.

## Overview

The Docker container is designed to run headless without requiring CLI arguments. It automatically reads configuration from a config file and can be managed with supervisor for advanced process management.

## Quick Start

### Simple Service Mode (Direct)

Mount a config file and run:

```bash
docker run -v /host/config.yaml:/app/config.yaml \
           -v /host/output:/app/output \
           podcast_scraper
```

The container will automatically:

- Read config from `/app/config.yaml` (default location)
- Start the service without requiring `--config` argument
- Process episodes according to your configuration

### Service Mode with Custom Config Path

Use environment variable to specify a custom config path:

```bash
docker run -e PODCAST_SCRAPER_CONFIG=/custom/path/config.yaml \
           -v /host/config.yaml:/custom/path/config.yaml \
           -v /host/output:/app/output \
           podcast_scraper
```

### Supervisor Mode (Advanced)

For advanced process management with automatic restarts and logging:

```bash
docker run -v /host/config.yaml:/app/config.yaml \
           -v /host/supervisor.conf:/etc/supervisor/conf.d/podcast_scraper.conf \
           -v /host/output:/app/output \
           -v /host/logs:/var/log/podcast_scraper \
           podcast_scraper
```

## Environment Variables

Service images are easiest to operate in a [twelve-factor](https://12factor.net/config) way:
inject **secrets and per-deploy settings** with `-e` / Compose `environment`, keep a **mounted YAML**
for stable defaults, and pass paths like `PODCAST_SCRAPER_CONFIG` when the file is not at the
default location. See [CONFIGURATION.md — Twelve-factor app alignment (config)](../api/CONFIGURATION.md#twelve-factor-app-alignment-config).

| Variable | Default | Description |
| :------- | :------ | :---------- |
| `PODCAST_SCRAPER_CONFIG` | `/app/config.yaml` | Path to configuration file |
| `PODCAST_SCRAPER_WORK_DIR` | `/app` | Working directory for service |

### Setting Environment Variables

**Using `-e` flag:**

```bash
docker run -e PODCAST_SCRAPER_CONFIG=/custom/path/config.yaml \
           podcast_scraper
```

**Using Docker Compose:**

```yaml
services:
  podcast_scraper:
    image: podcast_scraper:latest
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
      - PODCAST_SCRAPER_WORK_DIR=/app
```

## Volume Mounts

### Required Volumes

**Config file:** Mount your configuration file to `/app/config.yaml` (or path specified by `PODCAST_SCRAPER_CONFIG`):

```bash
-v /host/config.yaml:/app/config.yaml
```

**Output directory:** Mount output directory to location specified in your config file:

```bash
-v /host/output:/app/output
```

### Optional Volumes

**Supervisor config:** Mount supervisor configuration to enable supervisor mode:

```bash
-v /host/supervisor.conf:/etc/supervisor/conf.d/podcast_scraper.conf
```

**Logs:** Mount log directory for supervisor-managed logging:

```bash
-v /host/logs:/var/log/podcast_scraper
```

**Model cache directories (ML variant only, recommended for persistence):**

Mount model cache directories to persist downloaded models across container restarts. This significantly speeds up container startup after the first run:

```bash
# Whisper model cache
-v /host/whisper-cache:/opt/whisper-cache

# Transformers/Hugging Face model cache
-v /host/huggingface-cache:/home/podcast/.cache/huggingface
```

**Benefits:**

- **Faster startup**: Models don't need to be re-downloaded on container restart
- **Persistent storage**: Models survive container recreation
- **Bandwidth savings**: Models downloaded once, reused across containers
- **Offline capability**: Models available even if download fails

**Example with all recommended volumes (ML variant):**

```bash
docker run -v /host/config.yaml:/app/config.yaml \
           -v /host/output:/app/output \
           -v /host/whisper-cache:/opt/whisper-cache \
           -v /host/huggingface-cache:/home/podcast/.cache/huggingface \
           -v /host/logs:/var/log/podcast_scraper \
           podcast-scraper:ml
```

## Supervisor Integration

Supervisor provides advanced process management with automatic restarts, logging, and monitoring.

### Enabling Supervisor Mode

1. **Create supervisor config file** (see `docker/pipeline/supervisor.conf.example` in the repository):

   ```ini
   [supervisord]
   nodaemon=true
   logfile=/var/log/supervisor/supervisord.log
   pidfile=/var/run/supervisord.pid

   [program:podcast_scraper]
   command=python -m podcast_scraper.service
   directory=/app
   autostart=true
   autorestart=true
   startretries=3
   startsecs=10
   stopwaitsecs=30
   stdout_logfile=/var/log/podcast_scraper/stdout.log
   stderr_logfile=/var/log/podcast_scraper/stderr.log
   stdout_logfile_maxbytes=10MB
   stderr_logfile_maxbytes=10MB
   stdout_logfile_backups=5
   stderr_logfile_backups=5
   environment=PYTHONUNBUFFERED="1",PODCAST_SCRAPER_CONFIG="/app/config.yaml"
   ```

1. **Mount supervisor config:**

   ```bash
   docker run -v /host/config.yaml:/app/config.yaml \
              -v /host/supervisor.conf:/etc/supervisor/conf.d/podcast_scraper.conf \
              podcast_scraper
   ```

1. **Container automatically detects supervisor config** and starts supervisor instead of running service directly.

### Supervisor Benefits

- **Automatic restarts:** Service restarts automatically on failure
- **Logging:** Structured logging to files with rotation
- **Monitoring:** Supervisor provides status and control commands
- **Process management:** Supervisor manages the service lifecycle

## Docker Compose Examples

### Quick Start with Provided Files

The repository includes ready-to-use Docker Compose files:

**ML-enabled variant:**

```bash
# Standalone pipeline (ML variant) from repo root
docker compose -f compose/docker-compose.yml up -d
```

**LLM-only variant:**

```bash
# Use the LLM-only specific compose file
docker compose -f compose/docker-compose.llm-only.yml up -d
```

See `compose/docker-compose.yml` and `compose/docker-compose.llm-only.yml` for complete examples with resource limits, volume mounts, and environment variables.

### Basic Service

```yaml
version: '3.8'

services:
  podcast_scraper:
    image: podcast_scraper:latest
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./output:/app/output
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
    restart: unless-stopped
```

### Service with Supervisor

```yaml
version: '3.8'

services:
  podcast_scraper:
    image: podcast_scraper:latest
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./output:/app/output
      - ./supervisor.conf:/etc/supervisor/conf.d/podcast_scraper.conf:ro
      - ./logs:/var/log/podcast_scraper
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
    restart: unless-stopped
```

### Service with Model Cache Persistence

```yaml
version: '3.8'

services:
  podcast_scraper:
    image: podcast_scraper:ml
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./output:/app/output
      # Model cache persistence (recommended)
      - ./whisper-cache:/opt/whisper-cache
      - ./huggingface-cache:/home/podcast/.cache/huggingface
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
    restart: unless-stopped
```

### Service with Custom Config Path

```yaml
version: '3.8'

services:
  podcast_scraper:
    image: podcast_scraper:latest
    volumes:
      - ./config.yaml:/custom/path/config.yaml:ro
      - ./output:/app/output
    environment:
      - PODCAST_SCRAPER_CONFIG=/custom/path/config.yaml
    restart: unless-stopped
```

## Configuration File

The service reads configuration from a JSON or YAML file. See [Configuration API](../api/CONFIGURATION.md) for complete configuration options.

**Example `config.yaml`:**

```yaml
rss: https://example.com/feed.xml
output_dir: /app/output
max_episodes: 50
# Optional episode selection (GitHub #521): episode_order, episode_offset, episode_since, episode_until
transcribe_missing: true
whisper_model: base.en
generate_metadata: true
generate_summaries: true
summary_provider: transformers
summary_model: facebook/bart-base
```

## Error Handling

### Missing Config File

If the config file is not found, the container will exit with an error:

```text
Error: Config file not found: /app/config.yaml
Please mount a config file or set PODCAST_SCRAPER_CONFIG environment variable
```

**Solution:** Mount a config file or set `PODCAST_SCRAPER_CONFIG` environment variable.

### Invalid Config File

If the config file is invalid, the service will log an error and exit:

```text
Error: Failed to load configuration file: <error details>
```

**Solution:** Validate your config file format (JSON or YAML) and ensure all required fields are present.

## Backwards Compatibility

The service maintains backwards compatibility with CLI-style usage:

```bash
# Still works: explicit --config argument
docker run -v /host/config.yaml:/app/config.yaml \
           podcast_scraper --config /app/config.yaml
```

However, the recommended approach is to use the default config path or environment variable:

```bash
# Recommended: use default path or environment variable
docker run -v /host/config.yaml:/app/config.yaml \
           podcast_scraper
```

## Troubleshooting

### Container Exits Immediately

**Check config file:**

- Ensure config file is mounted correctly
- Verify config file path matches `PODCAST_SCRAPER_CONFIG` (or default `/app/config.yaml`)
- Check config file permissions (must be readable)

**Check logs:**

```bash
docker logs <container_id>
```

### Supervisor Not Starting

**Check supervisor config:**

- Ensure supervisor config is mounted to `/etc/supervisor/conf.d/podcast_scraper.conf`
- Validate supervisor config syntax (INI format)
- Check supervisor logs: `/var/log/supervisor/supervisord.log`

### Service Fails to Start

**Check service logs:**

- Direct mode: Check container logs with `docker logs`
- Supervisor mode: Check `/var/log/podcast_scraper/stdout.log` and `/var/log/podcast_scraper/stderr.log`

**Common issues:**

- Invalid config file format
- Missing required config fields
- Network issues (RSS feed unreachable)
- Permission issues (output directory not writable)

## Performance Optimization

### Model Cache Persistence

For ML-enabled variants, mount model cache directories to avoid re-downloading models on each container restart:

```bash
-v /host/whisper-cache:/opt/whisper-cache \
-v /host/huggingface-cache:/home/podcast/.cache/huggingface
```

**Performance impact:**

- **First run**: Models downloaded (~1-2GB, 5-10 minutes)
- **Subsequent runs**: Models loaded from cache (~10-30 seconds)
- **Without cache**: Models re-downloaded on every container restart

**Example with model cache persistence:**

```bash
docker run -v /host/config.yaml:/app/config.yaml \
           -v /host/output:/app/output \
           -v /host/whisper-cache:/opt/whisper-cache \
           -v /host/huggingface-cache:/home/podcast/.cache/huggingface \
           podcast-scraper:ml
```

### Resource Limits (Production)

For production deployments, consider setting resource limits:

```yaml
services:
  podcast_scraper:
    image: podcast-scraper:ml
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

**Recommendations:**

- **LLM-only variant**: 1 CPU, 512MB-1GB RAM
- **ML-enabled variant**: 2-4 CPUs, 2-4GB RAM (depending on model size)
- **GPU support**: Not included in base image, requires custom build

### Thread Optimization

The container sets thread limits to prevent resource contention:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

These are optimal for containerized environments. Override only if you have specific performance requirements.

### Build Optimization

**Faster builds:**

- Use `--build-arg PRELOAD_ML_MODELS=false` to skip model preloading during build
- Models will be downloaded on first container run instead
- Reduces build time from ~10-15 min to ~5-7 min

**Smaller images:**

- Use LLM-only variant (`INSTALL_EXTRAS=""`) if you only need API providers
- Reduces image size from ~1-3GB to ~200-300MB

## Security Best Practices

### Secrets Management

**Never commit secrets to Docker images or config files:**

```bash
# BAD: Hardcoded in config file
openai_api_key: sk-abc123...

# GOOD: Use environment variables
docker run -e OPENAI_API_KEY=sk-abc123... podcast-scraper

# GOOD: Use .env file (not committed)
docker run --env-file .env podcast-scraper
```

**Docker Compose with secrets:**

```yaml
services:
  podcast_scraper:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}  # From .env file or host environment
```

### Non-Root User

The container runs as non-root user (`podcast`, UID 1000) for security:

- **Reduced attack surface**: Limited privileges if container is compromised
- **File permissions**: Proper ownership for application directories
- **Industry standard**: Follows Docker security best practices

### Image Security

**Regular updates:**

- Rebuild images regularly to get security patches
- Use versioned tags instead of `:latest` in production
- Monitor for vulnerabilities with `docker scan` or Snyk

**Minimal base image:**

- Uses `python:3.12-slim` (Debian-based, minimal packages)
- Multi-stage build removes build tools from final image
- Only runtime dependencies included

### Network Security

**Limit network exposure:**

- Container doesn't expose ports (no web server)
- Only outbound connections (RSS feeds, API calls)
- Use Docker networks for service isolation

### File Permissions

**Config file permissions:**

- Mount config files as read-only (`:ro` flag)
- Use proper file permissions on host (e.g., `chmod 600 config.yaml`)

```bash
# Read-only mount for security
docker run -v ./config.yaml:/app/config.yaml:ro podcast-scraper
```

### Resource Limits (Docker Compose)

Set resource limits to prevent resource exhaustion:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Security Scanning

**Scan images for vulnerabilities:**

```bash
# Using Docker Scout (built-in)
docker scout cves podcast-scraper:latest

# Using Snyk
snyk test --docker podcast-scraper:latest
```

**CI/CD integration:**

- Snyk workflow scans Docker images automatically
- Results available in GitHub Security tab

### Best Practices Summary

**DO:**

- Use environment variables for secrets
- Mount config files as read-only
- Use versioned image tags in production
- Set resource limits
- Scan images regularly for vulnerabilities
- Use non-root user (already configured)

**DON'T:**

- Commit API keys to config files
- Use `:latest` tag in production
- Run as root user
- Expose unnecessary ports
- Skip security scanning
- Hardcode secrets in Dockerfiles

## Full stack (RFC-079 / GitHub #659)

Use [`compose/docker-compose.stack.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.stack.yml) for a **three-service** topology:
**viewer** (Nginx + built Vue SPA), **api** (FastAPI `podcast_scraper.cli serve` with `--no-static`),
and **pipeline** ([`docker/pipeline/Dockerfile`](https://github.com/chipi/podcast_scraper/blob/main/docker/pipeline/Dockerfile), behind Compose profile `pipeline`).

### Prerequisites

- Docker Engine + Compose v2.
- A host config file whose `output_dir` is **`/app/output`** (matches the named volume mount).
  Copy [`config/examples/docker-stack.example.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/examples/docker-stack.example.yaml)
  and point `CONFIG_FILE` at it when running Make targets.
- **API vs `.[server]` only:** importing `create_app` requires NumPy, FAISS, and sentence-transformers
  (search routes). The [`docker/api/Dockerfile`](https://github.com/chipi/podcast_scraper/blob/main/docker/api/Dockerfile) installs `.[server]`
  plus those dependencies (CPU torch); it does **not** ship the full `.[ml]` pipeline stack.
  **Jobs API:** By default, `POST /api/jobs` from the **`api` container** spawns `podcast_scraper.cli`
  **inside that container** (subprocess), which may lack full `.[ml]` deps — use `stack-run-pipeline`
  for ops-style runs. Set **`PODCAST_PIPELINE_EXEC_MODE=docker`** (see
  [`compose/docker-compose.jobs-docker.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.jobs-docker.yml)) so the
  server uses the **#660** factory (`docker compose run` into the `pipeline` / `pipeline-llm` service).
  **Native** workflows (`make serve-api` on a full host venv) are **unchanged**
  ([RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)). Docker path requires operator
  **`pipeline_install_extras: ml | llm`**; see
  [RFC-079 §Native vs Docker](../rfc/RFC-079-full-stack-docker-compose.md#native-vs-docker).
- **Shared volume:** `corpus_data` is mounted read-write on **both** `api` and `pipeline` so the API
  can rebuild indexes or write lock files under the corpus root (read-only was deferred; see RFC-079).

### Secrets (stack)

- **Never commit API keys** in YAML, Markdown, or examples. Use **environment variables** only.
- **Local:** put keys in a repo-root **`.env`** file (gitignored). Docker Compose reads `.env` for
  **variable substitution** in `compose/docker-compose.stack.yml` (e.g. `OPENAI_API_KEY: ${OPENAI_API_KEY:-}`).
  Defaults stay **empty** — do not ship placeholder strings that look like real keys.
- **`CONFIG_FILE`:** keep provider keys **out** of the mounted operator/config YAML when possible;
  prefer env-only injection so secrets are not copied into the bind-mounted file.
- **CI / GitHub Actions:** map repository **Secrets** into the job `env:` block; never log or echo values.
- **Docker-backed jobs (#660):** merge
  [`compose/docker-compose.jobs-docker.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.jobs-docker.yml) so the
  `api` service mounts **`/var/run/docker.sock`** and receives **`PODCAST_DOCKER_PROJECT_DIR`**
  (absolute host repo root the Docker daemon can resolve). Example:
  `docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.jobs-docker.yml up -d`
  after `export PODCAST_DOCKER_PROJECT_DIR="$PWD"`.

### Commands (Makefile)

```bash
make stack-build
CONFIG_FILE=$PWD/config/examples/docker-stack.example.yaml make stack-up
curl -fsS "http://127.0.0.1:${VIEWER_PORT:-8080}/api/health"   # through Nginx
```

- **`VIEWER_PORT`:** host port mapped to Nginx `:80` (default **8080**).
- **`CONFIG_FILE`:** path to YAML mounted at `/app/config.yaml` in `api` and `pipeline`.

Run the pipeline once (writes into the same volume):

```bash
CONFIG_FILE=$PWD/config/examples/docker-stack.example.yaml make stack-run-pipeline
```

Follow logs:

```bash
make stack-logs
```

Tear down (keep the named volume):

```bash
make stack-down
```

Tear down **and** delete corpus data volume:

```bash
REMOVE_VOLUMES=1 make stack-down
```

### Manual acceptance checklist (#659)

1. `docker compose -f compose/docker-compose.stack.yml build` succeeds.
2. `docker compose -f compose/docker-compose.stack.yml up -d` starts **viewer** + **api**;
   `curl -f "http://localhost:${VIEWER_PORT:-8080}/api/health"` returns **200** (proxied by Nginx).
3. `docker compose -f compose/docker-compose.stack.yml --profile pipeline run --rm pipeline` completes
   using your `CONFIG_FILE` (the example above uses `dry_run: true` and `max_episodes: 1`).
4. After a non–dry-run pipeline, open `http://127.0.0.1:${VIEWER_PORT:-8080}/` or hit a corpus API
   route to confirm data is visible.

### RFC-079 backlog (GitHub #659): pipeline vs API, FAISS, dev compose, prod merge, smoke handoff

#### Concurrent pipeline writes and API reads (OQ4)

Several server and pipeline code paths use **atomic replace** (temp file in the same directory,
then `rename` / `Path.replace`): e.g. `podcast_scraper.server.atomic_write.atomic_write_text`,
`write_jobs_atomic` in `pipeline_job_registry.py`, `write_pipeline_status_atomic`, and
`Metrics.save_to_file` in `workflow/metrics.py`. That reduces torn reads for those artifacts.

**Caveat:** a full corpus still involves many files and stages. **Operational guidance:** keep the
same ordering as RFC-078 smoke — **finish the `pipeline` one-shot** (or job) **before** relying on
the API for new JSON / graph data. Avoid pointing production traffic at the API while the pipeline
is mid-run on the same volume unless you accept occasional parse errors on partially written files.

#### FAISS / vector index after `compose run pipeline` (OQ5)

Semantic search loads the index **per request**: `run_corpus_search` calls `FaissVectorStore.load`
on the corpus `search/` directory (`src/podcast_scraper/search/corpus_search.py`). Once the
pipeline has **fully written** the index files on the shared volume, the **next** `/api/search`
sees the new on-disk state **without** restarting the `api` container.

For a deliberate rebuild (e.g. after manual edits or corruption), use **`POST /api/index/rebuild`**
(see `src/podcast_scraper/server/routes/index_rebuild.py`). Background rebuild is gated so two
rebuilds do not overlap per corpus.

#### Compose “dev override” (Goal 8 / OQ2)

**Decision:** there is **no** `compose/docker-compose.dev.yml` in this repo. For **hot reload**
(Vite + FastAPI on the host), use **`make serve-api`** / **`make serve-ui`** / `make serve`.
The Compose stack remains the **CI / prod-like** path (`stack-*`, smoke, VPS-style merges).

#### Prod-style merge (Phase 3)

Use [`compose/docker-compose.prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.prod.yml) as an **optional
second `-f` file** on top of `docker-compose.stack.yml`: adds **`restart: unless-stopped`** for
`viewer` and `api`, and documents **external named volumes** / resource limits in comments for
VPS operators. Example:

```bash
docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml up -d
```

#### Handoff to RFC-078 smoke (stack contract)

Smoke overlays [`compose/docker-compose.smoke.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.smoke.yml) on the
**same** `viewer` / `api` / `pipeline` images. Operators should know:

| Item | Stack smoke / CI expectation |
| ---- | ----------------------------- |
| Base file | `compose/docker-compose.stack.yml` |
| Overlay | `compose/docker-compose.smoke.yml` (RFC-078) |
| `CONFIG_FILE` | **Absolute** path to YAML with `output_dir: /app/output` (see example config) |
| `VIEWER_PORT` | Smoke defaults to **8090** in the smoke compose file; override `SMOKE_BASE_URL` for Playwright |
| Corpus volume | Named volume `smoke_data` in smoke overlay (export → `.smoke-corpus/` for artifact gates) |
| Docker jobs | Optional merge [`compose/docker-compose.jobs-docker.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.jobs-docker.yml) + **#660** — not required for smoke unless testing `POST /api/jobs` in Docker mode |

Details: [`tests/smoke/README.md`](https://github.com/chipi/podcast_scraper/blob/main/tests/smoke/README.md), [RFC-078](../rfc/RFC-078-ephemeral-acceptance-smoke-test.md).

### Pipeline image tiers

The `pipeline` service in the stack reuses [`docker/pipeline/Dockerfile`](https://github.com/chipi/podcast_scraper/blob/main/docker/pipeline/Dockerfile) with two
compose-level build args (**`STACK_PIPELINE_INSTALL_EXTRAS`**, **`STACK_PIPELINE_PRELOAD_ML`**):

| Tier | `INSTALL_EXTRAS` | Size | Can run |
| ---- | ----------------- | ---- | ------- |
| **ML** (default) | `ml` | 3-4 GB | Any profile (local Whisper, spaCy, transformers, FAISS, **plus** cloud APIs) |
| **LLM** (planned) | `llm` | ~1–1.5 GB (target) | API-heavy profiles without local torch/spaCy/FAISS (e.g. future **`cloud_thin`**) — pairs with **`pipeline_install_extras: llm`** on the Docker job path |
| **Core / minimal** | `""` | smallest | Bare optional install — dev or legacy; prefer **`llm`** once shipped for thin cloud profiles |

**Today's cloud profiles** (`cloud_balanced`, `cloud_quality`) still use **spaCy trf** for NER
and **FAISS** for vector indexing, so they require the **ML** tier. A true LLM-only pipeline
needs a profile that replaces all local stages with API providers. See
[RFC-079 §Pipeline image tiers](../rfc/RFC-079-full-stack-docker-compose.md#pipeline-image-tiers)
for the full matrix.

```bash
# Faster dev build (minimal tier, no model preload) — today: INSTALL_EXTRAS=""
# Once `llm` exists: STACK_PIPELINE_INSTALL_EXTRAS=llm STACK_PIPELINE_PRELOAD_ML=false make stack-build
STACK_PIPELINE_INSTALL_EXTRAS="" STACK_PIPELINE_PRELOAD_ML=false make stack-build
```

### Stack troubleshooting

| Symptom | Likely cause |
| :------ | :----------- |
| **502 / empty** from `/api/*` | `api` not healthy yet (first start downloads HF weights — wait or check `docker compose -f compose/docker-compose.stack.yml logs -f api`). |
| **403** from Nginx | Misconfigured `root` or missing `dist/` — rebuild viewer image. |
| **API exits: output directory does not exist** | Volume not mounted at `/app/output` or config `output_dir` not `/app/output`. |
| **Pipeline cannot write** | Wrong `output_dir` in config vs `/app/output` mount. |

## Related Documentation

- [Service API](../api/SERVICE.md) - Service mode API reference
- [Configuration API](../api/CONFIGURATION.md) - Config file format and options
- [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md) - LLM-only vs ML-enabled variants
- Supervisor Example: `config/examples/supervisor.conf.example` (in repository root)
- Docker Supervisor Config: `docker/pipeline/supervisor.conf.example`
