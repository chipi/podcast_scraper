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

1. **Create supervisor config file** (see `docker/supervisor.conf.example`):

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
# Use the default docker-compose.yml
docker-compose up -d
```

**LLM-only variant:**

```bash
# Use the LLM-only specific compose file
docker-compose -f docker-compose.llm-only.yml up -d
```

See `docker-compose.yml` and `docker-compose.llm-only.yml` in the repository root for complete examples with resource limits, volume mounts, and environment variables.

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

The service reads configuration from a JSON or YAML file. See [Configuration API](api/CONFIGURATION.md) for complete configuration options.

**Example `config.yaml`:**

```yaml
rss: https://example.com/feed.xml
output_dir: /app/output
max_episodes: 50
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
# ❌ BAD: Hardcoded in config file
openai_api_key: sk-abc123...

# ✅ GOOD: Use environment variables
docker run -e OPENAI_API_KEY=sk-abc123... podcast-scraper

# ✅ GOOD: Use .env file (not committed)
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

✅ **DO:**

- Use environment variables for secrets
- Mount config files as read-only
- Use versioned image tags in production
- Set resource limits
- Scan images regularly for vulnerabilities
- Use non-root user (already configured)

❌ **DON'T:**

- Commit API keys to config files
- Use `:latest` tag in production
- Run as root user
- Expose unnecessary ports
- Skip security scanning
- Hardcode secrets in Dockerfiles

## Related Documentation

- [Service API](api/SERVICE.md) - Service mode API reference
- [Configuration API](api/CONFIGURATION.md) - Config file format and options
- [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md) - LLM-only vs ML-enabled variants
- [Supervisor Example](../examples/supervisor.conf.example) - Supervisor configuration example
- [Docker Supervisor Config](../docker/supervisor.conf.example) - Docker-specific supervisor config
