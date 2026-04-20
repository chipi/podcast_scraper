# Service API

The Service API provides a clean, programmatic interface optimized for non-interactive use, such as running as a daemon or service (e.g., with supervisor, systemd).

## Overview

The service API is designed to:

- Work exclusively with configuration files (no CLI arguments)
- Provide structured return values and error handling
- Be suitable for process management tools
- Maintain clean separation from CLI concerns
- Use the same validated **`Config`** model as the CLI: **`service.run`** builds **`Config(**config_dict)`** from the merged configuration. There is **no** separate allowlist of keys in the service layer, so documented fields such as **`preprocessing_mp3_bitrate_kbps`** are accepted whenever they are valid on **`Config`** (GitHub #561).

## Quick Start

```python
from podcast_scraper import service, Config

# Option 1: From Config object
cfg = Config(
    rss="https://example.com/feed.xml",
    output_dir="./transcripts"
)
result = service.run(cfg)

if result.success:
    print(f"Processed {result.episodes_processed} episodes")
    print(f"Summary: {result.summary}")
else:
    print(f"Error: {result.error}")

# Option 2: From config file
result = service.run_from_config_file("config.yaml")
```

**Multi-feed (GitHub #440):** If the loaded config has **two or more** feed entries in `rss_urls` (from YAML `feeds` / `rss_urls`, a promoted `rss` list, or objects with `url` plus optional per-feed overrides), `service.run` / `run_from_config_file` runs **one pipeline per feed** under `<output_dir>/feeds/<stable_feed_id>/`, matching the CLI. `output_dir` must be set in that case. After the batch, **#506** writes **`corpus_manifest.json`** and **`corpus_run_summary.json`** at the corpus parent; with **`vector_search`** and FAISS, **#505** builds one **`<output_dir>/search`** index. The return value’s **`multi_feed_summary`** field holds the same JSON-shaped dict as **`corpus_run_summary.json`** (or `None` on single-feed runs), including **`batch_incidents`** and per-feed **`episode_incidents_unique`** (schema **1.1.0**). Field tables: [CORPUS_MULTI_FEED_ARTIFACTS.md](CORPUS_MULTI_FEED_ARTIFACTS.md). See also [CONFIGURATION.md — RSS and multi-feed](CONFIGURATION.md#rss-and-multi-feed-corpus-github-440).

**Soft-only multi-feed success (GitHub #559):** **`multi_feed_strict`** defaults to **false** (lenient). A multi-feed run can then return **`success=True`** with **`error=None`** if every failed feed is classified as **soft** (same rules as in [CONFIGURATION.md — RSS and multi-feed](CONFIGURATION.md#rss-and-multi-feed-corpus-github-440)). In that case the aggregated per-feed messages are on **`ServiceResult.soft_failures`** (non-empty string). If **`success`** is false because of a hard failure or **strict** mode (`multi_feed_strict: true`), **`soft_failures`** stays **`None`**. **`multi_feed_summary`** / **`corpus_run_summary.json`** still report **`overall_ok: false`** when any feed failed. In Python, pass **`multi_feed_strict=`** into **`Config`**; deprecated YAML-only keys are documented in the same CONFIGURATION section.

**Episode selection (GitHub #521):** The same `episode_order`, `episode_since`, `episode_until`, `episode_offset`, and `max_episodes` fields in YAML/JSON apply to **each** inner single-feed run. See [CONFIGURATION.md — Episode selection](CONFIGURATION.md#episode-selection-github-521).

**Append / resume (GitHub #444):** If `Config.append` is true, each inner run uses a stable `run_append_*` directory and skips episodes that are already complete on disk (metadata `episode_id` + required artifacts). Incompatible with `clean_output`. See [CONFIGURATION.md — Append / resume](CONFIGURATION.md#append-resume-github-444).

**Corpus lock (multi-feed):** While two or more feeds are processed, `service.run` acquires an **advisory** exclusive lock file **`.podcast_scraper.lock`** under the corpus parent (`output_dir`) using `filelock`. If another process already holds the lock, the call returns immediately with **`success=False`**, **`episodes_processed=0`**, and **`error`** describing the lock conflict. Disable locking with environment variable **`PODCAST_SCRAPER_CORPUS_LOCK=0`** (tests, advanced workflows). Single-feed `service.run` does not use this lock.

## API Reference

::: podcast_scraper.service.run
    options:
      show_root_heading: true
      heading_level: 3

::: podcast_scraper.service.run_from_config_file
    options:
      show_root_heading: true
      heading_level: 3

::: podcast_scraper.service.main
    options:
      show_root_heading: true
      heading_level: 3

## ServiceResult Class

::: podcast_scraper.service.ServiceResult
    options:
      show_root_heading: true
      heading_level: 3

## Daemon Usage

### Systemd Service

```ini
[Unit]
Description=Podcast Scraper Service
After=network.target

[Service]
Type=simple
User=podcast
WorkingDirectory=/opt/podcast-scraper
ExecStart=/usr/bin/python3 -m podcast_scraper.service --config /etc/podcast-scraper/config.yaml
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

### Supervisor Configuration

```ini
[program:podcast_scraper]
command=/usr/bin/python3 -m podcast_scraper.service --config /etc/podcast-scraper/config.yaml
directory=/opt/podcast-scraper
user=podcast
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/podcast-scraper.log
```

### Programmatic Error Handling

```python
import sys
from podcast_scraper import service

result = service.run_from_config_file("config.yaml")

if not result.success:
    # Log error and exit with appropriate code
    print(f"Service failed: {result.error}", file=sys.stderr)
    sys.exit(1)

# success is True; multi-feed may still have soft-classified feed failures
if result.soft_failures:
    print(f"Warning (soft-only feed failures): {result.soft_failures}", file=sys.stderr)

print(f"Success: {result.summary}")
sys.exit(0)
```

## Docker Usage

For Docker-based deployments, see the [Docker Service Guide](../guides/DOCKER_SERVICE_GUIDE.md) which covers:

- Service-oriented Docker execution
- Environment variables and volume mounts
- Supervisor integration
- Docker Compose examples
- Troubleshooting

## See Also

- [Configuration](CONFIGURATION.md) - Configuration options
- [API Reference](REFERENCE.md) - Complete API reference
- [Docker Service Guide](../guides/DOCKER_SERVICE_GUIDE.md) - Docker service deployment
