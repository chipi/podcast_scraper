# Service API

The Service API provides a clean, programmatic interface optimized for non-interactive use,
such as running as a daemon or service (e.g., with supervisor, systemd).

## Overview

The service API is designed to:

- Work exclusively with configuration files (no CLI arguments)
- Provide structured return values and error handling
- Be suitable for process management tools
- Maintain clean separation from CLI concerns

## Quick Start

````python
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
```text

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
```json

[program:podcast_scraper]
command=/usr/bin/python3 -m podcast_scraper.service --config /etc/podcast-scraper/config.yaml
directory=/opt/podcast-scraper
user=podcast
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/podcast-scraper.log

```python
from podcast_scraper import service

result = service.run_from_config_file("config.yaml")

if not result.success:

    # Log error and exit with appropriate code

    print(f"Service failed: {result.error}", file=sys.stderr)
    sys.exit(1)

# Continue with success

print(f"Success: {result.summary}")
sys.exit(0)

```text

- [Configuration](CONFIGURATION.md) - Configuration options
- Service examples: `examples/systemd.service.example`, `examples/supervisor.conf.example`

````
