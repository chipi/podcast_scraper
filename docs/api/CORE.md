# Core API

This is the primary public API for podcast_scraper. Use these functions for programmatic access.

## Quick Start

````python
import podcast_scraper

# Create configuration

cfg = podcast_scraper.Config(
    rss="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=10
)

# Run the pipeline

count, summary = podcast_scraper.run_pipeline(cfg)
print(f"Downloaded {count} transcripts: {summary}")
```text

    options:
      show_root_heading: true
      heading_level: 3

::: podcast_scraper.load_config_file
    options:
      show_root_heading: true
      heading_level: 3

## Package Information

::: podcast_scraper.__version__
    options:
      show_root_heading: true
      heading_level: 3

::: podcast_scraper.__api_version__
    options:
      show_root_heading: true
      heading_level: 3

## Related

- [Configuration](CONFIGURATION.md) - Detailed configuration options
- [Service API](SERVICE.md) - Non-interactive service interface
- [CLI Interface](CLI.md) - Command-line interface

````
