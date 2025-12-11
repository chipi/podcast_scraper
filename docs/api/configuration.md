# Configuration API

The `Config` class is the central configuration model for podcast_scraper, built on Pydantic for validation and type safety.

## Overview

All runtime options flow through the `Config` model:

```python
from podcast_scraper import Config

cfg = Config(
    rss="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=50,
    transcribe_missing=True,
    whisper_model="base",
    workers=8
)
```

## Config Class

::: podcast_scraper.Config
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      group_by_category: true
      show_category_heading: true

## Helper Functions

::: podcast_scraper.config.load_config_file
    options:
      show_root_heading: true
      heading_level: 3

## Configuration Files

### JSON Example

```json
{
  "rss": "https://example.com/feed.xml",
  "output_dir": "./transcripts",
  "max_episodes": 50,
  "transcribe_missing": true,
  "whisper_model": "base",
  "workers": 8,
  "transcription_parallelism": 1,
  "processing_parallelism": 2,
  "generate_metadata": true,
  "generate_summaries": true,
  "summary_batch_size": 1,
  "summary_chunk_parallelism": 1
}
```

### YAML Example

```yaml
rss: https://example.com/feed.xml
output_dir: ./transcripts
max_episodes: 50
transcribe_missing: true
whisper_model: base
workers: 8
transcription_parallelism: 1  # Number of episodes to transcribe in parallel (Whisper ignores >1, OpenAI uses for parallel API calls)
processing_parallelism: 2  # Number of episodes to process (metadata/summarization) in parallel
generate_metadata: true
generate_summaries: true
summary_batch_size: 1  # Episode-level parallelism: Number of episodes to summarize in parallel
summary_chunk_parallelism: 1  # Chunk-level parallelism: Number of chunks to process in parallel within a single episode
```

## Field Aliases

The `Config` model supports field aliases for convenience:

- `rss_url` or `rss` → `rss_url`
- `output_dir` or `output_directory` → `output_dir`
- `screenplay_gap` or `screenplay_gap_s` → `screenplay_gap_s`
- And more...

## Validation

The `Config` model performs validation on initialization:

```python
from podcast_scraper import Config
from pydantic import ValidationError

try:
    cfg = Config(
        rss="https://example.com/feed.xml",
        workers=-1  # Invalid: must be >= 1
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Related

- [Core API](core.md) - Main functions
- [CLI Interface](cli.md) - Command-line usage
- Configuration examples: `examples/config.example.json`, `examples/config.example.yaml`
