# Data Models

Data models used throughout the podcast_scraper codebase.

## Overview

The `models` module defines core data structures:

- `RssFeed` - Parsed RSS feed representation
- `Episode` - Individual podcast episode
- `TranscriptionJob` - Whisper transcription job

## Models

::: podcast_scraper.models.RssFeed
    options:
      show_root_heading: true
      heading_level: 3

::: podcast_scraper.models.Episode
    options:
      show_root_heading: true
      heading_level: 3

::: podcast_scraper.models.TranscriptionJob
    options:
      show_root_heading: true
      heading_level: 3

## Usage Examples

### Working with Episodes

```python
from podcast_scraper.models import Episode

episode = Episode(
    number=1,
    title="Example Episode",
    link="https://example.com/episode-1",
    transcript_url="https://example.com/transcript.txt",
    media_url="https://example.com/audio.mp3",
    media_type="audio/mpeg"
)

print(f"Episode {episode.number}: {episode.title}")
print(f"Transcript: {episode.transcript_url}")
```

### Working with Feeds

```python
from podcast_scraper.models import RssFeed, Episode

feed = RssFeed(
    title="Example Podcast",
    description="A great podcast",
    link="https://example.com",
    episodes=[episode1, episode2, episode3]
)

print(f"Feed: {feed.title}")
print(f"Episodes: {len(feed.episodes)}")
```

## Related

- [Core API](core.md) - Main functions that use these models
- [Configuration](configuration.md) - Configuration model
- RSS Parser: `rss_parser.py` - Constructs these models from RSS
