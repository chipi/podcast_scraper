# Data Models

Data models used throughout the `podcast_scraper` codebase.

## Overview

The `models` module defines core data structures:

- `RssFeed` - Parsed RSS feed representation
- `Episode` - Individual podcast episode
- `TranscriptionJob` - Whisper transcription job

## API Reference

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

# Create episodes
ep1 = Episode(number=1, title="Ep 1", link="...", media_url="...")
ep2 = Episode(number=2, title="Ep 2", link="...", media_url="...")

# Create feed
feed = RssFeed(
    title="Example Podcast",
    description="A great podcast",
    link="https://example.com",
    episodes=[ep1, ep2]
)

print(f"Feed: {feed.title}")
print(f"Episodes: {len(feed.episodes)}")
```

## See Also

- [RSS Parser](../ARCHITECTURE.md) - How these models are populated
- [Core API](CORE.md) - How to run the pipeline using these models
