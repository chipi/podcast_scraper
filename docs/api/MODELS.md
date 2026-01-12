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
    idx=0,
    title="Example Episode",
    title_safe="example-episode",
    item=xml_element,  # Original XML element from RSS
    transcript_urls=[("https://example.com/transcript.txt", "text/plain")],
    media_url="https://example.com/audio.mp3",
    media_type="audio/mpeg"
)

print(f"Episode {episode.idx}: {episode.title}")
print(f"Transcript URL: {episode.transcript_urls[0][0]}")
```python

from podcast_scraper.models import RssFeed, Episode

feed = RssFeed(
    title="Example Podcast",
    items=[item1, item2],  # List of ET.Element
    base_url="https://example.com/feed.xml",
    authors=["Host Name"]
)

print(f"Feed: {feed.title}")
print(f"Episodes in feed: {len(feed.items)}")

```