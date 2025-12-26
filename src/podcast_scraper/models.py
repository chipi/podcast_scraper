from __future__ import annotations

# Bandit: ElementTree usage limited to typing references
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RssFeed:
    """Represents a parsed RSS feed with metadata and episode items.

    This dataclass holds the parsed RSS feed information including the feed title,
    all episode items as XML elements, the base URL for resolving relative links,
    and a list of detected authors.

    Attributes:
        title: The podcast feed title (from <title> element).
        items: List of XML elements representing individual episodes (<item> elements).
        base_url: Base URL of the RSS feed, used for resolving relative URLs.
        authors: List of author names extracted from the feed metadata.

    Example:
        >>> feed = RssFeed(
        ...     title="My Podcast",
        ...     items=[item1, item2],
        ...     base_url="https://example.com/feed.xml",
        ...     authors=["John Doe"]
        ... )
    """

    title: str
    items: List[ET.Element]
    base_url: str
    authors: List[str] = field(default_factory=list)


@dataclass
class Episode:
    """Represents a podcast episode with metadata and content URLs.

    This dataclass encapsulates all information about a single podcast episode,
    including its position in the feed, title information, transcript URLs,
    and media file details.

    Attributes:
        idx: Episode index in the feed (0-based, starting from most recent).
        title: Original episode title from RSS feed.
        title_safe: Filesystem-safe version of the title for use in filenames.
        item: Original XML element from the RSS feed containing all episode data.
        transcript_urls: List of (url, mime_type) tuples for available transcripts.
        media_url: URL of the podcast media file (audio/video). None if not available.
        media_type: MIME type of the media file (e.g., "audio/mpeg"). None if not available.

    Example:
        >>> episode = Episode(
        ...     idx=0,
        ...     title="Episode 1: Introduction",
        ...     title_safe="episode-1-introduction",
        ...     item=xml_element,
        ...     transcript_urls=[("https://example.com/transcript.vtt", "text/vtt")],
        ...     media_url="https://example.com/audio.mp3",
        ...     media_type="audio/mpeg"
        ... )
    """

    idx: int
    title: str
    title_safe: str
    item: ET.Element
    transcript_urls: List[Tuple[str, Optional[str]]]
    media_url: Optional[str] = None
    media_type: Optional[str] = None


@dataclass
class TranscriptionJob:
    """Represents a media transcription job for Whisper.

    This dataclass tracks information needed to transcribe a podcast episode's
    media file using Whisper. It includes episode metadata and paths to temporary
    media files, along with any detected speaker names for diarization.

    Attributes:
        idx: Episode index in the feed (0-based, starting from most recent).
        ep_title: Original episode title from RSS feed.
        ep_title_safe: Filesystem-safe version of the title for output filenames.
        temp_media: Path to the temporary downloaded media file to transcribe.
        detected_speaker_names: Optional list of speaker names detected from episode
            metadata or show notes. Used for screenplay formatting if available.

    Example:
        >>> job = TranscriptionJob(
        ...     idx=0,
        ...     ep_title="Episode 1: Introduction",
        ...     ep_title_safe="episode-1-introduction",
        ...     temp_media="/tmp/episode-1.mp3",
        ...     detected_speaker_names=["Alice", "Bob"]
        ... )
    """

    idx: int
    ep_title: str
    ep_title_safe: str
    temp_media: str
    detected_speaker_names: Optional[List[str]] = None
