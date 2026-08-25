"""Core data model entities for the podcast scraper.

Defines RssFeed, Episode, and TranscriptionJob dataclasses used across
workflow, RSS parsing, and providers.
"""

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
        description: The channel-level <description> — the show's own blurb, which
            usually names the host(s) ("hosted by …"). Feeds host detection (#1169).

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
    description: Optional[str] = None


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
    # The per-item <description> blurb. ADR-137 feeds it (with the title) to the LLM's host/guest
    # role determination; without it, that prompt only ever saw the title. Populated by
    # create_episode_from_item; None when the feed item carries no description.
    description: Optional[str] = None


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
        episode: Optional reference to the source Episode (for metrics and stable IDs).

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
    # Every name the episode metadata STATED, including the ones corroboration rejected. It names
    # nobody on its own — it only lets the roster tell "we failed to place a stated name" (a defect)
    # apart from "nobody in the episode was ever named" (the vox-pop of a narrated show).
    metadata_named: Optional[List[str]] = None
    # Feed-stated host names (from the show blurb, via detect_hosts_from_feed). Anchors the
    # diarization roster so an ASR-garbled host surname canonicalizes to the feed's spelling.
    # Set on the transcription path from host_detection_result.cached_hosts.
    feed_hosts: Optional[List[str]] = None
    # Did the speaker-detection stage actually run for this episode? (#1647)
    #
    # ``detected_speaker_names`` cannot answer this: it is empty both when detection ran and
    # found nobody, and when detection never ran at all. The roster's defect accounting depends
    # on telling those apart — an ``unidentified`` voice is "nobody in the episode says who
    # they are" (not our failure) ONLY if we actually looked. When detection was skipped we
    # never looked, so the same voice is an unmeasured gap, not an accepted one.
    #
    # None = unknown (a caller that predates this field); the roster treats it as "assume we
    # looked", preserving the previous behaviour rather than inventing alarms retroactively.
    speaker_detection_ran: Optional[bool] = None
    episode: Optional[Episode] = None
    # Wall time for the media HTTP download (set when enqueueing Whisper jobs). Recorded in
    # metrics only after a transcript-cache miss so cache hits stay 0 for download_media_time.
    media_download_elapsed: Optional[float] = None
    # sha256 of the downloaded enclosure bytes, computed at the #1656 duplicate gate. Carried on
    # the job so the fingerprint is registered (post-transcription) without re-hashing the file.
    audio_sha256: Optional[str] = None
