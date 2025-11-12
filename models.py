from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - used only for typing references
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RssFeed:
    """Represents a parsed RSS feed with metadata and episode items."""

    title: str
    items: List[ET.Element]
    base_url: str


@dataclass
class Episode:
    """Represents a podcast episode with metadata and content URLs."""

    idx: int
    title: str
    title_safe: str
    item: ET.Element
    transcript_urls: List[Tuple[str, Optional[str]]]
    media_url: Optional[str] = None
    media_type: Optional[str] = None


@dataclass
class TranscriptionJob:
    """Represents a media transcription job for Whisper."""

    idx: int
    ep_title: str
    ep_title_safe: str
    temp_media: str
