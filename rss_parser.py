"""RSS feed parsing and episode metadata extraction."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET  # nosec B405 - parsing handled via defusedxml safe APIs
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urljoin

from defusedxml.ElementTree import ParseError as DefusedXMLParseError, fromstring as safe_fromstring

from . import config, downloader, filesystem, models

logger = logging.getLogger(__name__)


def parse_rss_items(xml_bytes: bytes) -> Tuple[str, List[str], List[ET.Element]]:
    """Parse RSS XML and extract channel title, authors, and items.

    Args:
        xml_bytes: Raw RSS feed XML content

    Returns:
        Tuple of (channel_title, list_of_authors, list_of_items)
    """
    root = safe_fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        channel = next(
            (e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("channel")), None
        )
    title = ""
    authors: List[str] = []
    if channel is not None:
        t = channel.find("title") or next(
            (e for e in channel.iter() if isinstance(e.tag, str) and e.tag.endswith("title")), None
        )
        if t is not None and t.text:
            title = t.text.strip()

        # Extract author tags from channel (top-level only)
        # RSS 2.0: <author> (should be only one at channel level)
        # iTunes: <itunes:author> and <itunes:owner> (can help confirm host)
        # Note: We only look at direct children of channel, not nested elements

        # RSS 2.0 author (channel-level only, should be single)
        author_elem = channel.find("author")
        if author_elem is not None and author_elem.text:
            author_text = author_elem.text.strip()
            if author_text:
                authors.append(author_text)

        # iTunes author (channel-level only)
        itunes_author_elem = channel.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}author")
        if itunes_author_elem is not None and itunes_author_elem.text:
            itunes_author_text = itunes_author_elem.text.strip()
            if itunes_author_text and itunes_author_text not in authors:
                authors.append(itunes_author_text)

        # iTunes owner (channel-level only, can help confirm host)
        # Format: <itunes:owner><itunes:name>Name</itunes:name></itunes:owner>
        itunes_owner_elem = channel.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}owner")
        if itunes_owner_elem is not None:
            itunes_owner_name_elem = itunes_owner_elem.find(
                "{http://www.itunes.com/dtds/podcast-1.0.dtd}name"
            )
            if itunes_owner_name_elem is not None and itunes_owner_name_elem.text:
                itunes_owner_text = itunes_owner_name_elem.text.strip()
                if itunes_owner_text and itunes_owner_text not in authors:
                    authors.append(itunes_owner_text)

        items = list(channel.findall("item"))
        if not items:
            items = [e for e in channel if isinstance(e.tag, str) and e.tag.endswith("item")]
    else:
        items = [e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("item")]
    return title, authors, items


def find_transcript_urls(item: ET.Element, base_url: str) -> List[Tuple[str, Optional[str]]]:
    """Find all transcript URLs in an RSS item.

    Args:
        item: RSS item element
        base_url: Base URL for resolving relative URLs

    Returns:
        List of (url, type) tuples
    """
    candidates: List[Tuple[str, Optional[str]]] = []
    for el in item.iter():
        tag = el.tag
        if isinstance(tag, str) and tag.lower().endswith("transcript"):
            url_attr = el.attrib.get("url") or el.attrib.get("href")
            if url_attr:
                t = el.attrib.get("type")
                resolved_url = urljoin(base_url, url_attr.strip())
                candidates.append((resolved_url, (t.strip() if t else None)))

    for el in item.findall("transcript"):
        if el.text and el.text.strip():
            resolved_url = urljoin(base_url, el.text.strip())
            candidates.append((resolved_url, None))

    seen = set()
    unique: List[Tuple[str, Optional[str]]] = []
    for u, t in candidates:
        key = (u, t or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append((u, t))
    return unique


def find_enclosure_media(item: ET.Element, base_url: str) -> Optional[Tuple[str, Optional[str]]]:
    """Find the enclosure media URL in an RSS item.

    Args:
        item: RSS item element
        base_url: Base URL for resolving relative URLs

    Returns:
        Tuple of (url, type) or None if not found
    """
    for el in item.iter():
        if isinstance(el.tag, str) and el.tag.lower().endswith("enclosure"):
            url_attr = el.attrib.get("url")
            if url_attr:
                resolved_url = urljoin(base_url, url_attr.strip())
                return resolved_url, (el.attrib.get("type") or None)
    return None


def choose_transcript_url(
    candidates: List[Tuple[str, Optional[str]]], prefer_types: List[str]
) -> Optional[Tuple[str, Optional[str]]]:
    """Choose the best transcript URL from candidates based on preferred types.

    Args:
        candidates: List of (url, type) tuples
        prefer_types: List of preferred MIME types or file extensions

    Returns:
        Chosen (url, type) tuple or None
    """
    if not candidates:
        return None
    if not prefer_types:
        return candidates[0]

    lowered = [(u, t.lower() if t else None) for (u, t) in candidates]
    for pref in prefer_types:
        p = pref.lower().strip()
        for idx, (u, t_lower) in enumerate(lowered):
            orig_url, orig_type = candidates[idx]
            if (t_lower and p in t_lower) or orig_url.lower().endswith(p):
                return orig_url, orig_type
    return candidates[0]


def extract_episode_title(item: ET.Element, idx: int) -> Tuple[str, str]:
    """Extract episode title from RSS item and create safe filename version.

    Args:
        item: RSS item element
        idx: Episode index number

    Returns:
        Tuple of (original_title, safe_filename_title)
    """
    title_el = item.find("title") or next(
        (e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("title")), None
    )
    ep_title = title_el.text.strip() if title_el is not None and title_el.text else f"episode_{idx}"
    ep_title_safe = filesystem.sanitize_filename(ep_title)
    return ep_title, ep_title_safe


class _HTMLStripper(HTMLParser):
    """Simple HTML tag stripper that preserves spacing between text segments."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.last_was_tag = False

    def handle_data(self, data):
        """Handle text data between tags."""
        if data.strip():  # Only add non-empty text
            self.text_parts.append(data.strip())
            self.last_was_tag = False

    def handle_starttag(self, tag, attrs):
        """Handle opening tags - add space if previous was tag."""
        if self.last_was_tag and self.text_parts:
            # Ensure space between tags that might have had text between them
            pass
        self.last_was_tag = True

    def handle_endtag(self, tag):
        """Handle closing tags - mark that we just saw a tag."""
        self.last_was_tag = True

    def get_text(self):
        """Join text parts with single spaces."""
        return " ".join(self.text_parts)


def _strip_html(text: str) -> str:
    """Strip HTML tags from text and decode HTML entities, preserving spacing.

    This function sanitizes HTML content by:
    1. Decoding HTML entities (&amp; -> &, &lt; -> <, etc.)
    2. Stripping HTML tags while preserving spacing between text segments
    3. Normalizing whitespace (multiple spaces -> single space)

    Args:
        text: Text potentially containing HTML

    Returns:
        Plain text with HTML tags removed, entities decoded, and proper spacing preserved
    """
    if not text:
        return ""

    # First decode HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = unescape(text)

    # Strip HTML tags using parser that preserves spacing
    stripper = _HTMLStripper()
    try:
        stripper.feed(text)
        cleaned = stripper.get_text()
    except Exception:
        # Fallback: simple regex-based stripping if parser fails
        # Remove HTML tags but preserve spacing
        cleaned = re.sub(r"<[^>]+>", " ", text)  # Replace tags with space
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Normalize whitespace: multiple spaces/newlines/tabs -> single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_episode_description(item: ET.Element) -> Optional[str]:
    """Extract episode description from RSS item and strip HTML.

    Args:
        item: RSS item element

    Returns:
        Description text with HTML stripped, or None if not found
    """
    desc_el = item.find("description") or next(
        (e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("description")), None
    )
    if desc_el is not None:
        # Get text content - RSS descriptions often contain HTML
        desc_text = desc_el.text or ""
        # Also check for CDATA or nested text content
        if not desc_text and desc_el.itertext():
            desc_text = "".join(desc_el.itertext())

        if desc_text:
            # Strip HTML tags before returning
            cleaned = _strip_html(desc_text.strip())
            return cleaned if cleaned else None
    return None


def create_episode_from_item(item: ET.Element, idx: int, feed_base_url: str) -> models.Episode:
    """Create an Episode object from an RSS item.

    Args:
        item: RSS item element
        idx: Episode index number
        feed_base_url: Base URL for resolving relative URLs

    Returns:
        Episode object with all metadata populated
    """
    title, title_safe = extract_episode_title(item, idx)
    transcript_urls = find_transcript_urls(item, feed_base_url)
    media = find_enclosure_media(item, feed_base_url)
    media_url, media_type = media if media else (None, None)

    return models.Episode(
        idx=idx,
        title=title,
        title_safe=title_safe,
        item=item,
        transcript_urls=transcript_urls,
        media_url=media_url,
        media_type=media_type,
    )


def fetch_and_parse_rss(cfg: config.Config) -> models.RssFeed:
    """Fetch RSS feed from URL and parse it into an RssFeed object.

    Args:
        cfg: Configuration object with RSS URL and request settings

    Returns:
        An RssFeed dataclass containing feed title, items, and base URL

    Raises:
        ValueError: If fetch or parse fails
    """
    if cfg.rss_url is None:
        raise ValueError("RSS URL is required")

    resp = downloader.fetch_url(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
    if resp is None:
        raise ValueError("Failed to fetch RSS feed.")
    try:
        rss_bytes = resp.content
        feed_base_url = resp.url or cfg.rss_url
    finally:
        resp.close()

    try:
        feed_title, feed_authors, items = parse_rss_items(rss_bytes)
    except (DefusedXMLParseError, ValueError) as exc:
        raise ValueError(f"Failed to parse RSS XML: {exc}") from exc

    return models.RssFeed(
        title=feed_title, authors=feed_authors, items=items, base_url=feed_base_url
    )
