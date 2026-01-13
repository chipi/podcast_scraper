"""RSS feed parsing and episode metadata extraction."""

from __future__ import annotations

import logging
import re

# Bandit: parsing handled via defusedxml safe APIs
import xml.etree.ElementTree as ET  # nosec B405
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urljoin

from defusedxml.ElementTree import fromstring as safe_fromstring, ParseError as DefusedXMLParseError

from . import config, downloader, filesystem, models

logger = logging.getLogger(__name__)

# Time conversion constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# Duration parsing constants
DURATION_PARTS_HHMMSS = 3
DURATION_PARTS_MMSS = 2
DURATION_PARTS_SS = 1


def parse_rss_items(xml_bytes: bytes) -> Tuple[str, List[str], List[ET.Element]]:
    """Parse RSS XML and extract channel title, authors, and items.

    Args:
        xml_bytes: Raw RSS feed XML content

    Returns:
        Tuple of (channel_title, list_of_authors, list_of_items)
    """
    try:
        root = safe_fromstring(xml_bytes)
        if root is None:
            return "", [], []
        channel = root.find("channel")
        if channel is None:
            channel = next(
                (e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("channel")),
                None,
            )
        title = ""
        authors: List[str] = []
        if channel is not None:
            t = channel.find("title")
            if t is None:
                t = next(
                    (
                        e
                        for e in channel.iter()
                        if isinstance(e.tag, str) and e.tag.endswith("title")
                    ),
                    None,
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
    except Exception:
        # If parsing fails, return empty values
        # This ensures the function always returns a tuple of 3 values
        logger.debug("Failed to parse RSS XML, returning empty values", exc_info=True)
        return "", [], []


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
    title_el = item.find("title")
    if title_el is None:
        title_el = next(
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


def extract_feed_metadata(
    xml_bytes: bytes, base_url: str
) -> tuple[Optional[str], Optional[str], Optional[datetime]]:
    """Extract additional feed-level metadata from RSS XML.

    Args:
        xml_bytes: Raw RSS feed XML content
        base_url: Base URL for resolving relative URLs

    Returns:
        Tuple of (description, image_url, last_updated) where each may be None
    """
    try:
        root = safe_fromstring(xml_bytes)
        if root is None:
            return None, None, None
        channel = root.find("channel")
        if channel is None:
            channel = next(
                (e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("channel")),
                None,
            )

        if channel is None:
            return None, None, None

        description = None
        desc_elem = channel.find("description")
        if desc_elem is None:
            desc_elem = next(
                (
                    e
                    for e in channel.iter()
                    if isinstance(e.tag, str) and e.tag.endswith("description")
                ),
                None,
            )
        if desc_elem is not None and desc_elem.text:
            description = _strip_html(desc_elem.text.strip())

        image_url = None
        # RSS 2.0 image
        image_elem = channel.find("image")
        if image_elem is not None:
            url_elem = image_elem.find("url")
            if url_elem is not None and url_elem.text:
                image_url = urljoin(base_url, url_elem.text.strip())
        # iTunes image
        if not image_url:
            itunes_image_elem = channel.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
            if itunes_image_elem is not None:
                href = itunes_image_elem.attrib.get("href")
                if href:
                    image_url = urljoin(base_url, href.strip())

        last_updated = None
        # RSS 2.0 lastBuildDate
        last_build_elem = channel.find("lastBuildDate")
        if last_build_elem is not None and last_build_elem.text:
            try:
                from email.utils import parsedate_to_datetime

                date_tuple = parsedate_to_datetime(last_build_elem.text.strip())
                if date_tuple:
                    last_updated = date_tuple
            # Intentional fallback for date parsing
            except Exception:  # nosec B110
                pass
        # Atom updated
        if not last_updated:
            atom_updated_elem = channel.find("{http://www.w3.org/2005/Atom}updated")
            if atom_updated_elem is not None and atom_updated_elem.text:
                try:
                    from datetime import datetime

                    last_updated = datetime.fromisoformat(
                        atom_updated_elem.text.strip().replace("Z", "+00:00")
                    )
                # Intentional fallback for date parsing
                except Exception:  # nosec B110
                    pass

        return description, image_url, last_updated
    except Exception:
        # If any part of extraction fails, return None values
        # This ensures the function always returns a tuple of 3 values
        return None, None, None


def _extract_duration_seconds(item: ET.Element) -> Optional[int]:
    """Extract episode duration in seconds from iTunes duration element.

    Args:
        item: RSS item element

    Returns:
        Duration in seconds, or None if not found or invalid
    """
    itunes_duration_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
    if itunes_duration_elem is None or not itunes_duration_elem.text:
        return None

    duration_str = itunes_duration_elem.text.strip()
    try:
        parts = duration_str.split(":")
        if len(parts) == DURATION_PARTS_HHMMSS:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds
        elif len(parts) == DURATION_PARTS_MMSS:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * SECONDS_PER_MINUTE + seconds
        elif len(parts) == DURATION_PARTS_SS:  # SS
            return int(parts[0])
    except (ValueError, IndexError):
        # Invalid duration format (e.g., non-numeric, malformed string)
        # Return None as documented in function signature
        pass
    return None


def _extract_episode_number(item: ET.Element) -> Optional[int]:
    """Extract episode number from iTunes episode element.

    Args:
        item: RSS item element

    Returns:
        Episode number, or None if not found or invalid
    """
    itunes_episode_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}episode")
    if itunes_episode_elem is None or not itunes_episode_elem.text:
        return None

    try:
        return int(itunes_episode_elem.text.strip())
    except ValueError:
        return None


def _extract_image_url(item: ET.Element, base_url: str) -> Optional[str]:
    """Extract episode image URL from iTunes image element.

    Args:
        item: RSS item element
        base_url: Base URL for resolving relative URLs

    Returns:
        Image URL, or None if not found
    """
    itunes_image_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
    if itunes_image_elem is None:
        return None

    href = itunes_image_elem.attrib.get("href")
    if href:
        return urljoin(base_url, href.strip())
    return None


def extract_episode_metadata(
    item: ET.Element, base_url: str
) -> tuple[
    Optional[str], Optional[str], Optional[str], Optional[int], Optional[int], Optional[str]
]:
    """Extract additional episode-level metadata from RSS item.

    Args:
        item: RSS item element
        base_url: Base URL for resolving relative URLs

    Returns:
        Tuple of (description, guid, link, duration_seconds, episode_number, image_url)
        where each may be None
    """
    description = None
    desc_elem = item.find("description")
    if desc_elem is None:
        desc_elem = item.find("summary")
    if desc_elem is None:
        desc_elem = next(
            (e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("description")),
            None,
        )
    if desc_elem is None:
        desc_elem = next(
            (e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("summary")), None
        )
    if desc_elem is not None and desc_elem.text:
        description = _strip_html(desc_elem.text.strip())

    guid = None
    guid_elem = item.find("guid")
    if guid_elem is not None:
        guid = guid_elem.text.strip() if guid_elem.text else None
        # Some feeds use guid as attribute
        if not guid:
            guid = guid_elem.attrib.get("isPermaLink")
            if guid == "false" and guid_elem.text:
                guid = guid_elem.text.strip()

    link = None
    link_elem = item.find("link")
    if link_elem is not None:
        if link_elem.text:
            link = urljoin(base_url, link_elem.text.strip())
        else:
            href = link_elem.attrib.get("href")
            if href:
                link = urljoin(base_url, href.strip())

    duration_seconds = _extract_duration_seconds(item)
    episode_number = _extract_episode_number(item)
    image_url = _extract_image_url(item, base_url)

    return description, guid, link, duration_seconds, episode_number, image_url


def extract_episode_published_date(item: ET.Element) -> Optional[datetime]:
    """Extract published date from RSS item.

    Args:
        item: RSS item element

    Returns:
        Parsed datetime object or None if not available
    """
    from datetime import datetime

    # RSS 2.0 pubDate
    pub_date_elem = item.find("pubDate")
    if pub_date_elem is not None and pub_date_elem.text:
        try:
            from email.utils import parsedate_to_datetime

            date_tuple = parsedate_to_datetime(pub_date_elem.text.strip())
            if date_tuple:
                return date_tuple
        # Intentional fallback for date parsing
        except Exception:  # nosec B110
            pass

    # Atom published
    atom_published_elem = item.find("{http://www.w3.org/2005/Atom}published")
    if atom_published_elem is not None and atom_published_elem.text:
        try:
            return datetime.fromisoformat(atom_published_elem.text.strip().replace("Z", "+00:00"))
        # Intentional fallback for date parsing
        except Exception:  # nosec B110
            pass

    # Atom updated (fallback)
    atom_updated_elem = item.find("{http://www.w3.org/2005/Atom}updated")
    if atom_updated_elem is not None and atom_updated_elem.text:
        try:
            return datetime.fromisoformat(atom_updated_elem.text.strip().replace("Z", "+00:00"))
        # Intentional fallback for date parsing
        except Exception:  # nosec B110
            pass

    return None


def extract_episode_description(item: ET.Element) -> Optional[str]:
    """Extract episode description from RSS item and strip HTML.

    Args:
        item: RSS item element

    Returns:
        Description text with HTML stripped, or None if not found
    """
    desc_el = item.find("description")
    if desc_el is None:
        desc_el = next(
            (e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("description")),
            None,
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
