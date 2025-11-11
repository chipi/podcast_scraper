"""RSS feed parsing and episode metadata extraction."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from urllib.parse import urljoin

from defusedxml.ElementTree import ParseError as DefusedXMLParseError, fromstring as safe_fromstring

from . import config, downloader, filesystem, models

logger = logging.getLogger(__name__)


def parse_rss_items(xml_bytes: bytes) -> Tuple[str, List[ET.Element]]:
    """Parse RSS XML and extract channel title and items.
    
    Args:
        xml_bytes: Raw RSS feed XML content
        
    Returns:
        Tuple of (channel_title, list_of_items)
    """
    root = safe_fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        channel = next((e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("channel")), None)
    title = ""
    if channel is not None:
        t = channel.find("title") or next((e for e in channel.iter() if isinstance(e.tag, str) and e.tag.endswith("title")), None)
        if t is not None and t.text:
            title = t.text.strip()
        items = list(channel.findall("item"))
        if not items:
            items = [e for e in channel if isinstance(e.tag, str) and e.tag.endswith("item")]
    else:
        items = [e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("item")]
    return title, items


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


def choose_transcript_url(candidates: List[Tuple[str, Optional[str]]], prefer_types: List[str]) -> Optional[Tuple[str, Optional[str]]]:
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
    title_el = item.find("title") or next((e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("title")), None)
    ep_title = (title_el.text.strip() if title_el is not None and title_el.text else f"episode_{idx}")
    ep_title_safe = filesystem.sanitize_filename(ep_title)
    return ep_title, ep_title_safe


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
    resp = downloader.fetch_url(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
    if resp is None:
        raise ValueError("Failed to fetch RSS feed.")
    try:
        rss_bytes = resp.content
        feed_base_url = resp.url or cfg.rss_url
    finally:
        resp.close()

    try:
        feed_title, items = parse_rss_items(rss_bytes)
    except (DefusedXMLParseError, ValueError) as exc:
        raise ValueError(f"Failed to parse RSS XML: {exc}") from exc

    return models.RssFeed(title=feed_title, items=items, base_url=feed_base_url)

