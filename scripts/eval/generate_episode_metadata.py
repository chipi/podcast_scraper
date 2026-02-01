#!/usr/bin/env python3
"""Generate episode metadata JSON files from RSS XML files.

This script:
- Scans a directory for RSS XML files
- Parses each RSS feed and extracts episode metadata
- Generates a metadata JSON file for each episode
- Saves JSON files next to the XML file or in a specified output directory

Usage:
    python scripts/generate_episode_metadata.py --input-dir data/eval/sources
        --output-dir data/eval/metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from defusedxml.ElementTree import fromstring as safe_fromstring, ParseError as DefusedXMLParseError

from podcast_scraper.rss_parser import (
    extract_episode_metadata,
    extract_episode_published_date,
    parse_rss_items,
)

logger = logging.getLogger(__name__)


def parse_duration_to_seconds(duration_str: str) -> Optional[int]:
    """Parse duration string (HH:MM:SS or MM:SS) to seconds.

    Args:
        duration_str: Duration string (e.g., "00:10:30" or "10:30")

    Returns:
        Duration in seconds or None if parsing fails
    """
    if not duration_str:
        return None

    try:
        parts = duration_str.strip().split(":")
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 1:  # SS
            return int(parts[0])
    except (ValueError, AttributeError):
        pass

    return None


def extract_language_from_channel(channel_elem) -> Optional[str]:
    """Extract language from RSS channel.

    Args:
        channel_elem: Channel XML element

    Returns:
        Language code (e.g., "en") or None
    """
    if channel_elem is None:
        return None

    # Try RSS 2.0 language
    lang_elem = channel_elem.find("language")
    if lang_elem is not None and lang_elem.text:
        lang = lang_elem.text.strip()
        # Normalize: "en-us" -> "en"
        if "-" in lang:
            lang = lang.split("-")[0]
        return lang.lower()

    return None


def extract_feed_url_from_channel(channel_elem, base_url: str) -> Optional[str]:
    """Extract feed URL from RSS channel.

    Args:
        channel_elem: Channel XML element
        base_url: Base URL for resolving relative URLs

    Returns:
        Feed URL or None
    """
    if channel_elem is None:
        return None

    # Try link element
    link_elem = channel_elem.find("link")
    if link_elem is not None and link_elem.text:
        return link_elem.text.strip()

    # Fallback to base_url
    return base_url if base_url else None


def process_rss_file(xml_path: Path, output_dir: Optional[Path] = None) -> int:  # noqa: C901
    """Process a single RSS XML file and generate metadata JSON files.

    Args:
        xml_path: Path to RSS XML file
        output_dir: Optional output directory (if None, saves next to XML file)

    Returns:
        Number of episodes processed
    """
    logger.info(f"Processing RSS file: {xml_path}")

    # Read XML file
    try:
        xml_bytes = xml_path.read_bytes()
    except Exception as e:
        logger.error(f"Failed to read {xml_path}: {e}")
        return 0

    # Parse RSS
    try:
        root = safe_fromstring(xml_bytes)
        if root is None:
            logger.error(f"Failed to parse XML: {xml_path}")
            return 0

        channel = root.find("channel")
        if channel is None:
            # Try to find channel with namespace
            channel = next(
                (e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("channel")),
                None,
            )

        if channel is None:
            logger.error(f"No channel element found in {xml_path}")
            return 0

        # Extract feed-level metadata
        feed_title = None
        title_elem = channel.find("title")
        if title_elem is not None and title_elem.text:
            feed_title = title_elem.text.strip()

        feed_url = extract_feed_url_from_channel(channel, str(xml_path))
        language = extract_language_from_channel(channel) or "en"

        # Parse RSS items
        try:
            feed_title_parsed, _, items = parse_rss_items(xml_bytes)
            if feed_title_parsed and not feed_title:
                feed_title = feed_title_parsed
        except Exception as e:
            logger.warning(f"Failed to parse RSS items with parse_rss_items: {e}")
            # Fallback: find items directly
            items = channel.findall("item")
            if not items:
                items = list(channel.iter())
                items = [item for item in items if item.tag.endswith("item")]

        if not items:
            logger.warning(f"No items found in {xml_path}")
            return 0

        # Determine output directory
        if output_dir:
            output_base = output_dir
        else:
            output_base = xml_path.parent

        # Process each episode/item
        processed = 0
        scraped_at = datetime.utcnow().isoformat() + "Z"

        for item in items:
            # Extract episode metadata
            title_elem = item.find("title")
            if title_elem is None:
                title_elem = next(
                    (e for e in item.iter() if isinstance(e.tag, str) and e.tag.endswith("title")),
                    None,
                )

            episode_title = None
            if title_elem is not None and title_elem.text:
                episode_title = title_elem.text.strip()

            if not episode_title:
                logger.warning(f"Skipping item without title in {xml_path}")
                continue

            # Extract GUID/source_episode_id
            guid_elem = item.find("guid")
            source_episode_id = None
            if guid_elem is not None:
                if guid_elem.text:
                    source_episode_id = guid_elem.text.strip()
                elif guid_elem.attrib.get("isPermaLink") == "false" and guid_elem.text:
                    source_episode_id = guid_elem.text.strip()

            # If no GUID, try to derive from title or use index
            if not source_episode_id:
                # Try to create a simple ID from title
                source_episode_id = episode_title.lower().replace(" ", "_")[:50]
                # Add index if needed for uniqueness
                if processed > 0:
                    source_episode_id = f"{source_episode_id}_{processed}"

            # Extract published date
            published_at = None
            pub_date = extract_episode_published_date(item)
            if pub_date:
                # Format as YYYY-MM-DD
                published_at = pub_date.strftime("%Y-%m-%d")

            # Extract duration
            duration_seconds = None
            duration_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
            if duration_elem is None:
                # Try without namespace
                duration_elem = next(
                    (
                        e
                        for e in item.iter()
                        if isinstance(e.tag, str) and "duration" in e.tag.lower()
                    ),
                    None,
                )

            if duration_elem is not None and duration_elem.text:
                duration_seconds = parse_duration_to_seconds(duration_elem.text)

            # If duration not found, try using extract_episode_metadata
            if duration_seconds is None:
                try:
                    _, _, _, duration_seconds, _, _ = extract_episode_metadata(item, str(xml_path))
                except Exception:
                    pass

            # Create metadata JSON (metadata_version first for schema versioning)
            metadata = {
                "metadata_version": "1.0",
                "source_episode_id": source_episode_id,
                "feed_name": feed_title or "Unknown Feed",
                "feed_url": feed_url or "",
                "episode_title": episode_title,
                "published_at": published_at,
                "duration_seconds": duration_seconds,
                "language": language,
                "scraped_at": scraped_at,
                "speakers": [
                    {"id": "host", "name": "TODO: Add host name", "role": "host"},
                    {"id": "guest", "name": "TODO: Add guest name", "role": "guest"},
                ],
                "expectations": {
                    "allow_speaker_names": False,
                    "allow_speaker_labels": False,
                    "allow_sponsor_content": False,
                },
            }

            # Remove None values (but keep speakers and expectations)
            metadata = {
                k: v
                for k, v in metadata.items()
                if v is not None or k in ("speakers", "expectations")
            }

            # Generate output filename
            # Use source_episode_id as base, sanitize for filesystem
            safe_id = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in source_episode_id
            )
            output_file = output_base / f"{safe_id}.metadata.json"

            # Save JSON file
            try:
                output_file.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                logger.info(f"Created metadata: {output_file.name} for episode: {episode_title}")
                processed += 1
            except Exception as e:
                logger.error(f"Failed to write {output_file}: {e}")

        return processed

    except DefusedXMLParseError as e:
        logger.error(f"XML parsing error in {xml_path}: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error processing {xml_path}: {e}", exc_info=True)
        return 0


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate episode metadata JSON files from RSS XML files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing RSS XML files (will search recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for metadata JSON files (default: same as XML file location)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Find all XML files
    xml_files = list(input_dir.rglob("*.xml"))
    if not xml_files:
        logger.warning(f"No XML files found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(xml_files)} XML file(s)")

    # Process each XML file
    total_episodes = 0
    for xml_file in sorted(xml_files):
        episodes = process_rss_file(xml_file, output_dir)
        total_episodes += episodes

    logger.info(
        f"Processed {len(xml_files)} RSS file(s), "
        f"generated metadata for {total_episodes} episode(s)"
    )


if __name__ == "__main__":
    main()
