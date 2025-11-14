"""Metadata generation for podcast episodes.

This module implements per-episode metadata document generation as per PRD-004 and RFC-011.
Metadata documents are structured JSON/YAML files that capture comprehensive feed and
episode information for search, analytics, integration, and archival use cases.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_serializer

from . import config, filesystem, models

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


def generate_feed_id(feed_url: str) -> str:
    """Generate stable unique identifier for feed.

    Args:
        feed_url: RSS feed URL

    Returns:
        Stable unique identifier string (format: sha256:<hex_digest>)
    """
    # Normalize feed URL (remove trailing slash, lowercase, remove query params/fragments)
    parsed = urlparse(feed_url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/").lower()

    # Generate SHA-256 hash
    hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return f"sha256:{hash_digest}"


def generate_episode_id(
    feed_url: str,
    episode_title: str,
    episode_guid: Optional[str] = None,
    published_date: Optional[datetime] = None,
    episode_link: Optional[str] = None,
    episode_number: Optional[int] = None,
) -> str:
    """Generate stable unique identifier for episode.

    Priority:
    1. RSS GUID if available
    2. Deterministic hash from feed URL + title + published_date + link
    3. Composite key as last resort

    Args:
        feed_url: RSS feed URL
        episode_title: Episode title
        episode_guid: RSS GUID if available
        published_date: Episode published date
        episode_link: Episode link/URL
        episode_number: Episode number/sequence

    Returns:
        Stable unique identifier string
    """
    # Priority 1: Use RSS GUID if available
    if episode_guid:
        return episode_guid.strip()

    # Priority 2: Generate deterministic hash
    # Normalize feed URL (remove trailing slash, lowercase)
    normalized_feed = urlparse(feed_url).geturl().rstrip("/").lower()

    # Normalize title (lowercase, strip whitespace)
    normalized_title = episode_title.strip().lower()

    # Build hash input from stable components
    hash_components = [
        normalized_feed,
        normalized_title,
    ]

    if published_date:
        hash_components.append(published_date.isoformat())

    if episode_link:
        normalized_link = urlparse(episode_link).geturl().rstrip("/").lower()
        hash_components.append(normalized_link)

    # Generate SHA-256 hash
    hash_input = "|".join(hash_components).encode("utf-8")
    hash_digest = hashlib.sha256(hash_input).hexdigest()

    return f"sha256:{hash_digest}"


def generate_content_id(content_url: str) -> str:
    """Generate stable unique identifier for content item (transcript or media).

    Args:
        content_url: URL of the content item

    Returns:
        Stable unique identifier string (format: sha256:<hex_digest>)
    """
    # Normalize URL (remove trailing slash, lowercase, remove query params/fragments)
    parsed = urlparse(content_url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/").lower()

    # Generate SHA-256 hash
    hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return f"sha256:{hash_digest}"


class FeedMetadata(BaseModel):
    """Feed-level metadata."""

    title: str
    url: str
    feed_id: str  # Stable unique identifier for database primary keys
    description: Optional[str] = None
    language: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    last_updated: Optional[datetime] = None

    @field_serializer("last_updated")
    def serialize_last_updated(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime as ISO 8601 string for database compatibility."""
        return value.isoformat() if value else None


class EpisodeMetadata(BaseModel):
    """Episode-level metadata."""

    title: str
    description: Optional[str] = None
    published_date: Optional[datetime] = None
    guid: Optional[str] = None  # RSS GUID if available
    link: Optional[str] = None
    duration_seconds: Optional[int] = None
    episode_number: Optional[int] = None
    image_url: Optional[str] = None
    episode_id: str  # Stable unique identifier for database primary keys

    @field_serializer("published_date")
    def serialize_published_date(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime as ISO 8601 string for database compatibility."""
        return value.isoformat() if value else None


class TranscriptInfo(BaseModel):
    """Transcript URL and type information."""

    url: str
    transcript_id: Optional[str] = (
        None  # Optional stable identifier for tracking individual transcripts
    )
    type: Optional[str] = None  # e.g., "text/plain", "text/vtt"
    language: Optional[str] = None


class ContentMetadata(BaseModel):
    """Content-related metadata."""

    transcript_urls: List[TranscriptInfo] = Field(default_factory=list)
    media_url: Optional[str] = None
    media_id: Optional[str] = None  # Optional stable identifier for media file
    media_type: Optional[str] = None
    transcript_file_path: Optional[str] = None
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]] = None
    whisper_model: Optional[str] = None
    detected_hosts: List[str] = Field(default_factory=list)
    detected_guests: List[str] = Field(default_factory=list)


class ProcessingMetadata(BaseModel):
    """Processing-related metadata."""

    processing_timestamp: datetime
    output_directory: str
    run_id: Optional[str] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @field_serializer("processing_timestamp")
    def serialize_processing_timestamp(self, value: datetime) -> str:
        """Serialize datetime as ISO 8601 string for database compatibility."""
        return value.isoformat()


class EpisodeMetadataDocument(BaseModel):
    """Complete episode metadata document.

    Schema designed for direct ingestion into databases:
    - PostgreSQL JSONB: Nested structure works natively
    - MongoDB: Document structure matches MongoDB document model
    - Elasticsearch: Nested objects can be indexed and queried
    - ClickHouse: JSON column type supports nested queries

    Field naming uses snake_case for database compatibility.
    All datetime fields are serialized as ISO 8601 strings.

    The `feed.feed_id` and `episode.episode_id` fields provide stable, unique identifiers
    suitable for use as primary keys in all target databases. Optional `transcript_id` and
    `media_id` fields enable tracking individual content items separately if needed.
    """

    feed: FeedMetadata
    episode: EpisodeMetadata
    content: ContentMetadata
    processing: ProcessingMetadata


def _build_feed_metadata(
    feed: models.RssFeed,
    feed_url: str,
    feed_id: str,
    cfg: config.Config,
    feed_description: Optional[str],
    feed_image_url: Optional[str],
    feed_last_updated: Optional[datetime],
) -> FeedMetadata:
    """Build FeedMetadata object.

    Args:
        feed: RssFeed object
        feed_url: RSS feed URL
        feed_id: Generated feed ID
        cfg: Configuration object
        feed_description: Feed description
        feed_image_url: Feed image URL
        feed_last_updated: Feed last updated date

    Returns:
        FeedMetadata object
    """
    return FeedMetadata(
        title=feed.title,
        url=feed_url,
        feed_id=feed_id,
        description=feed_description,
        language=cfg.language,
        authors=feed.authors if feed.authors else [],
        image_url=feed_image_url,
        last_updated=feed_last_updated,
    )


def _build_episode_metadata(
    episode: models.Episode,
    episode_id: str,
    episode_description: Optional[str],
    episode_published_date: Optional[datetime],
    episode_guid: Optional[str],
    episode_link: Optional[str],
    episode_duration_seconds: Optional[int],
    episode_number: Optional[int],
    episode_image_url: Optional[str],
) -> EpisodeMetadata:
    """Build EpisodeMetadata object.

    Args:
        episode: Episode object
        episode_id: Generated episode ID
        episode_description: Episode description
        episode_published_date: Episode published date
        episode_guid: Episode GUID
        episode_link: Episode link
        episode_duration_seconds: Episode duration
        episode_number: Episode number
        episode_image_url: Episode image URL

    Returns:
        EpisodeMetadata object
    """
    return EpisodeMetadata(
        title=episode.title,
        description=episode_description,
        published_date=episode_published_date,
        guid=episode_guid,
        link=episode_link,
        duration_seconds=episode_duration_seconds,
        episode_number=episode_number,
        image_url=episode_image_url,
        episode_id=episode_id,
    )


def _build_content_metadata(
    episode: models.Episode,
    transcript_infos: List[TranscriptInfo],
    media_id: Optional[str],
    transcript_file_path: Optional[str],
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]],
    whisper_model: Optional[str],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
) -> ContentMetadata:
    """Build ContentMetadata object.

    Args:
        episode: Episode object
        transcript_infos: List of transcript info objects
        media_id: Generated media ID
        transcript_file_path: Path to transcript file
        transcript_source: Source of transcript
        whisper_model: Whisper model used
        detected_hosts: List of detected hosts
        detected_guests: List of detected guests

    Returns:
        ContentMetadata object
    """
    return ContentMetadata(
        transcript_urls=transcript_infos,
        media_url=episode.media_url,
        media_id=media_id,
        media_type=episode.media_type,
        transcript_file_path=transcript_file_path,
        transcript_source=transcript_source,
        whisper_model=whisper_model,
        detected_hosts=detected_hosts if detected_hosts else [],
        detected_guests=detected_guests if detected_guests else [],
    )


def _build_processing_metadata(cfg: config.Config, output_dir: str) -> ProcessingMetadata:
    """Build ProcessingMetadata object.

    Args:
        cfg: Configuration object
        output_dir: Output directory path

    Returns:
        ProcessingMetadata object
    """
    config_snapshot = {
        "language": cfg.language,
        "whisper_model": cfg.whisper_model if cfg.transcribe_missing else None,
        "auto_speakers": cfg.auto_speakers,
        "screenplay": cfg.screenplay,
        "max_episodes": cfg.max_episodes,
    }

    return ProcessingMetadata(
        processing_timestamp=datetime.now(),
        output_directory=output_dir,
        run_id=cfg.run_id,
        config_snapshot=config_snapshot,
        schema_version=SCHEMA_VERSION,
    )


def _determine_metadata_path(
    episode: models.Episode,
    output_dir: str,
    run_suffix: Optional[str],
    cfg: config.Config,
) -> str:
    """Determine the output path for metadata file.

    Args:
        episode: Episode object
        output_dir: Output directory path
        run_suffix: Optional run suffix
        cfg: Configuration object

    Returns:
        Full path to metadata file
    """
    base_name = filesystem.build_whisper_output_name(
        episode.idx, episode.title_safe, run_suffix
    ).replace(".txt", "")
    extension = ".json" if cfg.metadata_format == "json" else ".yaml"

    if cfg.metadata_subdirectory:
        metadata_dir = os.path.join(output_dir, cfg.metadata_subdirectory)
        return os.path.join(metadata_dir, f"{base_name}.metadata{extension}")
    else:
        return os.path.join(output_dir, f"{base_name}.metadata{extension}")


def _serialize_metadata(
    metadata_doc: EpisodeMetadataDocument, metadata_path: str, cfg: config.Config
) -> None:
    """Serialize and write metadata document to file.

    Args:
        metadata_doc: Metadata document to serialize
        metadata_path: Path to output file
        cfg: Configuration object

    Raises:
        OSError: If file writing fails
    """
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)

    if cfg.metadata_format == "json":
        content_str = metadata_doc.model_dump_json(indent=2, exclude_none=False)
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(content_str)
    else:  # yaml
        content_dict = metadata_doc.model_dump(exclude_none=False)
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(
                content_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )


def generate_episode_metadata(
    feed: models.RssFeed,
    episode: models.Episode,
    feed_url: str,
    cfg: config.Config,
    output_dir: str,
    run_suffix: Optional[str] = None,
    transcript_file_path: Optional[str] = None,
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]] = None,
    whisper_model: Optional[str] = None,
    detected_hosts: Optional[List[str]] = None,
    detected_guests: Optional[List[str]] = None,
    feed_description: Optional[str] = None,
    feed_image_url: Optional[str] = None,
    feed_last_updated: Optional[datetime] = None,
    episode_description: Optional[str] = None,
    episode_published_date: Optional[datetime] = None,
    episode_guid: Optional[str] = None,
    episode_link: Optional[str] = None,
    episode_duration_seconds: Optional[int] = None,
    episode_number: Optional[int] = None,
    episode_image_url: Optional[str] = None,
) -> Optional[str]:
    """Generate metadata document for an episode.

    Args:
        feed: RssFeed object with feed metadata
        episode: Episode object with episode metadata
        feed_url: RSS feed URL
        cfg: Configuration object
        output_dir: Output directory path
        run_suffix: Optional run suffix for filename
        transcript_file_path: Path to transcript file (relative to output_dir)
        transcript_source: Source of transcript ("direct_download" or "whisper_transcription")
        whisper_model: Whisper model used (if applicable)
        detected_hosts: List of detected host names
        detected_guests: List of detected guest names
        feed_description: Feed description (if available)
        feed_image_url: Feed image/logo URL (if available)
        feed_last_updated: Feed last updated date (if available)
        episode_description: Episode description (if available)
        episode_published_date: Episode published date (if available)
        episode_guid: Episode GUID (if available)
        episode_link: Episode link/URL (if available)
        episode_duration_seconds: Episode duration in seconds (if available)
        episode_number: Episode number/sequence (if available)
        episode_image_url: Episode image/artwork URL (if available)

    Returns:
        Path to generated metadata file, or None if generation skipped
    """
    if not cfg.generate_metadata:
        return None

    # Generate IDs
    feed_id = generate_feed_id(feed_url)
    episode_id = generate_episode_id(
        feed_url=feed_url,
        episode_title=episode.title,
        episode_guid=episode_guid,
        published_date=episode_published_date,
        episode_link=episode_link,
        episode_number=episode_number,
    )

    # Build transcript URLs with IDs
    transcript_infos = []
    for url, transcript_type in episode.transcript_urls:
        transcript_id = generate_content_id(url) if url else None
        transcript_infos.append(
            TranscriptInfo(
                url=url,
                transcript_id=transcript_id,
                type=transcript_type,
                language=cfg.language if cfg.language else None,
            )
        )

    # Generate media ID if media URL exists
    media_id = generate_content_id(episode.media_url) if episode.media_url else None

    # Build metadata objects
    feed_metadata = _build_feed_metadata(
        feed, feed_url, feed_id, cfg, feed_description, feed_image_url, feed_last_updated
    )
    episode_metadata = _build_episode_metadata(
        episode,
        episode_id,
        episode_description,
        episode_published_date,
        episode_guid,
        episode_link,
        episode_duration_seconds,
        episode_number,
        episode_image_url,
    )
    content_metadata = _build_content_metadata(
        episode,
        transcript_infos,
        media_id,
        transcript_file_path,
        transcript_source,
        whisper_model,
        detected_hosts,
        detected_guests,
    )
    processing_metadata = _build_processing_metadata(cfg, output_dir)

    # Build complete metadata document
    metadata_doc = EpisodeMetadataDocument(
        feed=feed_metadata,
        episode=episode_metadata,
        content=content_metadata,
        processing=processing_metadata,
    )

    # Determine output path
    metadata_path = _determine_metadata_path(episode, output_dir, run_suffix, cfg)

    # Check skip_existing
    # Allow overwriting when transcript_source is "whisper_transcription" (complete transcript data)
    # This fixes the case where metadata was created with transcript_source=None (Whisper pending)
    # and needs to be updated with actual transcript information after transcription completes
    if cfg.skip_existing and os.path.exists(metadata_path):
        if transcript_source != "whisper_transcription":
            logger.debug(
                "[%s] Metadata file already exists; skipping (--skip-existing): %s",
                episode.idx,
                metadata_path,
            )
            return None
        # Allow overwriting for whisper_transcription to update incomplete metadata
        logger.debug(
            "[%s] Metadata file exists but will be overwritten with complete transcript data: %s",
            episode.idx,
            metadata_path,
        )

    # Handle dry-run mode
    if cfg.dry_run:
        logger.info(
            "[%s] Dry-run: would generate metadata file: %s",
            episode.idx,
            metadata_path,
        )
        return metadata_path

    # Serialize and write metadata
    try:
        _serialize_metadata(metadata_doc, metadata_path, cfg)
        logger.debug("[%s] Generated metadata file: %s", episode.idx, metadata_path)
        return metadata_path

    except Exception as exc:
        logger.warning(
            "[%s] Failed to generate metadata file %s: %s",
            episode.idx,
            metadata_path,
            exc,
            exc_info=True,
        )
        return None
