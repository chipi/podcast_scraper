"""Metadata generation for podcast episodes.

This module implements per-episode metadata document generation as per PRD-004 and RFC-011.
Metadata documents are structured JSON/YAML files that capture comprehensive feed and
episode information for search, analytics, integration, and archival use cases.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, computed_field, Field, field_serializer

from .. import config, models
from ..exceptions import ProviderRuntimeError
from ..utils import filesystem
from ..utils.timeout import TimeoutError

logger = logging.getLogger(__name__)

# Lazy import for summarization (optional dependency)
# Import is deferred until actually needed to avoid PyTorch initialization in dry-run mode
summarizer = None  # type: ignore

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


class SpeakerInfo(BaseModel):
    """Speaker information with structured role and identity."""

    id: str  # Stable identifier: "host", "guest", "host_1", "guest_1", etc.
    name: str  # Speaker name (e.g., "Alice Johnson", "Bob Smith")
    role: str  # Role: "host" or "guest"


class ExpectationsMetadata(BaseModel):
    """Expectations about output quality (not facts about the episode)."""

    allow_speaker_names: bool = Field(
        default=False,
        description="Whether speaker names are allowed in summaries",
    )
    allow_speaker_labels: bool = Field(
        default=False,
        description="Whether speaker labels (Name:) are allowed in summaries",
    )
    allow_sponsor_content: bool = Field(
        default=False,
        description="Whether sponsor/advertisement content is allowed in summaries",
    )


class EntityCorrection(BaseModel):
    """Information about an entity correction made to summary text."""

    original: str = Field(description="Original entity text from summary")
    corrected: str = Field(description="Corrected entity text (from extracted entities)")
    edit_distance: int = Field(description="Edit distance between original and corrected")


class QAFlags(BaseModel):
    """Quality assurance flags for detecting silent failures and quality issues.

    These flags make quality regressions explicit so downstream systems can
    filter or flag low-quality outputs.
    """

    speaker_detection: Literal["none", "partial", "ok"] = Field(
        default="none",
        description=(
            "Speaker detection quality: 'none' (no speakers detected), "
            "'partial' (some speakers detected), 'ok' (all speakers detected)"
        ),
    )
    defaults_injected: bool = Field(
        default=False,
        description="Whether default speaker names (Host/Guest) were injected",
    )
    summary_entity_mismatch: bool = Field(
        default=False,
        description=(
            "Whether summary contains entity names that don't match extracted entities "
            "(e.g., 'Kevin Walsh' in summary vs 'Kevin Warsh' in metadata)"
        ),
    )
    summary_has_named_entities: bool = Field(
        default=False,
        description="Whether summary contains any named entities (PERSON, ORG, etc.)",
    )
    corrected_entities: List[EntityCorrection] = Field(
        default_factory=list,
        description="List of entity corrections applied to summary text",
    )


class ContentMetadata(BaseModel):
    """Content-related metadata."""

    transcript_urls: List[TranscriptInfo] = Field(default_factory=list)
    media_url: Optional[str] = None
    media_id: Optional[str] = None  # Optional stable identifier for media file
    media_type: Optional[str] = None
    transcript_file_path: Optional[str] = None
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]] = None
    whisper_model: Optional[str] = None
    speakers: List[SpeakerInfo] = Field(
        default_factory=list
    )  # Structured speaker information (facts)
    expectations: Optional[ExpectationsMetadata] = Field(
        default=None,
        description="Expectations about output quality (not facts about the episode)",
    )
    qa_flags: Optional[QAFlags] = Field(
        default=None,
        description="Quality assurance flags for detecting silent failures",
    )

    @computed_field
    def detected_hosts(self) -> List[str]:
        """Backward compatibility: Extract host names from speakers list."""
        return [speaker.name for speaker in self.speakers if speaker.role == "host"]

    @computed_field
    def detected_guests(self) -> List[str]:
        """Backward compatibility: Extract guest names from speakers list."""
        return [speaker.name for speaker in self.speakers if speaker.role == "guest"]


class SummaryMetadata(BaseModel):
    """Summary metadata.

    Note: Provider/model information is available in processing.config_snapshot.ml_providers
    to avoid duplication and keep all ML configuration in one place.
    """

    short_summary: str
    generated_at: datetime
    word_count: Optional[int] = None

    @field_serializer("generated_at")
    def serialize_generated_at(self, value: datetime) -> str:
        """Serialize datetime as ISO 8601 string for database compatibility."""
        return value.isoformat()


class EpisodeStageTimings(BaseModel):
    """Per-episode stage timings (Issue #379)."""

    download_seconds: Optional[float] = None
    transcription_seconds: Optional[float] = None
    speaker_detection_seconds: Optional[float] = None
    summarization_seconds: Optional[float] = None
    metadata_generation_seconds: Optional[float] = None
    total_processing_seconds: Optional[float] = None


class ProcessingMetadata(BaseModel):
    """Processing-related metadata."""

    processing_timestamp: datetime
    output_directory: str
    run_id: Optional[str] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    stage_timings: Optional[EpisodeStageTimings] = Field(
        default=None,
        description="Per-episode stage timings for performance analysis (Issue #379)",
    )

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
    summary: Optional[SummaryMetadata] = None


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


def _build_speakers_from_detected_names(
    detected_hosts: Optional[List[str]], detected_guests: Optional[List[str]]
) -> List[SpeakerInfo]:
    """Build structured speakers array from detected_hosts and detected_guests.

    Args:
        detected_hosts: List of detected host names
        detected_guests: List of detected guest names

    Returns:
        List of SpeakerInfo objects
    """
    speakers: List[SpeakerInfo] = []

    # Add hosts
    if detected_hosts:
        for idx, host_name in enumerate(detected_hosts):
            speaker_id = "host" if len(detected_hosts) == 1 else f"host_{idx + 1}"
            speakers.append(SpeakerInfo(id=speaker_id, name=host_name, role="host"))

    # Add guests
    if detected_guests:
        for idx, guest_name in enumerate(detected_guests):
            speaker_id = "guest" if len(detected_guests) == 1 else f"guest_{idx + 1}"
            speakers.append(SpeakerInfo(id=speaker_id, name=guest_name, role="guest"))

    return speakers


def _calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of character changes needed)
    """
    if len(s1) < len(s2):
        return _calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _check_entity_consistency(
    extracted_entities: List[str], summary_text: Optional[str], nlp: Optional[Any] = None
) -> Tuple[bool, bool]:
    """Check entity consistency between extracted entities and summary.

    Args:
        extracted_entities: List of extracted entity names (from speaker detection)
        summary_text: Summary text to check
        nlp: Optional spaCy NLP model for entity extraction from summary

    Returns:
        Tuple of (summary_entity_mismatch, summary_has_named_entities)
        - summary_entity_mismatch: True if summary contains entities that don't match
        - summary_has_named_entities: True if summary contains any named entities
    """
    if not summary_text:
        return False, False

    summary_has_named_entities = False
    summary_entity_mismatch = False

    # If no NLP model available, skip entity extraction (can't check)
    if not nlp:
        logger.debug("No NLP model available for entity consistency check, skipping")
        return False, False

    try:
        # Extract PERSON entities from summary
        from ..providers.ml.ner_extraction import extract_all_entities

        summary_entities = extract_all_entities(summary_text, nlp, labels=["PERSON"])
        summary_has_named_entities = len(summary_entities) > 0

        if not extracted_entities or not summary_entities:
            return False, summary_has_named_entities

        # Normalize extracted entities (lowercase, strip)
        extracted_normalized = {e.lower().strip() for e in extracted_entities if e}

        # Check each summary entity against extracted entities
        for summary_ent in summary_entities:
            summary_name = summary_ent.get("text", "").strip()
            if not summary_name:
                continue

            summary_normalized = summary_name.lower().strip()

            # Check for exact match
            if summary_normalized in extracted_normalized:
                continue

            # Check for close match (edit distance <= 2)
            # This catches cases like "Warsh" vs "Walsh"
            found_match = False
            for extracted_name in extracted_normalized:
                distance = _calculate_levenshtein_distance(summary_normalized, extracted_name)
                if distance <= 2 and len(summary_normalized) > 3 and len(extracted_name) > 3:
                    # Close match found - this is likely the same person
                    found_match = True
                    break

            if not found_match:
                # Mismatch found: summary has entity that doesn't match extracted
                summary_entity_mismatch = True
                logger.debug(
                    "Entity mismatch detected: summary has '%s' but extracted entities are: %s",
                    summary_name,
                    list(extracted_normalized),
                )
                break

    except Exception as exc:
        logger.debug("Error checking entity consistency: %s", exc, exc_info=True)
        # On error, don't flag mismatch (conservative approach)
        return False, summary_has_named_entities

    return summary_entity_mismatch, summary_has_named_entities


def _reconcile_entities(
    extracted_entities: List[str],
    summary_text: str,
    nlp: Optional[Any] = None,
    edit_distance_threshold: int = 2,
) -> Tuple[str, List[EntityCorrection]]:
    """Reconcile entity names in summary with extracted entities.

    Auto-corrects summary text when entity names are close matches (edit distance ≤ threshold)
    to extracted entities. Prefers extracted entity spelling.

    Args:
        extracted_entities: List of extracted entity names (from speaker detection)
        summary_text: Summary text to correct
        nlp: Optional spaCy NLP model for entity extraction from summary
        edit_distance_threshold: Maximum edit distance for corrections (default: 2)

    Returns:
        Tuple of (corrected_summary_text, list_of_corrections)
        - corrected_summary_text: Summary text with entity names corrected
        - list_of_corrections: List of EntityCorrection objects describing changes made
    """
    if not summary_text or not extracted_entities:
        return summary_text, []

    corrections: List[EntityCorrection] = []

    # If no NLP model available, skip reconciliation (can't extract entities)
    if not nlp:
        logger.debug("No NLP model available for entity reconciliation, skipping")
        return summary_text, []

    try:
        # Extract PERSON entities from summary
        from ..providers.ml.ner_extraction import extract_all_entities

        summary_entities = extract_all_entities(summary_text, nlp, labels=["PERSON"])
        if not summary_entities:
            return summary_text, []

        # Normalize extracted entities (keep original for replacement)
        # Map normalized -> original for lookup
        extracted_map: Dict[str, str] = {}
        for entity in extracted_entities:
            if entity:
                normalized = entity.lower().strip()
                extracted_map[normalized] = entity.strip()

        corrected_text = summary_text
        # Process entities in reverse order (end to start) to preserve positions
        for summary_ent in reversed(summary_entities):
            summary_name = summary_ent.get("text", "").strip()
            if not summary_name:
                continue

            summary_normalized = summary_name.lower().strip()

            # Check for exact match (case-insensitive)
            if summary_normalized in extracted_map:
                # Exact match - no correction needed
                continue

            # Check for close match (edit distance <= threshold)
            best_match = None
            best_distance = edit_distance_threshold + 1

            for extracted_normalized, extracted_original in extracted_map.items():
                # Only consider if both names are long enough (avoid false positives)
                if len(summary_normalized) <= 3 or len(extracted_normalized) <= 3:
                    continue

                distance = _calculate_levenshtein_distance(summary_normalized, extracted_normalized)
                if distance <= edit_distance_threshold and distance < best_distance:
                    best_match = extracted_original
                    best_distance = distance

            if best_match:
                # Found close match - correct the summary text
                start_pos = summary_ent.get("start", 0)
                end_pos = summary_ent.get("end", len(summary_text))

                # Replace the entity in the summary text
                # Preserve original capitalization if possible
                original_in_text = summary_text[start_pos:end_pos]
                corrected_in_text = best_match

                # Try to preserve capitalization pattern
                if original_in_text and corrected_in_text:
                    if original_in_text[0].isupper():
                        corrected_in_text = corrected_in_text[0].upper() + corrected_in_text[1:]
                    if len(original_in_text) > 1 and original_in_text[1:].isupper():
                        # All caps
                        corrected_in_text = corrected_in_text.upper()
                    elif len(original_in_text) > 1 and original_in_text[1:].islower():
                        # Title case
                        if len(corrected_in_text) > 1:
                            corrected_in_text = (
                                corrected_in_text[0].upper() + corrected_in_text[1:].lower()
                            )

                corrected_text = (
                    corrected_text[:start_pos] + corrected_in_text + corrected_text[end_pos:]
                )

                corrections.append(
                    EntityCorrection(
                        original=summary_name,
                        corrected=best_match,
                        edit_distance=best_distance,
                    )
                )

                logger.debug(
                    "Entity reconciliation: corrected '%s' -> '%s' (edit distance: %d)",
                    summary_name,
                    best_match,
                    best_distance,
                )

    except Exception as exc:
        logger.debug("Error reconciling entities: %s", exc, exc_info=True)
        # On error, return original text (conservative approach)
        return summary_text, []

    return corrected_text, corrections


def _build_qa_flags(
    speakers: List[SpeakerInfo],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    summary_text: Optional[str] = None,
    nlp: Optional[Any] = None,
    corrected_entities: Optional[List[EntityCorrection]] = None,
) -> QAFlags:
    """Build QA flags from speaker detection and summary information.

    Args:
        speakers: List of detected speakers
        detected_hosts: List of detected host names (may be None)
        detected_guests: List of detected guest names (may be None)
        summary_text: Optional summary text for entity consistency checking
        nlp: Optional spaCy NLP model for entity extraction

    Returns:
        QAFlags object with quality assurance information
    """
    # Import DEFAULT_SPEAKER_NAMES here to avoid circular imports
    from ..providers.ml.speaker_detection import DEFAULT_SPEAKER_NAMES

    # Check if defaults were injected
    defaults_injected = False
    if speakers:
        speaker_names = {s.name for s in speakers}
        defaults_injected = any(name in DEFAULT_SPEAKER_NAMES for name in speaker_names)

    # Determine speaker detection status
    has_hosts = bool(detected_hosts)
    has_guests = bool(detected_guests)

    if has_hosts and has_guests:
        speaker_detection = "ok"
    elif has_hosts or has_guests:
        speaker_detection = "partial"
    else:
        speaker_detection = "none"

    # Check entity consistency if summary is available
    summary_entity_mismatch = False
    summary_has_named_entities = False

    if summary_text:
        # Combine all extracted entities for consistency check
        extracted_entities = []
        if detected_hosts:
            extracted_entities.extend(detected_hosts)
        if detected_guests:
            extracted_entities.extend(detected_guests)

        summary_entity_mismatch, summary_has_named_entities = _check_entity_consistency(
            extracted_entities, summary_text, nlp
        )

    # Use provided corrected_entities or empty list
    corrections = corrected_entities if corrected_entities is not None else []

    return QAFlags(
        speaker_detection=speaker_detection,
        defaults_injected=defaults_injected,
        summary_entity_mismatch=summary_entity_mismatch,
        summary_has_named_entities=summary_has_named_entities,
        corrected_entities=corrections,
    )


def _build_content_metadata(
    episode: models.Episode,
    transcript_infos: List[TranscriptInfo],
    media_id: Optional[str],
    transcript_file_path: Optional[str],
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]],
    whisper_model: Optional[str],
    speakers: List[SpeakerInfo],
    detected_hosts: Optional[List[str]] = None,
    detected_guests: Optional[List[str]] = None,
    summary_text: Optional[str] = None,
    nlp: Optional[Any] = None,
    corrected_entities: Optional[List[EntityCorrection]] = None,
) -> ContentMetadata:
    """Build ContentMetadata object.

    Args:
        episode: Episode object
        transcript_infos: List of transcript info objects
        media_id: Generated media ID
        transcript_file_path: Path to transcript file
        transcript_source: Source of transcript
        whisper_model: Whisper model used
        speakers: List of detected speakers with structured information
        detected_hosts: Optional list of detected host names for QA flags
        detected_guests: Optional list of detected guest names for QA flags
        summary_text: Optional summary text for entity consistency checking
        nlp: Optional spaCy NLP model for entity extraction
        corrected_entities: Optional list of entity corrections applied to summary

    Returns:
        ContentMetadata object
    """
    # Create default expectations (speaker labels and names not allowed by default)
    expectations = ExpectationsMetadata(
        allow_speaker_names=False,
        allow_speaker_labels=False,
        allow_sponsor_content=False,
    )

    # Build QA flags
    qa_flags = _build_qa_flags(
        speakers=speakers,
        detected_hosts=detected_hosts,
        detected_guests=detected_guests,
        summary_text=summary_text,
        nlp=nlp,
        corrected_entities=corrected_entities,
    )

    return ContentMetadata(
        transcript_urls=transcript_infos,
        media_url=episode.media_url,
        media_id=media_id,
        media_type=episode.media_type,
        transcript_file_path=transcript_file_path,
        transcript_source=transcript_source,
        whisper_model=whisper_model,
        speakers=speakers,
        expectations=expectations,
        qa_flags=qa_flags,
    )


def _build_processing_metadata(
    cfg: config.Config,
    output_dir: str,
    stage_timings: Optional[EpisodeStageTimings] = None,
) -> ProcessingMetadata:
    """Build ProcessingMetadata object.

    Args:
        cfg: Configuration object
        output_dir: Output directory path

    Returns:
        ProcessingMetadata object
    """
    # ML/Provider information - include all models whether implicit or explicit
    # Place at top of config_snapshot for prominence
    ml_providers: Dict[str, Any] = {}

    # Transcription provider
    if cfg.transcription_provider:
        ml_providers["transcription"] = {
            "provider": str(cfg.transcription_provider),
        }
        if cfg.transcription_provider == "whisper" and cfg.transcribe_missing:
            ml_providers["transcription"]["whisper_model"] = cfg.whisper_model
        elif cfg.transcription_provider == "openai":
            # Include OpenAI transcription model
            transcription_model = getattr(cfg, "openai_transcription_model", "whisper-1")
            ml_providers["transcription"]["openai_model"] = transcription_model

    # Speaker detection provider
    if cfg.speaker_detector_provider:
        ml_providers["speaker_detection"] = {
            "provider": str(cfg.speaker_detector_provider),
        }
        if cfg.speaker_detector_provider == "spacy" and cfg.ner_model:
            ml_providers["speaker_detection"]["ner_model"] = cfg.ner_model
        elif cfg.speaker_detector_provider == "openai":
            # Include OpenAI speaker detection model
            speaker_model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
            ml_providers["speaker_detection"]["openai_model"] = speaker_model

    # Summarization provider
    if cfg.summary_provider:
        ml_providers["summarization"] = {
            "provider": str(cfg.summary_provider),
        }
        if cfg.summary_provider in ("transformers", "local"):
            # Include model information for transformers provider
            if cfg.summary_model:
                ml_providers["summarization"]["map_model"] = cfg.summary_model
            if cfg.summary_reduce_model:
                ml_providers["summarization"]["reduce_model"] = cfg.summary_reduce_model
            if cfg.summary_device:
                ml_providers["summarization"]["device"] = cfg.summary_device

            # Record library versions and device for reproducibility and drift detection
            try:
                import torch
                import transformers

                # Convert torch version to string (it's a TorchVersion object)
                torch_version = getattr(torch, "__version__", "unknown")
                torch_version_str = str(torch_version) if torch_version != "unknown" else "unknown"
                ml_providers["summarization"]["versions"] = {
                    "transformers": getattr(transformers, "__version__", "unknown"),
                    "torch": torch_version_str,
                }
                # Record device (mps/cpu/cuda) for reproducibility
                if cfg.summary_device:
                    ml_providers["summarization"]["device"] = cfg.summary_device
                # Record model revision if available (from config or model)
                if hasattr(cfg, "summary_model_revision") and cfg.summary_model_revision:
                    ml_providers["summarization"]["model_revision"] = cfg.summary_model_revision
            except (ImportError, AttributeError, ValueError):
                pass  # Versions not available if libraries not installed or mocked
        elif cfg.summary_provider == "openai":
            # Include OpenAI summarization model
            summary_model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
            ml_providers["summarization"]["openai_model"] = summary_model

    # Build config_snapshot with ml_providers first for prominence
    config_snapshot: Dict[str, Any] = {}
    if ml_providers:
        config_snapshot["ml_providers"] = ml_providers

    # Add other config fields
    config_snapshot.update(
        {
            "language": cfg.language,
            "max_episodes": cfg.max_episodes,
            "auto_speakers": cfg.auto_speakers,
            "screenplay": cfg.screenplay,
        }
    )

    return ProcessingMetadata(
        processing_timestamp=datetime.now(),
        output_directory=output_dir,
        run_id=cfg.run_id,
        config_snapshot=config_snapshot,
        schema_version=SCHEMA_VERSION,
    )


def _generate_episode_summary(  # noqa: C901
    transcript_file_path: str,
    output_dir: str,
    cfg: config.Config,
    episode_idx: int,
    summary_provider=None,  # SummarizationProvider instance (required)
    whisper_model: Optional[str] = None,  # Whisper model used for transcription
    pipeline_metrics=None,  # Metrics object for tracking LLM calls
) -> Optional[SummaryMetadata]:
    """Generate summary for an episode transcript.

    Args:
        transcript_file_path: Path to transcript file (relative to output_dir)
        output_dir: Output directory path
        cfg: Configuration object
        episode_idx: Episode index for logging
        summary_provider: SummarizationProvider instance (required)

    Returns:
        SummaryMetadata object or None if generation failed/skipped
    """
    if not cfg.generate_summaries:
        return None

    # Handle dry-run mode - skip actual model loading and inference
    # Check this FIRST before any imports or device checks that might trigger PyTorch initialization
    if cfg.dry_run:
        logger.info(
            "[%s] (dry-run) would generate summary for transcript: %s",
            episode_idx,
            transcript_file_path,
        )
        return None

    # Read transcript file
    full_transcript_path = os.path.join(output_dir, transcript_file_path)
    try:
        with open(full_transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
    except Exception as e:
        logger.warning(
            "[%s] Failed to read transcript for summarization: %s",
            episode_idx,
            e,
        )
        return None

    if not transcript_text or len(transcript_text.strip()) < 50:
        logger.debug("[%s] Transcript too short for summarization, skipping", episode_idx)
        return None

    # Use provider if available (preferred path)
    if summary_provider is not None:
        try:
            import time

            summary_start = time.time()
            transcript_length = len(transcript_text)
            word_count_approx = len(transcript_text.split())
            logger.debug(
                "[%s] Generating summary using provider (transcript: ~%d words, %d chars)...",
                episode_idx,
                word_count_approx,
                transcript_length,
            )

            # Clean transcript before summarization
            logger.debug("[%s] Cleaning transcript before summarization...", episode_idx)
            from .. import preprocessing

            cleaned_text = preprocessing.clean_transcript(  # type: ignore[attr-defined]
                transcript_text,
                remove_timestamps=True,
                normalize_speakers=True,
                collapse_blank_lines=True,
                remove_fillers=False,
            )
            logger.debug(
                "[%s] Transcript cleaned: %d -> %d chars",
                episode_idx,
                len(transcript_text),
                len(cleaned_text),
            )

            # Save cleaned transcript to separate file if configured
            if cfg.save_cleaned_transcript:
                try:
                    transcript_path_obj = Path(full_transcript_path)
                    cleaned_path = transcript_path_obj.parent / (
                        transcript_path_obj.stem + ".cleaned" + transcript_path_obj.suffix
                    )
                    # Use write_file() for consistent metrics tracking
                    from ..utils import filesystem

                    filesystem.write_file(
                        str(cleaned_path), cleaned_text.encode("utf-8"), pipeline_metrics
                    )
                    logger.debug(
                        "[%s] Saved cleaned transcript to: %s",
                        episode_idx,
                        cleaned_path.name,
                    )
                except Exception as e:
                    logger.error(
                        "[%s] Failed to save cleaned transcript: %s",
                        episode_idx,
                        e,
                    )

            # Use provider's summarize method
            # For metadata generation (final summary), use reduce params
            params: Dict[str, Any] = {
                "max_length": cfg.summary_reduce_params.get("max_new_tokens"),
                "min_length": cfg.summary_reduce_params.get("min_new_tokens"),
            }
            if cfg.summary_chunk_size:
                params["chunk_size"] = cfg.summary_chunk_size
            if cfg.summary_word_chunk_size:
                params["word_chunk_size"] = cfg.summary_word_chunk_size
            if cfg.summary_word_overlap:
                params["word_overlap"] = cfg.summary_word_overlap
            if cfg.summary_chunk_parallelism:
                params["chunk_parallelism"] = cfg.summary_chunk_parallelism
            if cfg.summary_prompt:
                params["prompt"] = str(cfg.summary_prompt)

            # Pass pipeline_metrics for LLM call tracking (if OpenAI provider)
            import inspect

            from ...utils.timeout import with_timeout

            def _summarize():
                sig = inspect.signature(summary_provider.summarize)
                if "pipeline_metrics" in sig.parameters:
                    return summary_provider.summarize(
                        text=cleaned_text,
                        episode_title=None,  # Not available in this context
                        episode_description=None,  # Not available in this context
                        params=params,
                        pipeline_metrics=pipeline_metrics,
                    )
                else:
                    return summary_provider.summarize(
                        text=cleaned_text,
                        episode_title=None,  # Not available in this context
                        episode_description=None,  # Not available in this context
                        params=params,
                    )

            # Apply summarization timeout (Issue #379)
            result = with_timeout(
                _summarize,
                cfg.summarization_timeout,
                f"summarization for episode {episode_idx}",
            )

            summary_elapsed = time.time() - summary_start
            short_summary = result.get("summary")

            # Handle Mock objects in tests - convert to string if needed
            if short_summary is not None:
                # Check if it's a Mock object (common in tests)
                from unittest.mock import Mock

                if isinstance(short_summary, Mock):
                    # Try to get a string value from the Mock
                    # If Mock has a return_value or side_effect that returns a string, use that
                    if hasattr(short_summary, "return_value") and isinstance(
                        short_summary.return_value, str
                    ):
                        short_summary = short_summary.return_value
                    elif hasattr(short_summary, "_mock_name"):
                        # It's a Mock object without a proper string value
                        # Log and skip this summary
                        logger.warning(
                            "[%s] Summary provider returned Mock object instead of "
                            "string, skipping",
                            episode_idx,
                        )
                        return None
                    else:
                        # Try to convert to string
                        try:
                            short_summary = str(short_summary)
                        except Exception:
                            logger.warning(
                                "[%s] Could not convert summary to string, skipping",
                                episode_idx,
                            )
                            return None

            # Safely get length - handle Mock objects in tests
            try:
                summary_length = len(short_summary) if short_summary else 0
            except (TypeError, AttributeError):
                # Handle Mock objects or other non-string-like objects
                summary_length = 0

            logger.info(
                "[%s] Summary generated in %.1fs (length: %d chars)",
                episode_idx,
                summary_elapsed,
                summary_length,
            )

            if not short_summary:
                # Fail fast - empty summary should never be expected when generate_summaries=True
                error_msg = (
                    f"[{episode_idx}] Summary generation returned empty result. "
                    "Empty summaries are not allowed when generate_summaries=True."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Ensure short_summary is a string (Pydantic validation requirement)
            if not isinstance(short_summary, str):
                # Fail fast - non-string summary is invalid when generate_summaries=True
                error_msg = (
                    f"[{episode_idx}] Summary is not a string "
                    f"(type: {type(short_summary).__name__}). "
                    "Invalid summary format when generate_summaries=True."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            word_count = len(transcript_text.split())

            return SummaryMetadata(
                short_summary=short_summary,
                generated_at=datetime.now(),
                word_count=word_count,
            )
        except TimeoutError as e:
            # Handle timeout errors gracefully (Issue #379)
            error_msg = f"[{episode_idx}] Summarization timeout: {e}"
            logger.error(error_msg)
            if pipeline_metrics:
                pipeline_metrics.record_episode_status(
                    episode_id=str(episode_idx),
                    status="failed",
                    error_type="TimeoutError",
                    error_message=str(e),
                    stage="summarization",
                )
            # Return None to indicate summarization failed
            return None
        except ProviderRuntimeError as e:
            error_msg = str(e).lower()
            # Handle "Already borrowed" error from Rust tokenizer in parallel execution
            # This is a known threading issue with Rust-based tokenizers
            if "already borrowed" in error_msg or "tokenizer threading error" in error_msg:
                logger.warning(
                    f"[{episode_idx}] Summarization failed due to tokenizer threading error: {e}. "
                    "This can occur in parallel execution. Skipping summary for this episode."
                )
                return None
            # For other provider errors, fail fast
            error_msg_full = (
                f"[{episode_idx}] Failed to generate summary using provider: {e}. "
                "Summarization is required when generate_summaries=True."
            )
            logger.error(error_msg_full, exc_info=True)
            raise RuntimeError(error_msg_full) from e
        except Exception as e:
            # Fail fast - if summarization fails for a specific episode, raise exception
            error_msg = (
                f"[{episode_idx}] Failed to generate summary using provider: {e}. "
                "Summarization is required when generate_summaries=True."
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # Require summary_provider - no fallback to direct model loading
    # This ensures consistent provider pattern usage and proper encapsulation
    if summary_provider is None:
        error_msg = (
            f"[{episode_idx}] summary_provider is required when generate_summaries=True. "
            "A summarization provider must be created and passed to this function. "
            "Use create_summarization_provider() to create a provider from your Config."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _determine_metadata_path(
    episode: models.Episode,
    output_dir: str,
    run_suffix: Optional[str],
    cfg: config.Config,
) -> str:
    """Determine the output path for metadata file.

    Metadata files are stored in the metadata/ subdirectory within the output directory.
    If metadata_subdirectory is set in config, it takes precedence over the default location.

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

    # Use metadata_subdirectory if set, otherwise use default metadata/
    if cfg.metadata_subdirectory:
        metadata_dir = os.path.join(output_dir, cfg.metadata_subdirectory)
    else:
        metadata_dir = os.path.join(output_dir, filesystem.METADATA_SUBDIR)
    return os.path.join(metadata_dir, f"{base_name}.metadata{extension}")


def _serialize_metadata(
    metadata_doc: EpisodeMetadataDocument,
    metadata_path: str,
    cfg: config.Config,
    pipeline_metrics=None,
) -> None:
    """Serialize and write metadata document to file.

    Args:
        metadata_doc: Metadata document to serialize
        metadata_path: Path to output file
        cfg: Configuration object

    Raises:
        OSError: If file writing fails
    """
    import time

    serialize_start = time.time()
    # Directory creation is handled by write_file() or cached, so we don't need to create it here
    # This avoids redundant os.makedirs() calls

    if cfg.metadata_format == "json":
        # Time serialization separately from I/O
        json_start = time.time()
        content_str = metadata_doc.model_dump_json(indent=2, exclude_none=False)
        json_elapsed = time.time() - json_start

        # Time actual file write (use write_file for consistent metrics and directory caching)
        write_start = time.time()
        from ..utils import filesystem

        filesystem.write_file(
            metadata_path, content_str.encode("utf-8"), pipeline_metrics=pipeline_metrics
        )
        write_elapsed = time.time() - write_start
        bytes_written = len(content_str.encode("utf-8"))

        logger.debug(
            "[STORAGE I/O] file=%s bytes=%d serialize=%.3fs write=%.3fs total=%.3fs",
            metadata_path,
            bytes_written,
            json_elapsed,
            write_elapsed,
            time.time() - serialize_start,
        )
        # Note: write_file() already records metrics, so we don't need to record again here
    else:  # yaml
        # Time serialization separately from I/O
        yaml_start = time.time()
        content_dict = metadata_doc.model_dump(exclude_none=False)
        yaml_serialize_elapsed = time.time() - yaml_start

        # Time actual file write (serialize YAML to string first, then use write_file)
        write_start = time.time()
        yaml_buffer = io.StringIO()
        yaml.dump(
            content_dict,
            yaml_buffer,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        yaml_content = yaml_buffer.getvalue()
        from ..utils import filesystem

        filesystem.write_file(
            metadata_path, yaml_content.encode("utf-8"), pipeline_metrics=pipeline_metrics
        )
        write_elapsed = time.time() - write_start
        # Estimate bytes (YAML dump doesn't return size directly)
        bytes_written = len(yaml_content.encode("utf-8"))

        logger.debug(
            "[STORAGE I/O] file=%s bytes=%d serialize=%.3fs write=%.3fs total=%.3fs",
            metadata_path,
            bytes_written,
            yaml_serialize_elapsed,
            write_elapsed,
            time.time() - serialize_start,
        )
        # Note: write_file() already records metrics, so we don't need to record again here


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
    summary_provider=None,  # SummarizationProvider instance (required)
    pipeline_metrics=None,
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
        summary_model: Pre-loaded summary model (optional, will load if None)

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
    # Build speakers array from detected_hosts and detected_guests
    speakers = _build_speakers_from_detected_names(detected_hosts, detected_guests)

    # Build per-episode stage timings from pipeline metrics (Issue #379)
    stage_timings = None
    if pipeline_metrics:
        episode_idx_str = str(episode.idx)
        # Get timings for this episode from metrics
        # Note: Metrics stores lists, so we need episode index
        download_time = None
        transcribe_time = None
        extract_names_time = None
        summarize_time = None

        # Get episode index in lists (if available)
        if (
            hasattr(pipeline_metrics, "download_media_times")
            and episode.idx < len(pipeline_metrics.download_media_times)
        ):
            download_time = pipeline_metrics.download_media_times[episode.idx]

        if (
            hasattr(pipeline_metrics, "transcribe_times")
            and episode.idx < len(pipeline_metrics.transcribe_times)
        ):
            transcribe_time = pipeline_metrics.transcribe_times[episode.idx]

        if (
            hasattr(pipeline_metrics, "extract_names_times")
            and episode.idx < len(pipeline_metrics.extract_names_times)
        ):
            extract_names_time = pipeline_metrics.extract_names_times[episode.idx]

        if (
            hasattr(pipeline_metrics, "summarize_times")
            and episode.idx < len(pipeline_metrics.summarize_times)
        ):
            summarize_time = pipeline_metrics.summarize_times[episode.idx]

        # Calculate total if any timings available
        total_time = None
        if any(t is not None for t in [download_time, transcribe_time, extract_names_time, summarize_time]):
            total_time = sum(t for t in [download_time, transcribe_time, extract_names_time, summarize_time] if t is not None)

        if any(t is not None for t in [download_time, transcribe_time, extract_names_time, summarize_time, total_time]):
            stage_timings = EpisodeStageTimings(
                download_seconds=download_time,
                transcription_seconds=transcribe_time,
                speaker_detection_seconds=extract_names_time,
                summarization_seconds=summarize_time,
                total_processing_seconds=total_time,
            )

    processing_metadata = _build_processing_metadata(cfg, output_dir, stage_timings=stage_timings)

    # Generate summary if enabled and transcript is available
    summary_metadata = None
    summary_elapsed = 0.0
    summary_text = None
    corrected_entities: List[EntityCorrection] = []
    nlp = None

    # Get NLP model for entity reconciliation and consistency checking (if needed)
    if cfg.auto_speakers and cfg.generate_summaries and transcript_file_path:
        try:
            from ..providers.ml.speaker_detection import get_ner_model

            nlp = get_ner_model(cfg)
        except Exception as exc:
            logger.debug("Could not load NLP model for entity reconciliation: %s", exc)

    if cfg.generate_summaries and transcript_file_path:
        summary_start = time.time()
        summary_metadata = _generate_episode_summary(
            transcript_file_path=transcript_file_path,
            output_dir=output_dir,
            cfg=cfg,
            episode_idx=episode.idx,
            summary_provider=summary_provider,
            whisper_model=whisper_model,  # Whisper model used for transcription
            pipeline_metrics=pipeline_metrics,
        )
        summary_elapsed = time.time() - summary_start
        # Record summary generation time if metrics available
        if pipeline_metrics is not None and summary_elapsed > 0:
            pipeline_metrics.record_summarize_time(summary_elapsed)
        # Validate that summary was generated when required
        if cfg.generate_summaries and summary_metadata is None:
            error_msg = (
                f"[{episode.idx}] Summary generation failed but generate_summaries=True. "
                "Summarization is required when generate_summaries is enabled."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        # Extract summary text for QA flags and entity reconciliation
        if summary_metadata:
            summary_text = summary_metadata.short_summary

            # Reconcile entities in summary if NLP model and extracted entities available
            if nlp and summary_text:
                # Combine all extracted entities for reconciliation
                extracted_entities = []
                if detected_hosts:
                    extracted_entities.extend(detected_hosts)
                if detected_guests:
                    extracted_entities.extend(detected_guests)

                if extracted_entities:
                    corrected_summary_text, corrections = _reconcile_entities(
                        extracted_entities, summary_text, nlp
                    )
                    if corrections:
                        # Update summary metadata with corrected text
                        summary_metadata.short_summary = corrected_summary_text
                        summary_text = corrected_summary_text
                        corrected_entities = corrections
                        logger.info(
                            "[%s] Entity reconciliation: corrected %d entity name(s) in summary",
                            episode.idx,
                            len(corrections),
                        )

    # Build content metadata after summary is generated (so QA flags can use summary)
    content_metadata = _build_content_metadata(
        episode,
        transcript_infos,
        media_id,
        transcript_file_path,
        transcript_source,
        whisper_model,
        speakers,
        detected_hosts=detected_hosts,
        detected_guests=detected_guests,
        summary_text=summary_text,
        nlp=nlp,
        corrected_entities=corrected_entities,
    )

    # Build complete metadata document
    metadata_doc = EpisodeMetadataDocument(
        feed=feed_metadata,
        episode=episode_metadata,
        content=content_metadata,
        processing=processing_metadata,
        summary=summary_metadata,
    )

    # Determine output path
    metadata_path = _determine_metadata_path(episode, output_dir, run_suffix, cfg)

    # Check skip_existing
    # Allow overwriting when:
    # 1. transcript_source is "whisper_transcription" (complete transcript data)
    # 2. generate_summaries is enabled (to regenerate summaries even if metadata exists)
    # This fixes the case where metadata was created with transcript_source=None (Whisper pending)
    # and needs to be updated with actual transcript information after transcription completes
    if cfg.skip_existing and os.path.exists(metadata_path):
        if transcript_source != "whisper_transcription" and not cfg.generate_summaries:
            logger.debug(
                "[%s] Metadata file already exists; skipping (--skip-existing): %s",
                episode.idx,
                metadata_path,
            )
            return None
        # Allow overwriting for whisper_transcription or when generating summaries
        if cfg.generate_summaries:
            logger.debug(
                "[%s] Metadata file exists but will be regenerated with summaries: %s",
                episode.idx,
                metadata_path,
            )
        else:
            logger.debug(
                "[%s] Metadata file exists but will be overwritten with complete transcript data: %s",  # noqa: E501
                episode.idx,
                metadata_path,
            )

    # Handle dry-run mode
    if cfg.dry_run:
        logger.info(
            "[%s] (dry-run) would generate metadata file: %s",
            episode.idx,
            metadata_path,
        )
        return metadata_path

    # Serialize and write metadata
    try:
        _serialize_metadata(metadata_doc, metadata_path, cfg, pipeline_metrics=pipeline_metrics)
        logger.debug("[%s] Generated metadata file: %s", episode.idx, metadata_path)

        # Track metadata generation and summarization
        if pipeline_metrics is not None:
            pipeline_metrics.metadata_files_generated += 1
            if summary_metadata is not None:
                pipeline_metrics.episodes_summarized += 1

        return metadata_path

    except Exception as exc:
        logger.error(
            "[%s] Failed to generate metadata file %s: %s",
            episode.idx,
            metadata_path,
            exc,
            exc_info=True,
        )
        return None
