"""Metadata generation for podcast episodes.

This module implements per-episode metadata document generation as per PRD-004 and RFC-011.
Metadata documents are structured JSON/YAML files that capture comprehensive feed and
episode information for search, analytics, integration, and archival use cases.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, computed_field, Field, field_serializer

from .. import config, models
from ..exceptions import ProviderRuntimeError
from ..utils import filesystem

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


def _build_content_metadata(
    episode: models.Episode,
    transcript_infos: List[TranscriptInfo],
    media_id: Optional[str],
    transcript_file_path: Optional[str],
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]],
    whisper_model: Optional[str],
    speakers: List[SpeakerInfo],
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

    Returns:
        ContentMetadata object
    """
    # Create default expectations (speaker labels and names not allowed by default)
    expectations = ExpectationsMetadata(
        allow_speaker_names=False,
        allow_speaker_labels=False,
        allow_sponsor_content=False,
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
    )


def _build_processing_metadata(cfg: config.Config, output_dir: str) -> ProcessingMetadata:
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
                    with open(cleaned_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
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

            sig = inspect.signature(summary_provider.summarize)
            if "pipeline_metrics" in sig.parameters:
                result = summary_provider.summarize(
                    text=cleaned_text,
                    episode_title=None,  # Not available in this context
                    episode_description=None,  # Not available in this context
                    params=params,
                    pipeline_metrics=pipeline_metrics,
                )
            else:
                result = summary_provider.summarize(
                    text=cleaned_text,
                    episode_title=None,  # Not available in this context
                    episode_description=None,  # Not available in this context
                    params=params,
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
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)

    if cfg.metadata_format == "json":
        # Time serialization separately from I/O
        json_start = time.time()
        content_str = metadata_doc.model_dump_json(indent=2, exclude_none=False)
        json_elapsed = time.time() - json_start

        # Time actual file write
        write_start = time.time()
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(content_str)
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
        # Record actual file write time in metrics
        if pipeline_metrics is not None:
            pipeline_metrics.record_stage("writing_storage", write_elapsed)
    else:  # yaml
        # Time serialization separately from I/O
        yaml_start = time.time()
        content_dict = metadata_doc.model_dump(exclude_none=False)
        yaml_serialize_elapsed = time.time() - yaml_start

        # Time actual file write
        write_start = time.time()
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(
                content_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )
        write_elapsed = time.time() - write_start
        # Estimate bytes (YAML dump doesn't return size directly)
        bytes_written = len(str(content_dict).encode("utf-8"))

        logger.debug(
            "[STORAGE I/O] file=%s bytes=%d serialize=%.3fs write=%.3fs total=%.3fs",
            metadata_path,
            bytes_written,
            yaml_serialize_elapsed,
            write_elapsed,
            time.time() - serialize_start,
        )
        # Record actual file write time in metrics
        if pipeline_metrics is not None:
            pipeline_metrics.record_stage("writing_storage", write_elapsed)


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

    content_metadata = _build_content_metadata(
        episode,
        transcript_infos,
        media_id,
        transcript_file_path,
        transcript_source,
        whisper_model,
        speakers,
    )
    processing_metadata = _build_processing_metadata(cfg, output_dir)

    # Generate summary if enabled and transcript is available
    summary_metadata = None
    summary_elapsed = 0.0
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
