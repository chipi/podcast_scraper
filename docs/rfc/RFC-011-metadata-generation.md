# RFC-011: Per-Episode Metadata Document Generation

- **Status**: Draft
- **Authors**: GPT-5 Codex
- **Stakeholders**: Maintainers, archivists, developers building integrations
- **Related PRDs**: `docs/prd/PRD-004-metadata-generation.md`

## Abstract

Design and implement per-episode metadata document generation to capture comprehensive feed and episode information in structured JSON/YAML format. This enables search, analytics, integration, and archival use cases while maintaining backwards compatibility with existing workflows.

## Problem Statement

Currently, `podcast_scraper` focuses on downloading transcripts but doesn't systematically capture and persist rich metadata about feeds and episodes. This metadata is valuable for:

- **Search and discovery**: Finding episodes by guest names, topics, dates
- **Analytics**: Understanding feed patterns, guest frequency, publication schedules
- **Integration**: Enabling other tools to consume structured episode data
- **Archival**: Preserving complete episode context alongside transcripts
- **Future features**: Episode categorization, recommendation systems, summarization

Without structured metadata, users must parse RSS feeds or transcripts manually, which is error-prone and doesn't scale.

## Constraints & Assumptions

- Metadata generation must be opt-in (default `false`) for backwards compatibility
- Metadata files should be machine-readable (JSON) or human-readable (YAML)
- Metadata generation should not significantly impact processing performance (<5% overhead)
- Metadata schema must be versioned for future evolution
- Metadata should integrate seamlessly with existing pipeline (RFC-001, RFC-004)
- Metadata should leverage detected speaker names from RFC-010

## Design & Implementation

### 1. Configuration

Add new configuration fields to `config.Config`:

```python
generate_metadata: bool = False  # Opt-in for backwards compatibility
metadata_format: Literal["json", "yaml"] = "json"  # Default to JSON
metadata_subdirectory: Optional[str] = None  # None = same dir as transcripts, "metadata" = subdirectory
```

Add CLI flags:
- `--generate-metadata`: Enable metadata generation
- `--metadata-format`: Choose `json` or `yaml` (default: `json`)
- `--metadata-subdirectory`: Optional subdirectory name (default: same as transcripts)

### 2. Metadata Schema

Define Pydantic model for type safety and validation:

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any

class FeedMetadata(BaseModel):
    """Feed-level metadata."""
    title: str
    url: str
    description: Optional[str] = None
    language: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    last_updated: Optional[datetime] = None

class EpisodeMetadata(BaseModel):
    """Episode-level metadata."""
    title: str
    description: Optional[str] = None
    published_date: Optional[datetime] = None
    guid: Optional[str] = None
    link: Optional[str] = None
    duration_seconds: Optional[int] = None
    episode_number: Optional[int] = None
    image_url: Optional[str] = None

class TranscriptInfo(BaseModel):
    """Transcript URL and type information."""
    url: str
    type: Optional[str] = None  # e.g., "text/plain", "text/vtt"
    language: Optional[str] = None

class ContentMetadata(BaseModel):
    """Content-related metadata."""
    transcript_urls: List[TranscriptInfo] = Field(default_factory=list)
    media_url: Optional[str] = None
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
    schema_version: str = "1.0.0"

class EpisodeMetadataDocument(BaseModel):
    """Complete episode metadata document."""
    feed: FeedMetadata
    episode: EpisodeMetadata
    content: ContentMetadata
    processing: ProcessingMetadata
```

### 3. Metadata Generation Module

Create `podcast_scraper/metadata.py` module:

```python
def generate_episode_metadata(
    feed: RssFeed,
    episode: Episode,
    content_metadata: ContentMetadata,
    cfg: Config,
    output_dir: str,
    run_suffix: Optional[str] = None,
) -> Optional[str]:
    """Generate metadata document for an episode.
    
    Returns:
        Path to generated metadata file, or None if generation skipped
    """
    # Build metadata document
    # Write to file (JSON or YAML)
    # Return file path
```

### 4. Integration Points

#### Workflow Integration (`workflow.py`)
- Generate metadata after episode processing completes
- Pass detected speaker names from RFC-010
- Pass transcript file paths
- Pass Whisper model info (if applicable)

#### Episode Processor Integration (`episode_processor.py`)
- Capture transcript source (direct download vs Whisper)
- Capture Whisper model used
- Pass metadata to generation function

#### RSS Parser Integration (`rss_parser.py`)
- Extract additional feed metadata (image, last updated)
- Extract additional episode metadata (duration, episode number, image)

### 5. File Storage

**Default behavior** (same directory as transcripts):
- Metadata file: `<idx:04d> - <title_safe>.metadata.json`
- Stored alongside transcript file in same directory

**Optional subdirectory** (`metadata_subdirectory` set):
- Metadata file: `<metadata_subdirectory>/<idx:04d> - <title_safe>.metadata.json`
- Keeps metadata separate from transcripts

**Naming convention**:
- Match transcript filename base (without extension)
- Append `.metadata.json` or `.metadata.yaml`
- Respect `run_suffix` if present: `<base>_<run_suffix>.metadata.json`

### 6. Schema Versioning

- Include `schema_version` field in all metadata documents
- Start with version `1.0.0`
- Use semantic versioning:
  - **Major**: Breaking changes (structure changes, required field additions)
  - **Minor**: Non-breaking additions (optional fields)
  - **Patch**: Bug fixes, clarifications

### 7. Configuration Snapshot

Capture relevant configuration fields in metadata:

```python
config_snapshot = {
    "language": cfg.language,
    "whisper_model": cfg.whisper_model if cfg.transcribe_missing else None,
    "auto_speakers": cfg.auto_speakers,
    "screenplay": cfg.screenplay,
    "max_episodes": cfg.max_episodes,
}
```

## Implementation Details

### Metadata Collection

1. **Feed Metadata**: Extract from `RssFeed` object and RSS parsing
2. **Episode Metadata**: Extract from `Episode` object and RSS item parsing
3. **Content Metadata**: 
   - Transcript URLs: From `episode.transcript_urls`
   - Media URL: From `episode.media_url`
   - Speaker names: From `TranscriptionJob.detected_speaker_names` or detection results
   - Transcript source: Track during processing (`direct_download` vs `whisper_transcription`)
   - Whisper model: From `cfg.whisper_model` if transcription used
4. **Processing Metadata**: 
   - Timestamp: `datetime.now().isoformat()`
   - Output directory: From `cfg.output_dir` or derived path
   - Run ID: From `cfg.run_id` if set
   - Config snapshot: Selected fields from `cfg`

### File Writing

- Use `filesystem.write_file()` for consistency
- Validate metadata structure before writing (Pydantic validation)
- Handle encoding (UTF-8)
- Respect `--skip-existing` (check if metadata file exists)
- Respect `--dry-run` (log planned metadata without writing)

### Error Handling

- If metadata generation fails, log warning but don't fail episode processing
- If metadata validation fails, log error and skip generation
- If file write fails, log error but continue processing

## Testing Strategy

- Unit tests for metadata model validation
- Unit tests for JSON/YAML serialization
- Unit tests for file naming and path construction
- Integration tests for metadata generation in workflow
- Integration tests for `--skip-existing` behavior
- Integration tests for `--dry-run` mode
- Tests for schema versioning

## Backwards Compatibility

- Feature is opt-in (default `false`), so existing workflows unaffected
- When disabled, zero performance impact
- Metadata files are additive (don't modify existing transcript files)
- Schema versioning allows future evolution without breaking consumers

## Performance Considerations

- Metadata generation should add <5% overhead
- Use efficient JSON/YAML serialization (Pydantic's built-in methods)
- Lazy evaluation: only generate when `generate_metadata=True`
- Batch file writes if possible (though per-episode is fine)

## Alternatives Considered

### SQLite Database
- **Pros**: Queryable, efficient for large datasets
- **Cons**: Additional dependency, harder to version control, overkill for current use case
- **Decision**: Rejected in favor of file-based approach for simplicity

### Single Metadata File Per Feed
- **Pros**: Single file to manage
- **Cons**: Harder to update incrementally, larger files, concurrency issues
- **Decision**: Rejected in favor of per-episode files for flexibility

### Metadata in Transcript Files
- **Pros**: Single file per episode
- **Cons**: Mixes content and metadata, harder to parse programmatically
- **Decision**: Rejected in favor of separate metadata files

## Rollout Plan

1. Create PRD-004 and RFC-011 documents
2. Review with stakeholders
3. Implement metadata generation module
4. Integrate into workflow
5. Add tests
6. Update documentation
7. Release as opt-in feature
8. Collect user feedback
9. Consider making default in future release

## Open Questions

- Should metadata include full RSS item XML? (Decision: No, keep structured)
- Should metadata include checksums/hashes? (Future consideration)
- Should metadata support incremental updates? (Decision: Regenerate on each run)
- Should metadata include transcript excerpts? (Decision: No, transcripts are separate files)

## References

- PRD-004: Per-Episode Metadata Document Generation
- RFC-010: Automatic Speaker Name Detection
- RFC-004: Filesystem Layout & Run Management
- RFC-001: Workflow Orchestration
- Pydantic documentation: <https://docs.pydantic.dev/>
- JSON Schema: <https://json-schema.org/>

