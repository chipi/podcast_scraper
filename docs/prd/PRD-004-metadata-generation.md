# PRD-004: Per-Episode Metadata Document Generation

## Summary

Generate comprehensive metadata documents for each episode alongside transcripts, capturing feed-level and episode-level information for search, analytics, integration, and archival use cases.

## Background & Context

Currently, `podcast_scraper` focuses on downloading transcripts but doesn't systematically capture and persist rich metadata about feeds and episodes. This metadata is valuable for:

- **Search and discovery**: Finding episodes by guest names, topics, dates
- **Analytics**: Understanding feed patterns, guest frequency, publication schedules
- **Integration**: Enabling other tools to consume structured episode data
- **Archival**: Preserving complete episode context alongside transcripts
- **Future features**: Episode categorization, recommendation systems, summarization (PRD-005)

## Goals

- Generate structured metadata documents (JSON/YAML) for each processed episode
- Capture comprehensive feed and episode information in a machine-readable format
- Integrate with existing pipeline without disrupting current workflows
- Enable downstream tools to consume episode metadata programmatically
- Preserve metadata alongside transcripts for complete episode records

## Non-Goals

- Real-time metadata API (future consideration)
- Metadata versioning/migration (handled via schema versioning)
- Metadata search indexing (consumers can build their own indexes)
- Metadata editing/updates after generation (one-way generation)

## Personas

- **Archivist Ava**: Needs complete episode records with metadata for compliance and legal review
- **Researcher Riley**: Wants structured data for NLP analysis and pattern detection across podcast series
- **Developer Devin**: Building tools that consume episode metadata for search, recommendations, or analytics
- **Analyst Alex**: Analyzing podcast patterns, guest frequency, publication schedules

## User Stories

- *As Archivist Ava, I can generate metadata documents alongside transcripts to create complete episode records.*
- *As Researcher Riley, I can consume structured metadata JSON to analyze patterns across multiple podcast feeds.*
- *As Developer Devin, I can integrate episode metadata into my application without parsing RSS feeds directly.*
- *As Analyst Alex, I can aggregate metadata from multiple runs to understand feed evolution over time.*
- *As any operator, I can opt-in to metadata generation via configuration flag.*
- *As any operator, I can choose JSON or YAML format based on my preference.*

## Functional Requirements

### FR1: Metadata Generation Control
- **FR1.1**: Add `generate_metadata` config field (default `false` for backwards compatibility)
- **FR1.2**: Add `--generate-metadata` CLI flag
- **FR1.3**: Add `metadata_format` config field (`"json"` or `"yaml"`, default `"json"`)
- **FR1.4**: Metadata generation respects `--dry-run` mode (logs planned metadata without writing files)

### FR2: Feed-Level Metadata
- **FR2.1**: Capture feed title
- **FR2.2**: Capture feed URL
- **FR2.3**: Capture feed description (if available)
- **FR2.4**: Capture feed language (from config or detected)
- **FR2.5**: Capture feed authors (from RSS author tags)
- **FR2.6**: Capture feed image/logo URL (if available)
- **FR2.7**: Capture feed last updated date (if available)

### FR3: Episode-Level Metadata
- **FR3.1**: Capture episode title
- **FR3.2**: Capture episode description/summary (HTML-stripped)
- **FR3.3**: Capture episode published date (parsed and normalized)
- **FR3.4**: Capture episode GUID/ID (if available)
- **FR3.5**: Capture episode link/URL (if available)
- **FR3.6**: Capture episode duration (if available)
- **FR3.7**: Capture episode number/sequence (if available)
- **FR3.8**: Capture episode image/artwork URL (if available)

### FR4: Content Metadata
- **FR4.1**: Capture transcript URLs with types/formats (from Podcasting 2.0 tags)
- **FR4.2**: Capture media URL (enclosure)
- **FR4.3**: Capture media type/MIME type
- **FR4.4**: Capture detected guest names (from RFC-010 speaker detection)
- **FR4.5**: Capture detected host names (from RFC-010 speaker detection)
- **FR4.6**: Capture transcript source (`"direct_download"` or `"whisper_transcription"`)
- **FR4.7**: Capture Whisper model used (if applicable)
- **FR4.8**: Capture transcript file path (relative to output directory)

### FR5: Processing Metadata
- **FR5.1**: Capture processing timestamp (ISO 8601 format)
- **FR5.2**: Capture output directory path
- **FR5.3**: Capture run ID (if applicable)
- **FR5.4**: Capture processing configuration snapshot (selected config fields)
- **FR5.5**: Capture schema version for future compatibility

### FR6: File Storage
- **FR6.1**: Store metadata files in same directory as transcripts (by default)
- **FR6.2**: Use naming convention: `<episode_number> - <title>.metadata.json` or `.metadata.yaml`
- **FR6.3**: Respect `--skip-existing` semantics (skip metadata generation if file exists)
- **FR6.4**: Support optional separate `metadata/` subdirectory (configurable)

### FR7: Schema & Format
- **FR7.1**: Define JSON Schema for metadata structure
- **FR7.2**: Support JSON format (machine-readable, default)
- **FR7.3**: Support YAML format (human-readable, optional)
- **FR7.4**: Include schema version field for future evolution
- **FR7.5**: Validate metadata structure before writing

### FR8: Integration Points
- **FR8.1**: Generate metadata during episode processing workflow
- **FR8.2**: Integrate with RFC-010 speaker detection (populate host/guest names)
- **FR8.3**: Integrate with RFC-004 filesystem layout (use same output directory structure)
- **FR8.4**: Integrate with PRD-001 transcript pipeline (capture transcript URLs)
- **FR8.5**: Integrate with PRD-002 Whisper fallback (capture Whisper model info)

## Success Metrics

- Metadata files generated for 100% of processed episodes when feature enabled
- Metadata schema validates successfully for all generated files
- Zero impact on existing transcript download/transcription workflows when disabled
- Metadata files consumable by standard JSON/YAML parsers
- Processing time increase <5% when metadata generation enabled

## Dependencies

- RFC-010: Automatic Speaker Name Detection (populates host/guest names)
- RFC-004: Filesystem Layout & Run Management (output directory structure)
- PRD-001: Transcript Acquisition Pipeline (transcript URLs)
- PRD-002: Whisper Fallback Transcription (Whisper model info)
- Current models: `podcast_scraper/models.py` (Episode, RssFeed)

## Design Considerations

### Format Selection
- **JSON**: Machine-readable, widely supported, smaller file size
- **YAML**: Human-readable, easier to edit manually, larger file size
- **Decision**: Support both, default to JSON for performance

### Storage Location
- **Same directory as transcripts**: Simple, keeps related files together
- **Separate `metadata/` subdirectory**: Cleaner separation, easier to exclude from searches
- **Decision**: Default to same directory, allow configurable subdirectory

### Optional vs Required
- **Opt-in (default `false`)**: Backwards compatible, doesn't affect existing users
- **Opt-out (default `true`)**: More useful by default, but changes behavior
- **Decision**: Opt-in for backwards compatibility

### Schema Versioning
- **Version field**: Enables future schema evolution without breaking consumers
- **Semantic versioning**: Major.minor.patch format
- **Decision**: Start with version `1.0.0`, increment major for breaking changes

## Open Questions

- Should metadata include full transcript text or just references? (Decision: References only, transcripts are separate files)
- Should metadata be generated for episodes without transcripts? (Decision: Yes, metadata is independent of transcript availability)
- How to handle metadata updates for existing episodes? (Decision: Regenerate on each run, use `--skip-existing` to prevent overwrites)
- Should metadata include checksums/hashes? (Future consideration)

## Related Work

- RFC-010: Automatic Speaker Name Detection (will populate guest/host names)
- RFC-004: Filesystem Layout & Run Management (output directory structure)
- PRD-001: Transcript Acquisition Pipeline (current transcript workflow)
- PRD-002: Whisper Fallback Transcription (Whisper integration)
- PRD-005: Episode Summarization (future use case for metadata)

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-011 created with technical design
- [ ] Implementation completed
- [ ] Tests cover metadata generation, validation, format options
- [ ] Documentation updated (README, config examples)
- [ ] Schema versioning strategy documented

