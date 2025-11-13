# Release v2.1.0 - Automatic Speaker Detection & Metadata Generation Design

**Release Date:** November 13, 2025  
**Type:** Minor Release

## Summary

v2.1.0 introduces automatic speaker name detection using Named Entity Recognition (NER) and establishes the design foundation for per-episode metadata generation. This release significantly enhances the transcript pipeline with intelligent host and guest identification, while laying the groundwork for comprehensive metadata documents.

## What's New

### ✨ RFC-010: Automatic Speaker Name Detection (Implemented)

**Automatic host and guest identification from episode metadata:**

- **Named Entity Recognition (NER)**: Uses spaCy to automatically extract person names from episode titles and descriptions
- **RSS Author Tag Support**: Prioritizes RSS `<author>`, `<itunes:author>`, and `<itunes:owner>` tags for host identification
- **Host/Guest Distinction**: Intelligently distinguishes recurring hosts from episode-specific guests
- **Host Validation**: Validates detected hosts using the first episode's metadata
- **Pattern-Based Heuristics**: Analyzes sample episodes to learn title patterns (position, prefixes, suffixes) for better guest selection
- **Confidence Scoring**: Uses overlap detection and confidence scores to select the best guest candidate
- **Name Sanitization**: Removes punctuation, normalizes whitespace, and deduplicates extracted names
- **Graceful Fallback**: Falls back to manual speaker names if automatic detection fails
- **Language-Aware**: Single `language` configuration drives both Whisper model selection and NER processing

**Configuration:**

- `--auto-speakers` / `auto_speakers`: Enable automatic speaker detection (default: `true`)
- `--language` / `language`: Language for both Whisper and NER (default: `"en"`)
- `--ner-model` / `ner_model`: spaCy model to use (default: `"en_core_web_sm"`)
- `--cache-detected-hosts` / `cache_detected_hosts`: Cache host detection across episodes (default: `true`)
- `--speaker-names` / `speaker_names`: Manual fallback names (first = host, second = guest)

**Features:**

- Auto-downloads spaCy models when needed (similar to Whisper)
- Clear logging of detected hosts and guests at INFO level
- Detailed extraction details available at DEBUG level
- Works seamlessly in dry-run mode
- Pattern analysis from first few episodes improves accuracy

### 📋 PRD-004 & RFC-011: Metadata Generation Design

**Comprehensive design documents for per-episode metadata generation:**

- **PRD-004**: Product Requirements Document defining metadata schema, use cases, and functional requirements
- **RFC-011**: Technical design document with:
  - Pydantic model definitions for type-safe metadata structure
  - Database integration design (PostgreSQL, MongoDB, Elasticsearch, ClickHouse)
  - ID generation strategy (feed_id, episode_id, content IDs)
  - Unified JSON format with `snake_case` field names and ISO 8601 date serialization
  - Database loading examples for all target databases

**Key Design Decisions:**

- Opt-in feature (default `false` for backwards compatibility)
- JSON (default) and YAML format support
- Database-friendly schema for direct ingestion without transformation
- Stable, deterministic ID generation suitable for primary keys
- Schema versioning strategy (semantic versioning starting at 1.0.0)

**Note:** Implementation will follow in a future release. This release establishes the design foundation.

## Improvements

### RSS Parser Enhancements

- **Author Tag Extraction**: Extracts RSS `<author>`, `<itunes:author>`, and `<itunes:owner>` tags from feed channel
- **HTML Stripping**: Removes HTML tags and decodes entities from episode descriptions
- **Improved Episode Description Extraction**: Better handling of HTML content in RSS descriptions

### Code Quality & Tooling

- **CI/CD Alignment**: GitHub Actions workflow now uses `make ci` directly, ensuring local and CI checks match exactly
- **Markdown Linting**: Added markdownlint to CI pipeline, catching formatting issues early
- **Complexity Management**: Added complexity warnings to `.flake8` per-file-ignores for appropriate modules
- **Code Cleanup**: Removed unused variables and imports

### Documentation Updates

- **ARCHITECTURE.md**: Updated to reflect speaker detection integration and metadata generation design
- **TESTING_STRATEGY.md**: Expanded with speaker detection and metadata generation testing requirements
- **PRD-002 & PRD-003**: Updated to reflect RFC-010 speaker detection features
- **RFC-010**: Expanded with final implementation details (RSS author tags, host validation, pattern heuristics)
- **RFC-011**: Comprehensive technical design for metadata generation

## Technical Details

### New Dependencies

- **spacy>=3.7.0**: Required dependency for Named Entity Recognition

### Module Changes

- **`speaker_detection.py`** (new): Complete NER-based speaker detection implementation
- **`rss_parser.py`**: Enhanced with author tag extraction and HTML stripping
- **`workflow.py`**: Integrated speaker detection into pipeline
- **`episode_processor.py`**: Updated to use detected speaker names
- **`whisper_integration.py`**: Renamed from `whisper.py` to avoid naming conflicts
- **`models.py`**: Added `authors` field to `RssFeed` dataclass

### Configuration Changes

New configuration fields:

```yaml
language: "en"                    # Language for Whisper and NER
auto_speakers: true               # Enable automatic speaker detection
ner_model: "en_core_web_sm"       # spaCy model name
cache_detected_hosts: true        # Cache host detection
```

## Changes

### Files Changed

**New Files:**

- `speaker_detection.py` - NER-based speaker detection implementation
- `docs/prd/PRD-004-metadata-generation.md` - Metadata generation PRD
- `docs/rfc/RFC-011-metadata-generation.md` - Metadata generation RFC
- `.markdownlint.json` - Markdown linting configuration

**Modified Files:**

- `rss_parser.py` - Author tag extraction, HTML stripping
- `workflow.py` - Speaker detection integration
- `episode_processor.py` - Use detected speaker names
- `config.py` - New configuration fields
- `models.py` - Added authors field
- `whisper.py` → `whisper_integration.py` - Renamed to avoid conflicts
- `test_podcast_scraper.py` - Comprehensive tests for speaker detection
- `docs/ARCHITECTURE.md` - Updated architecture documentation
- `docs/TESTING_STRATEGY.md` - Updated testing requirements
- `.github/workflows/python-app.yml` - Aligned with `make ci`
- `.flake8` - Added complexity ignores

### Statistics

- 20+ files changed
- 2,000+ lines added (speaker detection implementation)
- 766 lines (RFC-011 metadata generation design)
- 240+ lines (comprehensive test coverage)

## Related Issues

- Closes #21: Implement NER as per RFC-010
- Closes #15: Create PRD and RFC for generating metadata document per episode
- Related to #28: CI/CD improvements

## Documentation

- Speaker Detection: [RFC-010](../rfc/RFC-010-speaker-name-detection.md)
- Metadata Generation Design: [PRD-004](../prd/PRD-004-metadata-generation.md), [RFC-011](../rfc/RFC-011-metadata-generation.md)
- Architecture: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Testing Strategy: [TESTING_STRATEGY.md](../TESTING_STRATEGY.md)

## Migration Notes

### For Users Upgrading from v2.0.1

1. **New Required Dependency**: Install spaCy models automatically or manually:

   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Automatic Speaker Detection**: Enabled by default (`auto_speakers: true`). To disable:

   ```yaml
   auto_speakers: false
   ```

3. **Manual Speaker Names**: Now used as fallback only if automatic detection fails. First name = host, second = guest.

4. **Language Configuration**: Single `language` field now controls both Whisper and NER:

   ```yaml
   language: "en"  # Used for both Whisper model selection and NER
   ```

5. **No Breaking Changes**: All existing functionality preserved. New features are additive.

### Configuration Example

```yaml
rss: "https://example.com/feed.xml"
output_dir: "./transcripts"
language: "en"                    # New: Controls Whisper + NER
auto_speakers: true               # New: Enable automatic detection
ner_model: "en_core_web_sm"       # New: spaCy model
cache_detected_hosts: true        # New: Cache hosts across episodes
speaker_names: "Host, Guest"      # Fallback if detection fails
```

## Testing

- 63 tests passing
- Comprehensive coverage for speaker detection:
  - RSS author tag extraction
  - NER model loading and validation
  - Host detection from feed metadata
  - Guest detection from episode metadata
  - Pattern analysis and heuristics
  - Name sanitization and deduplication
  - Fallback to manual names
  - Dry-run mode support

## Contributors

- Marko Dragoljevic (@chipi)

## Next Steps

- Implement metadata generation module (`podcast_scraper/metadata.py`) per RFC-011
- Integrate metadata generation into workflow pipeline
- Add configuration fields and CLI flags for metadata generation
- Generate metadata documents alongside transcripts

**Full Changelog**: <https://github.com/chipi/podcast_scraper/compare/v2.0.1...v2.1.0>
