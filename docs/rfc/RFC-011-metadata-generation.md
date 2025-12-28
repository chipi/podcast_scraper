# RFC-011: Per-Episode Metadata Document Generation

- **Status**: Completed
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
- **Database ingestion**: Loading metadata directly into databases (PostgreSQL, MongoDB, Elasticsearch, ClickHouse) for querying, indexing, and analysis
- **Archival**: Preserving complete episode context alongside transcripts
- **Future features**: Episode categorization, recommendation systems, summarization

Without structured metadata, users must parse RSS feeds or transcripts manually, which is error-prone and doesn't scale. Additionally, loading metadata into databases requires custom transformation code, adding friction to data pipeline workflows.

## Constraints & Assumptions

- Metadata generation must be opt-in (default `false`) for backwards compatibility
- Metadata files should be machine-readable (JSON) or human-readable (YAML)
- **Database-friendly schema**: Metadata must be loadable into PostgreSQL (JSONB), MongoDB, Elasticsearch, and ClickHouse without transformation code
- **Unified format**: Single JSON schema works across all target databases (no format variations needed)
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
from pydantic import BaseModel, Field, field_serializer
from datetime import datetime
from typing import List, Optional, Dict, Any

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

    @field_serializer('last_updated')
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

    @field_serializer('published_date')
    def serialize_published_date(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime as ISO 8601 string for database compatibility."""
        return value.isoformat() if value else None

class TranscriptInfo(BaseModel):
    """Transcript URL and type information."""
    url: str
    transcript_id: Optional[str] = None  # Optional stable identifier for tracking individual transcripts
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
    schema_version: str = "1.0.0"

    @field_serializer('processing_timestamp')
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
   - **Feed ID Generation**: Generate stable unique identifier from feed URL (see ID Generation Strategy below)
2. **Episode Metadata**: Extract from `Episode` object and RSS item parsing
   - **Episode ID Generation**: Generate stable unique identifier (see ID Generation Strategy below)
   - **RSS GUID**: Extract from RSS item `<guid>` tag if available
3. **Content Metadata**:
   - Transcript URLs: From `episode.transcript_urls`
   - Media URL: From `episode.media_url`
   - **Content IDs** (optional): Generate `transcript_id` and `media_id` if tracking content separately
   - Speaker names: From `TranscriptionJob.detected_speaker_names` or detection results
   - Transcript source: Track during processing (`direct_download` vs `whisper_transcription`)
   - Whisper model: From `cfg.whisper_model` if transcription used
4. **Processing Metadata**:
   - Timestamp: `datetime.now().isoformat()`
   - Output directory: From `cfg.output_dir` or derived path
   - Run ID: From `cfg.run_id` if set
   - Config snapshot: Selected fields from `cfg`

### ID Generation Strategy

Each metadata document must include stable, unique identifiers suitable for use as primary keys in databases:

1. **Feed ID** (`feed.feed_id`): Identifies the feed uniquely
2. **Episode ID** (`episode.episode_id`): Identifies the episode uniquely
3. **Content IDs** (optional): `transcript_id` and `media_id` for tracking individual content items

#### Feed ID Generation

The feed ID is generated from the feed URL (normalized):

```python
import hashlib
from urllib.parse import urlparse

def generate_feed_id(feed_url: str) -> str:
    """Generate stable unique identifier for feed.

    Args:
        feed_url: RSS feed URL

    Returns:
        Stable unique identifier string (format: sha256:<hex_digest>)
    """
    # Normalize feed URL (remove trailing slash, lowercase, remove query params/fragments)
    parsed = urlparse(feed_url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/').lower()

    # Generate SHA-256 hash
    hash_digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    return f"sha256:{hash_digest}"
```

**Benefits**:

- Stable across runs (same feed URL = same ID)
- Unique (different feeds = different IDs)
- Database-friendly (string format)
- Deterministic and collision-resistant

#### Episode ID Generation

Episode ID generation follows this priority:

1. **RSS GUID** (if available): Use the RSS item's `<guid>` tag value directly
   - Most reliable source as it's explicitly provided by the feed
   - Already unique and stable across runs
   - Format: Use as-is (may be URL, UUID, or other format)

2. **Deterministic Hash** (fallback if no GUID):
   - Generate SHA-256 hash from stable components:
     - Feed URL (normalized)
     - Episode title (normalized)
     - Published date (ISO 8601 format, if available)
     - Episode link/URL (if available, as additional uniqueness factor)
   - Format: `sha256:<hex_digest>` (e.g., `sha256:a1b2c3d4...`)
   - Ensures same episode = same ID across runs
   - Collision-resistant for practical purposes

3. **Composite Key** (fallback if no published date):
   - Format: `feed:<normalized_feed_url>:episode:<normalized_title>:idx:<episode_number>`
   - Less ideal but provides uniqueness when other fields unavailable

**Implementation**:

```python
import hashlib
from urllib.parse import urlparse

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

    Returns:
        Stable unique identifier string
    """
    # Priority 1: Use RSS GUID if available
    if episode_guid:
        return episode_guid.strip()

    # Priority 2: Generate deterministic hash
    # Normalize feed URL (remove trailing slash, lowercase)
    normalized_feed = urlparse(feed_url).geturl().rstrip('/').lower()

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
        normalized_link = urlparse(episode_link).geturl().rstrip('/').lower()
        hash_components.append(normalized_link)

    # Generate SHA-256 hash
    hash_input = '|'.join(hash_components).encode('utf-8')
    hash_digest = hashlib.sha256(hash_input).hexdigest()

    return f"sha256:{hash_digest}"
```

#### Content ID Generation (Optional)

Content IDs are optional and only generated when needed for tracking individual content items separately:

**Transcript ID** (`transcript_id`):

- Generated from transcript URL (normalized)
- Format: `sha256:<hex_digest>` of normalized URL
- Useful for tracking transcript availability across episodes

**Media ID** (`media_id`):

- Generated from media URL (normalized)
- Format: `sha256:<hex_digest>` of normalized URL
- Useful for tracking media files across episodes

```python
def generate_content_id(content_url: str) -> str:
    """Generate stable unique identifier for content item (transcript or media).

    Args:
        content_url: URL of the content item

    Returns:
        Stable unique identifier string (format: sha256:<hex_digest>)
    """
    # Normalize URL (remove trailing slash, lowercase, remove query params/fragments)
    parsed = urlparse(content_url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/').lower()

    # Generate SHA-256 hash
    hash_digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    return f"sha256:{hash_digest}"
```

**When to use content IDs**:

- When tracking content items in separate tables/collections
- When building content-level indexes or analytics
- When linking episodes to shared content (same transcript/media used by multiple episodes)

**When not needed**:

- If content is only accessed via episode metadata
- If URLs are sufficient identifiers for your use case

#### Database Usage

**Feed ID**:

- **PostgreSQL**: Use `feed.feed_id` as PRIMARY KEY in feed-level tables
- **MongoDB**: Use `feed.feed_id` as `_id` in feed collections
- **Elasticsearch**: Use `feed.feed_id` as document `_id` in feed indices
- **ClickHouse**: Use `feed.feed_id` in ORDER BY clause for feed tables

**Episode ID**:

- **PostgreSQL**: Use `episode.episode_id` as PRIMARY KEY or UNIQUE constraint
- **MongoDB**: Use `episode.episode_id` as `_id` field (or create unique index)
- **Elasticsearch**: Use `episode.episode_id` as document `_id`
- **ClickHouse**: Use `episode.episode_id` as ORDER BY key or primary key

**Content IDs** (if used):

- **PostgreSQL**: Use `transcript_id`/`media_id` as PRIMARY KEY in content tables
- **MongoDB**: Use as `_id` in content collections
- **Elasticsearch**: Use as document `_id` in content indices
- **ClickHouse**: Use in ORDER BY clause for content tables

**Benefits**:

- Stable across runs (same feed/episode/content = same ID)
- Unique across feeds (includes feed URL in hash)
- Database-friendly (string format works in all databases)
- Deterministic (no random UUIDs, reproducible)
- Collision-resistant (SHA-256 provides sufficient entropy)
- Enables relational queries (join episodes to feeds, link content to episodes)

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
- Tests for database compatibility (JSON structure, ISO 8601 dates, snake_case fields)

## Database Integration Design Principles

### Unified Format Strategy

**Decision**: Use a single JSON schema that works across all target databases without format variations.

**Rationale**:

- JSON is the universal format supported natively by PostgreSQL (JSONB), MongoDB, Elasticsearch, and ClickHouse
- Format variations add complexity and maintenance burden
- Database-specific optimizations can be handled at ingestion time (indexing, flattening) rather than at generation time

### Schema Design Principles

1. **Field Naming**: Use `snake_case` for all field names
   - Compatible with SQL databases (avoids keyword conflicts)
   - Works naturally in MongoDB and Elasticsearch
   - ClickHouse supports both snake_case and camelCase, but snake_case is more universal

2. **Date/Time Serialization**: All datetime fields serialized as ISO 8601 strings
   - Format: `"2025-01-15T10:30:00Z"` or `"2025-01-15T10:30:00+00:00"`
   - Easily parsed by all databases
   - Supports timezone-aware queries
   - Can be indexed and queried efficiently

3. **Nested Structures**: Logical grouping (feed, episode, content, processing)
   - **Document databases** (MongoDB, Elasticsearch): Nested objects work natively
   - **PostgreSQL JSONB**: Supports nested queries with `->` and `->>` operators
   - **ClickHouse**: JSON column type supports nested field access
   - **Relational databases**: Can be flattened if needed using views or ETL

4. **Array Handling**: Arrays for multi-value fields (hosts, guests, transcript URLs)
   - Native support in all target databases
   - Enables array queries (contains, intersection, etc.)

5. **Type Consistency**: Consistent data types across all fields
   - Strings: Always strings (no mixed types)
   - Numbers: Integers for counts/durations, floats only when needed
   - Booleans: Explicit boolean values (not strings like "true"/"false")
   - Nulls: Use `null` (not empty strings or missing fields)

### Database-Specific Considerations

#### PostgreSQL (JSONB)

- Nested queries: `metadata->'episode'->>'title'`
- Array queries: `metadata @> '{"content": {"detected_guests": ["John"]}}'`
- Indexing: GIN indexes on JSONB columns for fast queries
- Flattening: Can create views with flattened columns if needed
- **Primary Key**: Use `metadata->'episode'->>'episode_id'` as PRIMARY KEY or UNIQUE constraint

#### MongoDB

- Document structure matches MongoDB document model exactly
- Nested queries: `db.episodes.find({"content.detected_guests": "John"})`
- Indexing: Can index nested fields directly
- **Document ID**: Use `episode.episode_id` as `_id` field: `db.episodes.insertOne({_id: doc.episode.episode_id, ...doc})`
- No transformation needed

#### Elasticsearch

- JSON is native document format
- Nested objects can be mapped as nested type for better querying
- Arrays are automatically handled
- Full-text search on text fields (title, description)
- **Document ID**: Use `episode.episode_id` as document `_id` in bulk operations

#### ClickHouse

- JSON column type supports nested field access
- Can query nested fields: `metadata.content.detected_guests`
- Can create materialized columns for frequently queried nested fields
- Supports JSONEachRow format for bulk loading
- **Primary Key**: Use `metadata.episode.episode_id` in ORDER BY clause or as primary key

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

## Database Loading Examples

### Example: PostgreSQL JSONB

```sql
-- Create table
CREATE TABLE episode_metadata (
    id SERIAL PRIMARY KEY,
    episode_guid VARCHAR(255) UNIQUE,
    metadata JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create GIN index for fast queries
CREATE INDEX idx_metadata_gin ON episode_metadata USING GIN (metadata);

-- Load single file (using episode_id from metadata)
INSERT INTO episode_metadata (episode_id, metadata)
SELECT
    metadata->'episode'->>'episode_id' as episode_id,
    metadata::jsonb
FROM (SELECT '{"feed": {...}, "episode": {"episode_id": "sha256:abc123..."}, ...}'::jsonb as metadata) as m
ON CONFLICT (episode_id) DO UPDATE SET metadata = EXCLUDED.metadata;

-- Load from file (using COPY or external tool)
-- Note: PostgreSQL COPY doesn't support JSONB directly, use psql or application code

-- Query examples
SELECT metadata->'episode'->>'title' as title
FROM episode_metadata
WHERE metadata->'episode'->>'published_date' > '2025-01-01';

SELECT * FROM episode_metadata
WHERE metadata @> '{"content": {"detected_guests": ["John Doe"]}}';

SELECT metadata->'content'->'detected_hosts' as hosts
FROM episode_metadata
WHERE jsonb_array_length(metadata->'content'->'detected_hosts') > 0;
```

### Example: MongoDB

```javascript
// Load single document (using episode_id as _id)
const doc = {
  feed: {...},
  episode: {episode_id: "sha256:abc123...", ...},
  content: {...},
  processing: {...}
};
db.episodes.insertOne({_id: doc.episode.episode_id, ...doc});

// Bulk load from JSON file
db.episodes.insertMany(JSON.parse(fs.readFileSync('/path/to/metadata.json', 'utf8')));

// Create indexes
db.episodes.createIndex({"episode.published_date": 1});
db.episodes.createIndex({"content.detected_guests": 1});
db.episodes.createIndex({"content.detected_hosts": 1});

// Query examples
db.episodes.find({"content.detected_guests": "John Doe"});
db.episodes.find({"episode.published_date": {"$gte": "2025-01-01"}});
db.episodes.find({"content.detected_hosts": {"$in": ["Jane Host"]}});
```

### Example: Elasticsearch

```bash
# Create index with mapping
PUT /episodes
{
  "mappings": {
    "properties": {
      "feed": {"type": "object"},
      "episode": {
        "type": "object",
        "properties": {
          "title": {"type": "text"},
          "published_date": {"type": "date"},
          "description": {"type": "text"}
        }
      },
      "content": {
        "type": "object",
        "properties": {
          "detected_hosts": {"type": "keyword"},
          "detected_guests": {"type": "keyword"}
        }
      }
    }
  }
}

# Bulk load metadata files (using episode_id as document _id)
# Format: {"index": {"_id": "episode_id"}}\n{"feed": {...}, "episode": {...}, ...}\n
curl -X POST "localhost:9200/episodes/_bulk" \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @metadata_bulk.json

# Query examples
GET /episodes/_search
{
  "query": {
    "match": {
      "content.detected_guests": "John Doe"
    }
  }
}

GET /episodes/_search
{
  "query": {
    "range": {
      "episode.published_date": {
        "gte": "2025-01-01"
      }
    }
  }
}
```

### Example: ClickHouse

```sql
-- Create table with JSON column, using episode_id as ordering key
CREATE TABLE episode_metadata (
    metadata JSON
) ENGINE = MergeTree()
ORDER BY (metadata.episode.episode_id);

-- Load from JSON file (JSONEachRow format)
INSERT INTO episode_metadata
SELECT * FROM file('/path/to/metadata.json', JSONEachRow);

-- Create materialized columns for frequently queried fields
ALTER TABLE episode_metadata
ADD COLUMN episode_title String MATERIALIZED metadata.episode.title,
ADD COLUMN published_date Date MATERIALIZED toDate(metadata.episode.published_date);

-- Query examples
SELECT metadata.episode.title
FROM episode_metadata
WHERE has(metadata.content.detected_guests, 'John Doe');

SELECT * FROM episode_metadata
WHERE toDate(metadata.episode.published_date) >= '2025-01-01';
```

## Open Questions

- Should metadata include full RSS item XML? (Decision: No, keep structured)
- Should metadata include checksums/hashes? (Future consideration)
- Should metadata support incremental updates? (Decision: Regenerate on each run)
- Should metadata include transcript excerpts? (Decision: No, transcripts are separate files)
- Do we need database-specific format variations? (Decision: No, unified JSON with snake_case and ISO 8601 dates works universally across all target databases)

## References

- PRD-004: Per-Episode Metadata Document Generation
- RFC-010: Automatic Speaker Name Detection
- RFC-004: Filesystem Layout & Run Management
- RFC-001: Workflow Orchestration
- Pydantic documentation: <https://docs.pydantic.dev/>
- JSON Schema: <https://json-schema.org/>
