# ADR-007: Universal Episode Identity

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-011](../rfc/RFC-011-metadata-generation.md)
- **Related PRDs**: [PRD-004](../prd/PRD-004-metadata-generation.md)

## Context & Problem Statement

To enable long-term archival and database integration, we need a stable way to identify an episode across multiple runs, even if the title or RSS feed structure changes slightly.

## Decision

We adopt a multi-tiered **Universal Episode Identity** strategy:

1. **Primary**: Use the RSS `<guid>` tag. This is the official, stable identifier provided by the feed.
2. **Secondary (Fallback)**: If no GUID is present, generate a deterministic SHA-256 hash based on: `feed_url` + `episode_title` + `published_date`.

## Rationale

- **Interoperability**: Using GUIDs allows our metadata to be joined with other podcasting datasets.
- **Stability**: Content hashes provide a reliable backup that resists title drifts (provided the date remains).
- **Relational Integrity**: Stable IDs enable the "AI Quality Platform" to track model improvements against the same episode over months of testing.

## Alternatives Considered

1. **Random UUIDs**: Rejected as they are not reproducible across different machines or runs.
2. **Filenames as IDs**: Rejected as filenames are sanitized and truncated, making them poor unique keys.

## Consequences

- **Positive**: Reliable database primary keys; easy detection of duplicates in large archives.
- **Negative**: Requires careful normalization of URLs and dates before hashing.

## References

- [RFC-011: Per-Episode Metadata Document Generation](../rfc/RFC-011-metadata-generation.md)
