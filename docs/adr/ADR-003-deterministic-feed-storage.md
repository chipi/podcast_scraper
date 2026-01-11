# ADR-003: Deterministic Feed Storage

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-004](../rfc/RFC-004-filesystem-layout.md)
- **Related PRDs**: [PRD-001](../prd/PRD-001-transcript-pipeline.md)

## Context & Problem Statement

Users often track multiple feeds. Storing them in a single directory leads to name collisions, and using just the podcast title in the directory name is unreliable (titles change, and different feeds can share titles).

## Decision

Output directories are derived deterministically using a combination of the host name and a hash of the RSS URL:
`output/rss_<sanitized_host>_<hash>`
Where `<hash>` is the first 8 characters of the SHA-1 of the normalized RSS URL.

## Rationale

- **Stability**: The same feed always lands in the same directory, regardless of title changes.
- **Collision Resistance**: Two feeds with the same name (e.g., "The Daily") but different URLs will have unique directories.
- **Safety**: Sanitizing the host name prevents illegal characters in the directory path.

## Alternatives Considered

1. **Timestamped Folders**: Rejected as it makes diffing and skipping existing episodes difficult.
2. **Manual Naming Only**: Rejected to maintain zero-config defaults for the CLI.

## Consequences

- **Positive**: Predictable, resumable runs; easier to automate cleanup and backup scripts.
- **Negative**: Directory names are slightly less human-readable than plain titles.

## References

- [RFC-004: Filesystem Layout & Run Management](../rfc/RFC-004-filesystem-layout.md)
