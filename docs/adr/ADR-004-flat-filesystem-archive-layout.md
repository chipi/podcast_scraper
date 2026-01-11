# ADR-004: Flat Filesystem Archive Layout

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-004](../rfc/RFC-004-filesystem-layout.md)
- **Related PRDs**: [PRD-001](../prd/PRD-001-transcript-pipeline.md)

## Context & Problem Statement

Podcast episodes generate multiple artifacts (transcripts, metadata, summaries). We need a structure that is easy for humans to browse and for machines to parse.

## Decision

We adopt a **Flat Layout** within the feed run directory. All episode-specific files are stored in the same folder, using a consistent prefix to group related items:
`<idx:04d> - <title_safe>.<ext>`
`<idx:04d> - <title_safe>.metadata.json`

## Rationale

- **Simplicity**: No deep nesting makes it easy to `grep` or browse in a file explorer.
- **Sortability**: The numeric prefix (`04d`) ensures episodes are listed in their chronological/feed order.
- **Resilience**: Flat structures are easier to diff and backup than complex nested trees.

## Alternatives Considered

1. **Per-Episode Subdirectories**: Rejected as it adds navigation friction for small numbers of files.
2. **Content-Type Directories**: Rejected as it separates a transcript from its own metadata.

## Consequences

- **Positive**: High visibility of results; simple implementation of "skip existing" logic.
- **Negative**: Can become cluttered if a single episode generates 10+ different artifact types.

## References

- [RFC-004: Filesystem Layout & Run Management](../rfc/RFC-004-filesystem-layout.md)
