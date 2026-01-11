# ADR-002: Security-First XML Processing

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-002](../rfc/RFC-002-rss-parsing.md)
- **Related PRDs**: [PRD-001](../prd/PRD-001-transcript-pipeline.md)

## Context & Problem Statement

Podcast RSS feeds are external XML documents. Standard XML parsers are vulnerable to several classes of attacks, most notably the "Billion Laughs" (entity expansion) attack and external entity injection (XXE).

## Decision

We mandate the use of the `defusedxml` library for all XML parsing across the project. No module is permitted to use the standard library `xml.etree.ElementTree` or `xml.dom.minidom` directly on untrusted input.

## Rationale

- **Safety**: `defusedxml` provides a drop-in replacement that disables dangerous XML features by default.
- **Institutional Quality**: Enforcing this at the architectural level ensures that even new feed parsers remain secure without the developer needing to remember specific security flags.

## Alternatives Considered

1. **Standard Library with Flags**: Rejected as it is error-prone and differs across Python versions.
2. **External Feed Parsers**: Considered `feedparser`, but it adds significant dependency weight; raw `defusedxml` offers better control.

## Consequences

- **Positive**: Hardened against malicious or malformed RSS feeds.
- **Negative**: Adds one mandatory external dependency.

## References

- [RFC-002: RSS Parsing & Episode Modeling](../rfc/RFC-002-rss-parsing.md)
