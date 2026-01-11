# RFC-002: RSS Parsing & Episode Modeling

- **Status**: Completed
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, data ingestion contributors
- **Related PRD**: `docs/prd/PRD-001-transcript-pipeline.md`
- **Related ADRs**:
  - [ADR-002: Security-First XML Processing](../adr/ADR-002-security-first-xml-processing.md)

## Abstract

Document the approach for safely parsing podcast RSS feeds, extracting transcript references, and materializing `Episode` dataclasses that downstream modules consume.

## Problem Statement

RSS feeds vary in structure, namespace usage, and XML quirks. We require a predictable parsing layer that is resilient to malformed feeds, respects Podcasting 2.0 extensions, and generates rich episode metadata for the pipeline.

## Constraints & Assumptions

- Feeds must be fetched over HTTP/HTTPS; validation occurs prior to parsing.
- XML parsing must be hardened against malicious inputs (use `defusedxml`).
- Episode titles are used in filenames and logs, so sanitization is critical to avoid filesystem issues (see RFC-004).

## Design & Implementation

1. **Parsing strategy**
   - Use `defusedxml.ElementTree.fromstring` to load XML with protections against billion laughs, etc.
   - Identify `<channel>` root regardless of namespaces; fallback iterators handle non-standard feeds.
2. **Title extraction**
   - Extract channel title for logging only.
   - Episode titles derived from `<title>` heuristics with fallback to `episode_{idx}`.
3. **Transcript detection**
   - Traverse each item, capturing `<podcast:transcript>` and `<transcript>` elements.
   - Resolve relative URLs with `urljoin(feed.base_url, candidate)`.
   - Deduplicate candidates to avoid redundant downloads.
4. **Media enclosure discovery**
   - Locate `<enclosure>` elements for Whisper fallback.
   - Capture both URL and media type (if provided) for downstream extension inference.
5. **Episode model**
   - Create `models.Episode` with index, titles (raw + sanitized), transcript candidates, and media metadata.
   - Keep raw ElementTree node for potential future metadata extraction.

## Key Decisions

- **Namespace-agnostic search**: Instead of hardcoding namespace prefixes, suffix matching ensures compatibility with more feeds.
- **Lazy extension inference**: Leave transcript extension derivation to episode processing (more context available there).
- **Safe default titles**: Use sanitized filenames and fallback names to guarantee deterministic paths.

## Alternatives Considered

- **External RSS parser libraries**: Considered `feedparser`, but raw ElementTree offers more control and fewer dependencies.
- **Full DOM normalization**: Rejected due to overhead; targeted search suffices.

## Testing Strategy

- Fixtures in `tests/test_podcast_scraper.py` cover varied RSS shapes (namespace differences, missing attributes, relative URLs).
- Regression tests ensure new edge cases (e.g., uppercase tags) are covered before release.

## Rollout & Monitoring

- Parsing errors raise `ValueError` surfaced through CLI/automation logs.
- Future schema evolutions can extend `models.Episode` without breaking current consumers.

## References

- Source: `podcast_scraper/rss_parser.py`
- File naming rules: `docs/rfc/RFC-004-filesystem-layout.md`
- Transcript download logic: `docs/rfc/RFC-003-transcript-downloads.md`
