# ADR-074: Multi-Feed Corpus Parent Layout and Machine-Readable Manifest

- **Status**: Accepted
- **Date**: 2026-04-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md), [RFC-004](../rfc/RFC-004-filesystem-layout.md)
- **Related PRDs**: (tracked via issues [#440](https://github.com/chipi/podcast_scraper/issues/440), [#444](https://github.com/chipi/podcast_scraper/issues/444))

## Context & Problem Statement

Single-feed output layout (per [ADR-003](ADR-003-deterministic-feed-storage.md) and
[ADR-004](ADR-004-flat-filesystem-archive-layout.md)) does not define how **N feeds** share one
**corpus parent**, how **append/resume** keeps stable per-feed workspaces, or how **semantic index**,
`gi explore`, and **`podcast serve`** discover metadata when episodes live under
`feeds/<stable_feed_id>/…` instead of a flat `metadata/` tree at the corpus root.

Without a recorded decision, contributors could reintroduce duplicate indexes per feed, break
composite `(feed_id, episode_id)` keys, or treat manifest files as ad hoc logs rather than
operational inputs for CLI, HTTP (`corpus_metrics`), and dashboards.

## Decision

1. **Layout A (corpus parent)**: For multi-feed configurations, the **corpus parent** is the
   authoritative root. It contains `feeds/<stable_feed_id>/` subtrees (each preserving existing
   pipeline folder semantics) and, when built, a **unified** `search/` index at the parent per
   [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) / [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md).
2. **Explicit parent for N > 1**: Two or more feeds require an explicit corpus parent `output_dir`;
   the tool must not infer the parent from only the first RSS URL.
3. **Unified discovery**: Indexing, exploration, and server-side corpus routes use **recursive**
   metadata discovery and composite keys so GI, KG, and search operate on **all** feeds under one
   parent without duplicating artifacts.
4. **Machine-readable summaries**: Where implemented, **`corpus_manifest.json`** and optional
   **`corpus_run_summary.json`** at the corpus parent are **normative operational artifacts** for
   throughput and status (documented in
   [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md)), not informal debug
   dumps.

## Rationale

- **One corpus, one index**: Operators and the viewer expect a single “Corpus root folder” for
  `serve` and search; Layout A preserves that mental model.
- **Append safety**: Stable per-feed directories under append avoid unbounded `run_*` churn and
  align skip/retry with artifact validation and `episode_id`.
- **HTTP and UI alignment**: Aggregates such as **`GET /api/corpus/stats`** and Dashboard charts
  consume the same parent-level artifacts as the CLI ([RFC-071](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)).

## Alternatives Considered

1. **Separate output tree per feed with no shared parent**: Rejected; breaks unified search,
   viewer corpus path, and cross-feed analytics.
2. **SQLite as primary resume store (this phase)**: Rejected for v1 of RFC-063; filesystem and
   `index.json` remain the source of truth ([RFC-051](../rfc/RFC-051-database-projection-gil-kg.md)
   remains future).
3. **Infer corpus parent from first feed only**: Rejected; ambiguous with multiple feeds and
   unsafe for automation.

## Consequences

- **Positive**: Clear boundary for server, indexer, and viewer; manifest files become reviewable
  contracts in PRs and docs.
- **Negative**: More paths to test (recursive discovery, composite keys); docs must stay aligned
  with [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md).
- **Neutral**: Single-feed behavior remains backward compatible when N = 1.

## Implementation Notes

- **Docs**: [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md),
  [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md).
- **Server**: `src/podcast_scraper/server/routes/corpus_metrics.py` (parent-relative manifest and
  `run.json` discovery).
- **Config / CLI**: multi-feed `feeds:` list, `--append`, corpus parent validation.

## References

- [RFC-063: Multi-feed corpus](../rfc/RFC-063-multi-feed-corpus-append-resume.md)
- [RFC-004: Filesystem layout](../rfc/RFC-004-filesystem-layout.md)
- [ADR-051: Per-episode JSON artifacts](../adr/ADR-051-per-episode-json-artifacts-with-logical-union.md)
- [ADR-060: VectorStore protocol](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md)
