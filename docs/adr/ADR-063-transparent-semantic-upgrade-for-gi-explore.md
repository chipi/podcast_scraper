# ADR-063: Transparent Semantic Upgrade for gi explore

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-061](../rfc/RFC-061-semantic-corpus-search.md), [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md)
- **Related PRDs**: [PRD-021](../prd/PRD-021-semantic-corpus-search.md)

## Context & Problem Statement

`gi explore --topic` and `gi query` currently use substring matching to find relevant
insights. Semantic search (RFC-061) adds a vector index that enables meaning-based
matching. The system needs to upgrade these commands to use semantic matching when
available — but without breaking behavior for users who have not built a vector index.

## Decision

We adopt a **transparent semantic upgrade** pattern:

1. **Auto-detection**: When `gi explore` or `gi query` starts, it checks if a vector
   index exists at the default path (`<output_dir>/search/`).
2. **Semantic path**: If an index exists, topic/question matching uses
   `VectorStore.search()` to find semantically similar insights.
3. **Substring fallback**: If no index exists, the existing substring matching logic
   runs unchanged. Same `ExploreOutput` contract, same CLI flags, same output format.
4. **Zero configuration**: No flag required to activate semantic matching. The presence
   of the index is the trigger.
5. **Same output contract**: Both paths produce `ExploreOutput` (RFC-050). Consumers
   see the same shape — just better results when semantic matching is active.

## Rationale

- **Zero regression**: Users who never enable `vector_search` see identical behavior.
  No flag to learn, no config to change.
- **Gradual adoption**: Build the index when ready; `gi explore` automatically improves.
- **Single command surface**: Users don't need to learn `podcast search` to benefit.
  Their existing `gi explore --topic` workflow gets better transparently.
- **Reusable pattern**: The "detect capability, use if available, fallback if not"
  pattern can apply to future KG queries or cross-layer search.

## Alternatives Considered

1. **Require explicit `--semantic` flag**: Rejected; adds cognitive load, fragments the
   command surface, users must remember which mode they're in.
2. **Separate `gi search` command**: Rejected; duplicates the explore surface. The
   existing `podcast search` CLI handles dedicated search. `gi explore` should just
   get better.
3. **Always require vector index**: Rejected; forces all users to build an index even
   for simple substring queries on small corpora.

## Consequences

- **Positive**: Seamless upgrade path. No breaking changes. No new commands to learn.
  Better results with zero effort once the index exists.
- **Negative**: Users may not realize their results improved (or degraded) because of
  index presence. Mitigated by logging which matching path was used.
- **Neutral**: Explore function gains an optional `vector_store` parameter. Internal
  change only; external contract unchanged.

## Implementation Notes

- **Module**: `src/podcast_scraper/gi/explore.py`
- **Function**: `_insight_matches_topic(artifact, insight_id, insight_text, topic,
  vector_store=None)` — if `vector_store` is provided, use semantic matching
- **Detection**: Check for `<output_dir>/search/index_meta.json` at explore startup
- **Logging**: `INFO: Using semantic matching (vector index found)` or
  `INFO: Using substring matching (no vector index)`

## References

- [RFC-061: Semantic Corpus Search — gi explore upgrade](../rfc/RFC-061-semantic-corpus-search.md)
- [RFC-050: GIL Use Cases — UC4/UC5](../rfc/RFC-050-grounded-insight-layer-use-cases.md)
- [ADR-060: VectorStore Protocol](ADR-060-vectorstore-protocol-with-backend-abstraction.md)
