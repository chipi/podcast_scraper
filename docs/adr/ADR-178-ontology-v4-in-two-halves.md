# ADR-178: Ontology v4 is adopted in two halves: the schema now, the podcast corpus later

- **Status**: Proposed
- **Date**: 2026-11-11
- **Authors**: Marko Dragoljevic
- **Related**: [RFC-134](../rfc/RFC-134-news-as-the-second-content-type.md) (news as the second
  content type), `docs/architecture/kg/kg.schema.json`, `docs/architecture/gi/gi.schema.json`

## Context & Problem Statement

Ontology v4 (the R3 step of the terminology refactor) makes evidence medium-neutral:

- evidence anchored by a character range in its item's text, with a time range only when the item
  is timed;
- `text_origin` (`spoken`, `translated`, `authored`);
- `ATTRIBUTED_TO`;
- roles on people;
- declared `Place` and time fields.

Today's GI schema (3.1) requires `episode_id`, `timestamp_start_ms`, `timestamp_end_ms` and
`transcript_ref` on every quote. A news article has no timestamps and no transcript.

R3 was planned as one step: the schema and the migration of the whole podcast corpus together.
The migration is the large, risky part, and news only needs the schema.

## Decision

1. **R3a, the schema, comes first.** Items of new types write v4 from the start. Every reader of
   insights, quotes and entities accepts both versions.
2. **A 3.x artifact reads as v4:**
   - its quotes carry their character range and their time range;
   - `text_origin` is `spoken`;
   - nothing in the file changes.
3. **Podcasts keep writing 3.x until R3b.**
4. **R3b,** the podcast corpus migration to v4 under the change method, comes in a later release,
   before or with the corpus database. The database schema is then designed once, against v4.
5. **Each schema stays with its owner.** KG's schema is public's and is versioned here. GI's moves
   with the private extension and follows these rules there.

## Consequences

- **Positive:**
  - news ships without the corpus migration and without a freeze for it;
  - v4 is practised on real data in the file corpus before the database schema is fixed.
- **Negative:**
  - two versions live side by side until R3b, and every reader must handle both;
  - tests cover both versions on every route that reads insights or quotes.
- **Neutral:** no podcast artifact changes in this release.

## Alternatives considered

- **R3 in one step, with news waiting for it.** Rejected. News would wait on the largest migration
  of the release.
- **News writes 3.x with placeholder timestamps.** Rejected. The placeholders would be read as real
  times by the player, search and the graph.
- **A separate news-only schema.** Rejected. It would be a third shape to migrate later, instead of
  the one the corpus is moving to.
