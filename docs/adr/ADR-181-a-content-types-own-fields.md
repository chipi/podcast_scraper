# ADR-181: A content type's own fields live with the type, not on the shared item

- **Status**: Proposed
- **Date**: 2026-11-11
- **Authors**: Marko Dragoljevic
- **Related**: [RFC-134](../rfc/RFC-134-news-as-the-second-content-type.md) (news as the second
  content type), [ADR-172](ADR-172-module-migration-histories-and-readiness.md) (each module's
  own migration history), [ADR-177](ADR-177-content-types-and-transports-are-seams.md) (content types as seams)

## Context & Problem Statement

Every content type shares some fields:

- title;
- the times kept apart by ontology v3 (ADR-178);
- publisher;
- rights;
- language.

Each type also has fields of its own:

- a podcast episode: an audio enclosure, a duration, ad maps;
- a news article: a paywall flag and an editorial type;
- a later filing: a form, a company identifier, a reporting period.

The corpus is files now and moves to the database later (the corpus cut-over). A shape chosen now in
files that does not map cleanly to the database would have to be migrated twice. A shared item table
with a column per type grows with every type and mixes fields only one type uses.

## Decision

1. **Shared fields are on the item.** In files, at the top level of the item's metadata. In the
   database, on the shared item table.
2. **A type's own fields sit under a key named for the type.** In files, for example
   `news_article: {…}`. The key's content belongs to the type that registered it (ADR-177).
3. **In the database, a type's own fields go one of two ways:**
   - **filtered, joined or sorted on:** the type module's own table, keyed by the item, shipped in
     that module's history (ADR-172). Podcast-only tables already work this way.
   - **only displayed:** one JSON column on the shared item table, read whole.
4. **The shared item table never gets a column per type.**
5. **A type moved to a private extension or a product repo keeps its fields there,** with its module
   history. The shared table does not change.

## Consequences

- **Positive:**
  - a new type adds a key and, if needed, a table in its own module, and nothing else;
  - the importer at the cut-over maps one key to one place per type;
  - a private type's fields never appear in public's schema.
- **Negative:**
  - a reader of a type's fields must know the type, which is what the content-type registry
    answers;
  - choosing between a table and the JSON column is a per-field call each type makes.
- **Neutral:** podcast metadata keeps its current top-level fields until R3b (ADR-178). Moving its
  own fields under a `podcast_episode` key is part of that migration, not of this release.

## Alternatives considered

- **A column per type on the shared table.** Rejected. The table would grow with every type, and a
  private type would leak its fields into public's schema.
- **Everything type-specific in one JSON column.** Rejected for fields that are filtered or joined
  on. They need indexes and constraints, which belong to a table.
- **One table per type for everything, shared fields included.** Rejected. Search, the catalog and
  every cross-type surface would need a union across types for the fields they all share.
