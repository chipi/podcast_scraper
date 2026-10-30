# ADR-166: Person identity is decided once, in a ledger; without a valid ledger there are no joins

- **Status**: Proposed
- **Date**: 2026-10-30
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-131](../rfc/RFC-131-person-identity-one-home.md)

## Context & Problem Statement

The decision that two person ids are one person across episodes is made today at read time, in
memory. `kg/entity_clusters.build_entity_canonical_map` builds it, `cached_entity_id_map` caches
it, and four surfaces apply it:

- `app_kg_index`;
- `cil_queries`;
- `routes/corpus_persons`;
- `search/corpus_graph`.

It leaves no record. Nobody can say why two ids were joined, which build joined them, or what
changed since the last build. The map's own payload is built and thrown away
(`entity_clusters.py:505`). Fixes that add guards to it (ADR-167) also have to make sure no
failure path quietly brings back the unguarded map.

The operator's rule decides the failure mode: a wrong name shown is worse than a name hidden.

## Decision

1. **One deterministic step writes an identity ledger,** a derived corpus file, as the last stage
   of every corpus-writing path: ingest, migration apply, relabel repair, enrich-edges.
   - It records every cross-episode join, every blocked join, and the display spelling with its
     evidence.
   - Its header carries the schema version, rule version, code sha and corpus token.
   - Each build appends to a change log.
2. **The server reads the ledger** at warm-up, and again when the file changes. It applies the ledger
   at read time on every surface that shows a person.
3. **If the ledger is missing, unreadable, or from another schema or rule version, the server makes
   no cross-episode joins.** It logs a warning and counts the state in a metric. The in-memory
   unguarded build is not a serving fallback.
4. **A corpus newer than the ledger is served from the ledger as it is.**
   - New episodes stay unjoined until the next build.
   - A changed episode keeps its previous joins until the next build.
   - The gap is measured and alerted on.
5. **The write is atomic and owned by the corpus owner.** The ledger file is part of the corpus
   change token that projections key on, and is excluded from its own token.

## Rationale

- **No joins is the only failure mode that cannot show a wrong name.** Every other fallback can
  reintroduce merges the guards exist to block. The in-memory build is one: it merges two
  co-presenters who appear as different voices in one episode.
- **The decision and its record are the same file, so they cannot drift.** A log written next to
  an in-memory decision can disagree with what was served; a ledger the server reads cannot.
- **It extends what exists.** The versioned clusters payload and `id_map_from_clusters_payload`
  already serialise and read the map.

## Alternatives Considered

1. **Keep the in-memory map, and log decisions next to it.** Rejected: the log and the served map
   can diverge, and a restart rebuilds without history.
2. **Fall back to the in-memory build when the ledger is missing.** Rejected: it serves exactly the
   wrong merges the guards were added to stop, and it does so silently.
3. **The api writes the ledger at warm-up.** Rejected: the api does not write into the corpus
   today, and a request-serving process should not own a corpus artifact.

## Consequences

- **Positive:**
  - every join can be explained from one file;
  - the change log shows what each build changed;
  - the server's 90-second warm-up build becomes a file read;
  - rollback is the previous image.
- **Negative:**
  - one more derived corpus file, and a step that must run after every corpus write;
  - until it runs, new episodes are unjoined;
  - a missing ledger makes person pages show one person per spelling until it is rebuilt.
- **Neutral:**
  - the previous image does not read the ledger;
  - per-episode naming detail stays in `*.speakers.diagnostics.json`.

## Implementation Notes

- **Module:** `podcast_scraper/identity/` (the person-identity module of RFC-131); the reader in
  the server's cache warm-up path.
- **Format:** an extension of `ENTITY_CLUSTERS_SCHEMA_VERSION`'s payload.
- **Token:** `perf_cache.corpus_mtime` gains the ledger file name.
- **Measure:** the `name_report` field R7 reports the ledger's state, age, and the episodes newer
  than it.
