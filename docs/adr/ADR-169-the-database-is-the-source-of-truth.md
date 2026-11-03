# ADR-169: The database is the source of truth for app state and the corpus

- **Status**: Accepted (2026-11-03)
- **Date**: 2026-11-03
- **Authors**: Marko Dragoljevic
- **Supersedes**: [ADR-051](ADR-051-per-episode-json-artifacts-with-logical-union.md) ("CLI mode
  requires no database"), the "files remain the canonical source of truth" part of
  [ADR-054](ADR-054-relational-postgres-projection-for-gil-and-kg.md), and decision D2 of
  [PRD-035](../prd/PRD-035-learning-platform.md) ("no persistence layer")
- **Related**: [ADR-116](ADR-116-privilege-split-public-control-api.md),
  [ADR-168](ADR-168-installed-extensions.md), [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md),
  [ADR-099](ADR-099-lancedb-first-single-index-search.md)

## Context & Problem Statement

All state is files today. Per-user app state is JSON and JSONL under `APP_DATA_DIR/users/<id>/`
behind file locks; the corpus is per-episode JSON artifacts in run directories, merged at read
time; operator config and the job queue are files too. That has reached its limits:

- **Cross-user and cross-item queries are directory walks**, several on request paths (episode
  stats over every user's listen events in `server/app_stats.py`, unsubscribe lookup in
  `server/app_comms_store.py`, handle allocation in `server/app_user_store.py`, the outbox poll in
  `server/app_outbox_store.py`, the catalog parse in `server/catalog_cache.py`).
- **Concurrency rests on per-directory file locks**: some writers take none, a per-feed job locks
  a different directory from a corpus-wide one, and appends are check-then-write without a lock
  (`server/app_user_state.py`).
- **Caches are keyed on the mtimes of a few stamp files** (`perf_cache.py`); a writer that touches
  none of them serves stale data.
- **One fact is copied into many files** (feed metadata into every episode, speaker names into
  transcripts and graphs), so a rename or a correction rewrites thousands of files and moves char
  offsets; migrations, receipts and undo exist largely to manage that.
- **"Newest run wins"** is re-applied by every reader, with two different rules.

ADR-054 chose a rebuildable Postgres projection over canonical files. A projection keeps every
one of these problems on the write side.

## Decision

1. **The database is the source of truth** for app state and for the corpus. Corpus files are
   eliminated; media, artwork and raw fetch caches stay as files referenced by rows. Exports to
   JSON remain for inspection.
2. **Postgres 18 in every edition.** The public compose stack runs it as a service; there is no
   SQLite edition and no file mode. Access through SQLAlchemy Core with Alembic migrations and the
   psycopg 3 driver.
3. **One database, schemas by lifecycle and privilege**: one schema per API plane (ADR-116) for
   users and per-user state, and a `corpus` schema for the corpus, operator config, jobs and
   leases. Roles and grants enforce the plane split; no corpus role reads user tables.
4. **App state moves first**, then the corpus. The corpus moves in one freeze-window cut-over:
   files stay canonical until then, a full import plus a served-surface parity check gates it,
   and frozen files are the rollback.
5. **Modeling rules**: relational for anything filtered or joined across items, JSONB only for
   data read whole; derived artifacts (bridge, catalog, context digest) become views; transcripts
   are stored as segments and rendered, with evidence anchored to a segment range; one revision
   counter bumped by every content transaction replaces the mtime stamps; existing id values never
   change.
6. **Concurrency**: a lease table replaces the file locks; short transactions; long jobs re-check
   their lease in every chunk.
7. **Migrations**: expand/contract only; prod migrates in a dedicated step, never at app start;
   no downgrades in prod.
8. **Search stays on LanceDB** until a later step moves it into Postgres behind the existing
   search seam, gated on search-evaluation parity.

## Rationale

- Every listed problem is a query, a transaction or a uniqueness rule, which a relational
  database provides and files do not.
- At this scale (a few thousand items, a few GB of corpus text, a single host) one Postgres on the
  same host is the least machinery; Postgres features and mature extensions cover search, time
  ranges and queues later without new services.
- One dialect everywhere keeps one code path, one migration history and one test matrix.

## Alternatives Considered

1. **Keep files, add caches**: keeps directory walks, file-lock races and copied facts.
2. **Postgres as a rebuildable projection (ADR-054)**: fixes reads, not writes; durable state
   (cluster ids, enricher cursors, migration receipts) is not rebuildable.
3. **SQLite in the public edition, Postgres in prod**: two dialects, no range types, exclusion
   constraints, `SKIP LOCKED` or extensions in public; rejected for one dialect.
4. **An embedded analytical engine (DuckDB) as the serving store**: single-writer, a second
   dialect; kept only as an offline analytics tool.
5. **A database per domain**: turns cross-domain queries back into application-side joins;
   schemas with grants give the same isolation.

## Consequences

- **Positive**: transactional writes across what are files today; indexed queries for the catalog,
  people, topics and per-user stats; one change counter for caches; account deletion by cascade;
  consistent backups by `pg_dump`.
- **Negative**: the public edition needs a database service (compose); every corpus writer
  (pipeline stages, enrichers, identity, migrations) is rewritten to rows; the corpus-change
  method and its tooling are reworded for rows before the first database write.
- **Neutral**: a one-time import of app state and of the corpus; LanceDB stays until its own step.

## Implementation Notes

- **Dependencies**: `sqlalchemy`, `alembic`, `psycopg` added to `pyproject.toml`.
- **App state**: store modules under `src/podcast_scraper/server/app_*_store.py` keep their
  functions and move their internals to SQL; a one-time `db import --from-files` verifies every
  record.
- **Corpus**: a repository layer of named query intents is the only read surface; the existing
  `upgrade` framework's `StateStore` protocol takes a database backend.

## References

- [ADR-051](ADR-051-per-episode-json-artifacts-with-logical-union.md),
  [ADR-054](ADR-054-relational-postgres-projection-for-gil-and-kg.md),
  [ADR-099](ADR-099-lancedb-first-single-index-search.md),
  [ADR-116](ADR-116-privilege-split-public-control-api.md),
  [ADR-168](ADR-168-installed-extensions.md)
- [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md),
  [PRD-035](../prd/PRD-035-learning-platform.md)
