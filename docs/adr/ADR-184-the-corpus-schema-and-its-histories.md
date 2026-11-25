# ADR-184: The corpus schema has its own built-in history; a type module's corpus tables live beside it

- **Status**: Accepted (the operator, 2026-11-25)
- **Date**: 2026-11-25
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-169](ADR-169-the-database-is-the-source-of-truth.md) (the database is the source
  of truth), [ADR-171](ADR-171-shared-schema-for-cross-plane-rows.md) (the `shared` schema),
  [ADR-172](ADR-172-module-migration-histories-and-readiness.md) (module histories; amended here),
  [ADR-173](ADR-173-how-the-app-uses-database-connections.md) (connections),
  [ADR-181](ADR-181-a-content-types-own-fields.md) (a type's own fields)

## Context & Problem Statement

The corpus moves from files into one Postgres schema, `corpus`, at the corpus cut-over (ADR-169).
Today the migration runner knows three kinds of history (`podcast_scraper/db/migrate.py`): the
kernel, run once per plane schema; `shared`; and an extension's history, whose tables carry the
extension's prefix in a plane schema (ADR-172 decision 4). `settings.py` says it already: "A later
`corpus` schema is not a plane."

Two things have no home:

1. **The corpus tables themselves.** They are not a plane's: every plane and the pipeline read
   them, and they reference no user. They are not an extension's either: sources, items, documents,
   entities and the graph's edges are the platform's.
2. **A type module's own corpus tables** (ADR-181): the podcast type's word timings and ad maps,
   the news type's re-check state. They are keyed by a corpus item, and the word timings are
   partitioned like the segments they extend. ADR-172 decision 4 puts a module's tables in a plane
   schema, where they cannot sit beside the item they extend.

## Decision

1. **`corpus` is a built-in history, like `shared`.** Scripts in `podcast_scraper/db/migrations_corpus/`,
   metadata `corpus_metadata` in `db/schema.py`, version table `alembic_version` and its own
   `schema_history` in schema `corpus`. `db upgrade` runs `shared`, then `corpus`, then each plane's
   kernel and modules; `db status` and readiness (ADR-172 decision 5) report it like any history.
2. **A module may declare corpus tables** (amends ADR-172 decision 4). A `DbModule` names either
   planes or the `corpus` schema. A corpus module's tables and revision ids carry the module's
   prefix (`podcast_`, `news_`), its version table is `alembic_version_<name>` in `corpus`, it runs
   after the `corpus` history, and its tables may reference corpus tables. Corpus tables never
   reference a module's tables.
3. **Infra owns the schema, the extensions and the grants; a migration only creates objects.** As
   for the planes: the deploy creates `corpus` owned by `migrator` and the `pg_trgm` extension, and
   grants after every migrate, so a new table is never left without grants: `pipeline_rw` reads and
   writes every corpus table, `corpus_ro` reads them, both read the version tables and
   `schema_history` only. No plane role reads or writes `corpus`.
4. **Upgrade-only, expand-only** (ADR-172 decision 6): revisions have no `downgrade()`; the undo
   of the cut-over is the previous release on the frozen files.
5. **What the ORM cannot say is written as SQL in the revision**: hash partitions, partial and
   covering indexes, trigram indexes, and per-table autovacuum settings. A partition key is right in
   the first revision, because expand-only cannot change it; autovacuum settings go on each
   partition, not on the partitioned parent.
6. **Foreign keys go to the parent tables** (`sources`, `content_items`, `documents`, `entities`),
   with `ON DELETE CASCADE` from what an item owns. A range into the partitioned segments
   (`start_seq` to `end_seq`) is checked by the importer and the writers, not by a key.
7. **Search vectors are not in the first revisions.** Search stays on its own index until the search
   epic decides the vector column and its extension.

## Rationale

The `shared` schema already showed the pattern for a schema that is not a plane (ADR-171): a
built-in history the runner always knows. A type module keeps owning its fields (ADR-181), but
the item it extends lives in `corpus`, so its tables must too; the prefix and the one-way
reference rule are ADR-172's, applied to a second schema.

## Alternatives Considered

1. **The corpus as an extension's history**: the platform's own tables would ship as an optional
   module, and every plane's readiness would depend on something that may not be installed.
2. **A module's corpus tables in a plane schema**: the podcast word timings could not share the
   segments' partitioning or reference the item, and a plane would hold pipeline data.
3. **Grants in the migrations**: the migrating role would need to grant to roles it does not own,
   and a grant written in revision N misses the tables of revision N+1; the repo's test forbids it.

## Consequences

- **Positive**: one runner, one readiness rule and one grant step for every schema; a type module
  ships its fields with itself, plane or corpus.
- **Negative**: the grant step and the readiness check grow a schema; the corpus revisions carry
  hand-written SQL that the metadata comparison test does not see, so tests assert partitions,
  partial indexes and settings separately.
- **Deploy**: infra creates `corpus` and `pg_trgm` before the first `db upgrade` that knows it, and
  runs the corpus grant step after each migrate.

## Implementation Notes

- **Runner**: `podcast_scraper/db/migrate.py` (`History`, `upgrade_all`, `status`), `db/engines.py`
  (`check_ready`), `db/cli_handlers.py`
- **Corpus history**: `podcast_scraper/db/migrations_corpus/`, `db/schema.py` (`corpus_metadata`)
- **Module declaration**: `podcast_scraper/extensions.py` (`DbModule`)
