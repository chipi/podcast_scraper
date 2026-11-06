# ADR-172: Module-owned migration histories, and a readiness check that allows a database ahead

- **Status**: Proposed (2026-11-06); decision 5 (readiness) implemented in `podcast_scraper/db/migrate.py` (`readiness`)
- **Date**: 2026-11-06
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-169](ADR-169-the-database-is-the-source-of-truth.md),
  [ADR-168](ADR-168-installed-extensions.md) (installed extensions),
  [ADR-171](ADR-171-shared-schema-for-cross-plane-rows.md) (the `shared` schema)

## Context & Problem Statement

ADR-169 puts every store into one Postgres database. Installed extensions (ADR-168) keep stores
of their own, so they will own tables of their own. Two things follow.

1. **Who migrates an extension's tables.** The kernel's Alembic history knows only the kernel's
   tables. An extension's tables cannot go into it, because the platform does not know which
   extensions a deployment installs, and an extension must not be able to rewrite kernel tables.
2. **What a process does when the database is ahead of its code.** Prod never migrates at start
   (the migrate step does), and a rollback is "redeploy the previous image on the migrated
   database", which only works when migrations are expand-only and the previous image accepts a
   database it does not fully know. The readiness check shipped with the first two database
   releases refuses whenever the database's revision differs from the code's head, a database
   ahead included (`podcast_scraper/db/engines.py`, `check_ready`). So the previous release does
   not start on the migrated database, and the only rollback that starts is the release before the
   database, on the frozen files.

## Decision

1. **Each installed extension that owns tables declares its own Alembic history** through a field
   on `Extension` (its script location and its metadata), and its importers through another.
2. **Each history has its own version table** in the plane schema
   (`alembic_version_<extension>`) and writes `schema_history` rows naming itself.
3. **`db upgrade --plane P` runs, in order**: `shared`, the kernel, then each installed
   extension's history on the plane. `db status` reports every history and exits 2 when any is
   behind.
4. **Extension tables carry the extension's prefix** in the plane schema and may reference
   kernel tables (`users` with `ON DELETE CASCADE`), so account deletion stays one transaction
   across every installed extension. Kernel tables never reference extension tables.
5. **Readiness, per history**: a process starts when, for each history it knows, the database is
   at the code's head **or ahead of it**. Ahead means the database's revision is one the code does
   not ship and the code's head has a `schema_history` row in that schema (the code cannot walk
   newer revisions; they are not in its script directory). It refuses when the database is behind,
   or at an unknown revision whose history it cannot show held the head, including when its role
   may not read that `schema_history`. A history the code does not have is ignored (an extension
   installed in a later release).
6. **Migrations are expand-only within a release**: a release only adds; a destructive change
   ships a release later, so the previous release keeps working on the migrated database and is
   the rollback.

## Rationale

Ownership follows the code: the repo that owns a store owns its tables and its migrations, and the
platform runs whatever is installed without knowing it in advance. The readiness rule is what makes
"redeploy the previous image" a real rollback; without it, every database release burns its own
rollback, which is what the first two releases did.

## Alternatives Considered

1. **One history for everything, the extensions' revisions included**: the platform would have
   to ship revisions for code it does not contain.
2. **Extensions create their tables at start (`create_all`)**: no versions, no dry run, no undo,
   and a start-time migration, which prod forbids.
3. **Readiness refuses any mismatch** (as shipped): safe against running old code on a database it
   does not know, but it removes the rollback; expand-only migrations make the ahead case safe.

## Consequences

- **Positive**: modules migrate independently; account deletion stays one transaction; a
  rollback to the previous release works once this rule is in the release being rolled back to.
- **Negative**: more histories to run and check; the readiness check reads each history's
  `schema_history`, so the plane roles need SELECT on it (`shared`'s included); a migration that
  is not expand-only breaks the rollback silently, so reviews check it.
- **Neutral**: the first release with this rule cannot be rolled back to one without it: the
  releases before it still refuse a database that is ahead.

## Implementation Notes

- **Extension fields**: `podcast_scraper/extensions.py` (`Extension`)
- **Runner, status and readiness**: `podcast_scraper/db/migrate.py`,
  `podcast_scraper/db/engines.py` (`check_ready`)
- **Kernel and shared histories**: `podcast_scraper/db/migrations/`,
  `podcast_scraper/db/migrations_shared/`

## References

- [ADR-169: The database is the source of truth](ADR-169-the-database-is-the-source-of-truth.md)
- [ADR-171: The `shared` schema](ADR-171-shared-schema-for-cross-plane-rows.md)
- [Database architecture](../architecture/DATABASE.md)
