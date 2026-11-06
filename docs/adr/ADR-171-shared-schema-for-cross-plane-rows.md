# ADR-171: A `shared` schema for the rows both planes read and write

- **Status**: Accepted (2026-11-06, deployed in release f3606a2)
- **Date**: 2026-11-06
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-169](ADR-169-the-database-is-the-source-of-truth.md) (the database is the
  source of truth), [ADR-116](ADR-116-privilege-split-public-control-api.md) (the API planes),
  [ADR-172](ADR-172-module-migration-histories-and-readiness.md) (migration histories)

## Context & Problem Statement

ADR-169 gives each API plane its own schema (`player`, `operator`) with its own role, and keeps
the planes' accounts and grants apart: a plane role cannot read the other plane's tables. Kernel
tables are defined once, without a schema, and placed into the process's plane by the engine's
`schema_translate_map`, so the same history runs once per plane.

One kernel store does not fit: the released version of each client app (the "update available"
prompt). The admin field that sets it is in the operator viewer, the phones read it from the
player API, and both must see one value. When it was a file, both APIs mounted one small shared
directory (`APP_RELEASE_DIR`) for exactly this, after a bug where the operator wrote its own copy
and the phones never saw it (#2296). A table in each plane's schema would bring that bug back; a
table in the player schema with an exception grant for the operator role would make a hole in the
planes' separation.

## Decision

1. A third schema, **`shared`**, holds only the rows both planes must read and write. Today that
   is one table, `shared.app_releases` (`app` primary key, `version`, `updated_at`, both values
   CHECKed exactly as the store validated them).
2. `shared` tables **name their schema** (their own `MetaData(schema="shared")`), so the planes'
   `schema_translate_map` leaves them alone. Kernel tables still never name one.
3. `shared` has **its own Alembic history**, with its own `alembic_version` and `schema_history`
   in `shared`. `db upgrade --plane P` brings `shared` to its head first, in its own transaction,
   then the plane; `db status` and the start-up readiness check cover both.
4. The deploy creates the schema owned by the migrating role and grants each plane role DML on
   `shared.app_releases` and SELECT on `shared.alembic_version` and `shared.schema_history`,
   nothing else; the plane roles cannot write `shared`'s bookkeeping. (`schema_history` added
   2026-11-06: the start-up check reads it to accept a database ahead, ADR-172.)
5. Nothing that holds a plane's user data goes into `shared`. A table belongs there only when both
   planes must read and write the same rows; anything else is plane-local.
6. A process with no database (the control-plane API) has no release override, as it had no file.

## Rationale

The planes' separation is the security property ADR-116 and ADR-169 are built on; a schema whose
whole contents are listed in one grant keeps it intact while giving the one cross-plane value one
home. A separate history keeps `shared` independent of the per-plane kernel history: the kernel
history runs twice (once per plane), `shared`'s once.

## Alternatives Considered

1. **The table in the player schema, the operator granted UPDATE on it**: no new schema, but one
   cross-plane grant inside a plane's schema, the first exception to the separation.
2. **Keep the release override a file in a shared directory**: no database change, but the one
   kernel store left on files, and the shared mount the database was meant to retire.
3. **The table in `public`**: `public` is not a schema the deploy manages, and granting into it
   widens what every role can see.

## Consequences

- **Positive**: one value, read by the phones and written from the operator viewer, with the
  planes' grants unchanged; a pattern for any later cross-plane row.
- **Negative**: a third history to migrate, check and back up; the pre-migration dump covers
  `-n P -n shared`, and the plane-only undo leaves `shared` to be restored by hand.
- **Neutral**: whole-database backups and the restore drill include it.

## Implementation Notes

- **Schema and history**: `podcast_scraper/db/schema.py` (`SHARED_SCHEMA`, `shared_metadata`,
  `app_releases`), `podcast_scraper/db/migrations_shared/`
- **Runner and readiness**: `podcast_scraper/db/migrate.py` (`upgrade`, `upgrade_shared`,
  `shared_status`), `podcast_scraper/db/engines.py` (`check_ready`)
- **Store**: `podcast_scraper/server/app_release_store.py`; the importer reads the old files from
  their own directory (`db import --from-files <release dir> --only releases`)

## References

- [ADR-169: The database is the source of truth](ADR-169-the-database-is-the-source-of-truth.md)
- [Database architecture](../architecture/DATABASE.md)
