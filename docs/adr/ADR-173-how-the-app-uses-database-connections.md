# ADR-173: How the app uses database connections

- **Status**: Accepted (the operator, 2026-11-06); implemented in `podcast_scraper/db/settings.py` and `podcast_scraper/db/engines.py`
- **Date**: 2026-11-06
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-169](ADR-169-the-database-is-the-source-of-truth.md),
  [ADR-172](ADR-172-module-migration-histories-and-readiness.md)

## Context & Problem Statement

ADR-169 puts the app's state in one Postgres database, so every request now reaches the database.
Postgres holds one backend process per connection, and a server tolerates a few hundred of them,
not thousands. The first two database releases opened a small pool per engine and set nothing else:
no limit on how long a statement or an idle transaction may hold a connection, a pool size fixed in
code, and a wait for a free connection of 30 s (SQLAlchemy's default). Three questions follow.

1. **What bounds the connections a deployment needs**, and so what the rules for using them are.
2. **What stops one stuck statement or abandoned transaction from holding a connection** until the
   pool is empty and every request waits.
3. **What changes when one server no longer holds all the connections the workers want.**

## Decision

1. **Hold a connection only for the duration of a query or a transaction**, never across
   non-database work (an outbound HTTP call, a file read, a model call, a sleep). A request checks
   a connection out per call, runs, and returns it. Work that is not database work happens before
   the connection is taken or after it is returned.
2. **Every app connection carries two server-side timeouts**, set per engine through libpq
   `options` in the engine's `connect_args` (combined with `default_transaction_read_only=on` on
   the read-only engine, never overwriting it):
   - `statement_timeout`, default 15 s (`PODCAST_DB_STATEMENT_TIMEOUT_MS`);
   - `idle_in_transaction_session_timeout`, default 60 s (`PODCAST_DB_IDLE_TX_TIMEOUT_MS`).

   `0` disables one. A statement over its timeout fails with SQLSTATE 57014 (`QueryCanceled`), a
   database error the existing GlitchTip grouping keys on (driver class plus SQLSTATE). Migrations
   (`db/migrate.py`) and the importers get **no** statement timeout and no idle-in-transaction limit,
   set to 0 explicitly because the role may carry a role-level default that a session value
   overrides (the plane roles do in prod): a migration may legitimately
   run long.
3. **The pool is sized from settings**: `PODCAST_DB_POOL_SIZE` and `PODCAST_DB_MAX_OVERFLOW`
   (defaults 5 and 5, as before) and `PODCAST_DB_POOL_TIMEOUT_S` (default 10 s, was 30 s), so a
   starved pool fails fast with SQLAlchemy's `TimeoutError` instead of hanging. `pool_pre_ping`
   stays on. Pool size x engines (two per process) x workers x processes must stay under the
   server's `max_connections`, or under PgBouncer's pool.
4. **The app is kept ready for PgBouncer in transaction mode.** App code holds no per-connection
   session state: no session advisory locks (`pg_advisory_xact_lock` is the form to use), no
   session-level `SET` (connection settings go through `options`; `SET LOCAL` is allowed), no
   LISTEN/NOTIFY, temp tables or `WITH HOLD` cursors.
   `tests/unit/podcast_scraper/db/test_db_session_state_policy.py` fails when app code adds one.
   psycopg 3 prepares a statement server-side after `prepare_threshold` executions, which
   transaction pooling breaks; `PODCAST_DB_PREPARE_THRESHOLD` sets it (a number, or `none` to
   disable), default unchanged at psycopg's 5.
5. **At scale out, PgBouncer in transaction mode** sits between the workers and Postgres once
   workers x pool would exceed what Postgres should hold. Then `PODCAST_DB_PREPARE_THRESHOLD=none`
   is set, and the timeouts are kept: PgBouncer is configured to accept the `options` startup
   parameter (its setting and version support are to be confirmed then), or the timeouts are set
   per role (`ALTER ROLE ... SET`) instead.
   Until then the app connects to Postgres directly.

## Rationale

**Connections follow queries per second times hold time, not users.** By Little's law the number of
connections in use at once is the arrival rate of queries times how long each holds a connection.
An idle user, or a thousand of them, holds none; a request that holds a connection for 2 s while it
calls an outbound API holds the capacity of twenty requests that hold it for 100 ms. So the rule is
about hold time, and the first thing to keep short is anything that is not the database: a
connection is never held across non-database work.

**Timeouts cap the tail of hold time.** One runaway query or one transaction left open by a crashed
handler would otherwise hold a connection indefinitely, and a handful of them empties the pool. The
server enforces both limits, so they hold even when the app code or its process is the thing that
stalled. Setting them in `options` puts them on the connection itself, ahead of any statement.

**A short pool timeout turns exhaustion into an error.** A request that waits 30 s for a connection
has already failed from the client's side and still occupies a worker; failing at 10 s with a clear
error is visible in GlitchTip and frees the worker.

**Transaction pooling multiplexes many clients over few server connections** by handing a server
connection to a client for one transaction only. That is what makes thousands of client connections
cheap, and it is only safe for code that keeps nothing on the session between transactions, which
is why the rule is enforced by a test now, while the app still connects directly.

## Alternatives Considered

1. **Hold a connection for the whole request** (one session per request): simple, and wrong under
   Little's law whenever a request does outbound work.
2. **Timeouts per statement in code** (`SET LOCAL statement_timeout`): easy to forget on one query;
   a server-side default on the connection covers every one, and a query that needs more can raise
   it for its own transaction with `SET LOCAL`.
3. **Timeouts per role (`ALTER ROLE ... SET`)** only: also valid, and the right place behind
   PgBouncer if it drops `options`; but it lives outside the repo and the tests cannot see it.
4. **Pooling in the app only, with larger pools**: moves the limit to the server's
   `max_connections` and its memory; each connection is a process.
5. **PgBouncer from the start**: another component to run and watch, for a load one server holds.

## Consequences

- **Positive**: a stuck statement or an abandoned transaction costs a bounded time, not a
  connection; pool exhaustion fails fast and visibly; moving behind PgBouncer is a setting, not a
  code change, and a test keeps it that way.
- **Negative**: a legitimately long query in app code (an export, a backfill) is cancelled at 15 s
  unless it raises its own limit with `SET LOCAL` or runs outside the API processes; a transaction
  left idle for 60 s loses its connection, and the next statement on it fails.
- **Neutral**: the pool timeout drops from 30 s to 10 s by default.
- **Operations**: size `PODCAST_DB_POOL_SIZE`, `PODCAST_DB_MAX_OVERFLOW`, workers and processes
  against `max_connections`; the settings are validated at start and a bad value stops the process.

## Implementation Notes

- **Settings and defaults**: `podcast_scraper/db/settings.py` (`DatabaseSettings`)
- **Engines and `options`**: `podcast_scraper/db/engines.py` (`app_connect_args`, `_make_engine`)
- **Policy test**: `tests/unit/podcast_scraper/db/test_db_session_state_policy.py`
- **Tests of the timeouts and the pool**: `tests/unit/podcast_scraper/db/test_db_engines.py`

## References

- [ADR-169: The database is the source of truth](ADR-169-the-database-is-the-source-of-truth.md)
- [ADR-172: Module-owned migration histories and readiness](ADR-172-module-migration-histories-and-readiness.md)
- [Database architecture](../architecture/DATABASE.md)
