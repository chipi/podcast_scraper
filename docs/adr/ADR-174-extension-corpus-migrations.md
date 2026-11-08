# ADR-174: Installed extensions add corpus migrations to the one registry

- **Status**: Accepted (2026-11-08); implemented in `podcast_scraper/extensions.py`
  (`Extension.corpus_migrations`), `podcast_scraper/upgrade/registry.py` and
  `podcast_scraper/upgrade/outcome_hook.py`
- **Date**: 2026-11-08
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-168](ADR-168-installed-extensions.md) (installed extensions),
  [ADR-172](ADR-172-module-migration-histories-and-readiness.md) (module-owned database
  histories), [Corpus change method](../guides/CORPUS_CHANGE_METHOD.md)

## Context & Problem Statement

A corpus migration (`podcast_scraper/upgrade/migrations/`) rewrites stored artifacts. It is one
step in one ordered sequence (`0001` … `0045`), recorded in the corpus's ledger by id. Since the
change method (from `0031`) the outcome hook checks at apply that the installed code is the code
that was measured: a sha256 over the migration's module closure, the module and every module it
reaches by import.

Two things stop an installed extension (ADR-168) from owning a corpus migration:

1. **The registry is a static list.** `upgrade/registry.py` imports every migration by name. An
   extension cannot add one, and a migration that lives with its code in an extension cannot stay
   runnable, so it cannot be undone or rerun.
2. **The closure stops at this package.** The outcome hook followed imports only into
   `podcast_scraper` and resolved files only under it. An extension's migration would measure as an
   empty closure, so its digest would not cover the code that runs.

Some migrations import code that an extension will own. They have to move with that code and stay
runnable for undo and rerun.

## Decision

1. **An `Extension` field, `corpus_migrations`.** It returns `Migration` instances.
   `get_migrations()` merges them with this package's into one list sorted by id, the apply order.
   The runner, `upgrade verify`, `rerun` and the change gate read that one list, so they see an
   extension's migrations as they see this package's.
2. **One numbering across every package.** An id is `NNNN_name`. When the registry loads it
   refuses:
   - an id of another shape;
   - a retired number (`0032`);
   - a number used twice, whichever packages add it.

   The ledger records ids as applied, and a shared number would leave the order ambiguous between
   packages. A new migration takes the next free number across all installed packages.
3. **The closure follows the migration's own package.** For a migration of package `p`, the
   outcome hook follows imports into `podcast_scraper` and into `p`. It resolves files under each
   package's directory, and names each file in the digest by its path from the directory that holds
   its package (`p/…`, `podcast_scraper/…`).
4. **A migration of this package is measured as before.** Its closure follows `podcast_scraper`
   only, so its closure and digest are those of the previous algorithm. This was checked on the
   same tree, old code against new, for every registered migration.

## Consequences

- **What an extension can do:** ship corpus migrations with the code they import. Prod runs them
  in the same `upgrade run`, under the same outcome hook and the same change method.
- **What a move changes:** moving an existing migration into an extension changes its files'
  digest paths (`podcast_scraper/…` becomes `p/…`), so its digest changes.
  - The outcome hook compares a digest at apply and at `upgrade rerun`.
  - A moved migration that is already applied needs its outcome re-recorded, or a decision that
    reruns of it end with the move. The move decides which, under its own change record.
  - Nothing moves with this decision.
- **What any code change already does:** a migration's closure covers most of this package (621
  modules for `0001`, because the package's `__init__` imports widely). So any change inside it,
  this one included, changes every migration's digest, and a stored outcome then no longer matches
  for `upgrade rerun`. This was true before; this decision does not change it.
- **What it costs:** extensions are loaded when the registry loads. The registry already runs in
  the same processes that load extensions for the server and the CLI.

## Alternatives considered

- **A second registry per extension, run after this package's.** This was rejected. Corpus
  migrations are one sequence over one corpus: a later migration of this package may depend on an
  extension's earlier one. Two sequences would need their own ordering rules and two ledgers.
- **Keeping the moving migrations in this package, with their code moved.** This was rejected. The
  migrations import the moved code, so this package would import the extension's code, and this
  repository does not name private packages (ADR-170).
- **A separate number range per package (for example 9000+).** This was rejected. The order
  across packages would follow from the ranges instead of from when a migration was written, and a
  migration that moves would need a new id. The ledger already names the old one.
