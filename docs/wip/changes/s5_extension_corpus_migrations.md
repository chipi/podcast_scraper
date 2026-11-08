---
id: s5_extension_corpus_migrations
class: light
status: open
issue: none (E04 release 1, seam S5; the operator, 2026-11-08)
touches:
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/upgrade/outcome_hook.py
  - src/podcast_scraper/extensions.py
  - tests/unit/podcast_scraper/upgrade/
---

# S5 — installed extensions add corpus migrations to the one registry

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4.

## 0. Why light

**What changes.** No stored file changes, and no migration changes or is added. The change touches
three things:

- **Registry loading:** `get_migrations()` also takes the migrations installed extensions declare
  (`Extension.corpus_migrations`). It refuses an id that is not `NNNN_name`, a retired number
  (0032) or a number used twice.
- **Closure resolution:** the outcome hook resolves a migration's module closure in an extension's
  package as well as in this one.
- **Today's behaviour is kept:** no installed extension declares a corpus migration today, so the
  registry stays the 44 in-tree migrations in the same order. For an in-tree migration, the new
  closure and digest algorithm gives what the old one gives on the same tree.
- **Digests still change:** this change edits `outcome_hook.py` and `registry.py`, which lie in
  every migration's closure, so every migration's digest differs from main's. That is true of any
  code change in the package (the closure covers 621 modules). All 44 migrations are applied on
  prod, so no apply is affected. `upgrade rerun` of a migration compares the digest and refuses
  until that migration's outcome is re-measured, as after any earlier code change (C5 the same).

**Why.** The core release moves 18 migrations that import moving code (m0007 … m0043) into Common
with that code (Common `docs/wip/platform-plan/epics/E04-split-line.md` §3 S5). This change makes
the move possible; it moves nothing.

**Proof:**

- the old algorithm (`outcome_hook.py` at origin/main dbf9ba97, loaded from git) against the new
  one, on the same tree, for every registered migration: 44 migrations, closures and digests
  identical ("migrations=44 differing=0", 2026-11-08);
- the tests:
  - a duplicate number, a retired number and a malformed id from an extension are refused;
  - an extension's migration sorts into the sequence by id;
  - an extension package's migration gets a closure that holds its own files and the public
    modules it imports.

Not covered: the move itself, and what a moved migration's digest becomes (its files' paths
change). That is the core release's step, with its own record.
