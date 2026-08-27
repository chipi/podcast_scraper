"""Read-only dry run of the #1685 bare-name backfill, against a real corpus.

WHY THIS EXISTS AS ITS OWN ENTRY POINT

m0007 rewrites ``.gi.json`` and ``.kg.json`` in place and there is no reverse migration, so the
one thing worth having before running it on production is an honest answer to "what would this
change?" — measured on the real corpus, not on a 36-episode fixture.

The instruments that already existed did not give that:

* ``drill-corpus-upgrade`` answers it, but only inside a full DR cycle — OpenTofu apply of a real
  Hetzner host, deploy, restore, exercise, destroy, orphan sweep. That is the right tool for
  proving disaster recovery end to end and the wrong one for rehearsing a single migration.
* ``podcast-scraper upgrade plan`` answers it, but imports the whole migration registry — m0001
  (FAISS→Lance) and m0002 (two-tier native reindex) pull lancedb and friends. ``inspect-prod-
  corpus.yml`` runs our code MOUNTED over the deployed image, and its own comment warns that only
  holds "while our code needs no dependency the deployed image lacks". Importing the registry to
  ask about one migration is a needless bet against that.

So: instantiate m0007 alone and call ``plan()``, which is already documented as a pure read. No
registry, no ledger, no writes. stdlib plus our own modules, which is what the mount can carry.

HOW TO READ THE OUTPUT

``heal`` is the number the policy decision hangs on. The corpus measurement (2026-08-27) put
genuinely-resolvable bare names at 12 of 215 occurrences (5.6%), and the agreed backfill policy is
``heal=False`` — scope everything, because a wrong scoping is reversible and a wrong heal writes a
real person's id onto someone else's content and is not. This prints BOTH so the choice is made
against the corpus rather than against a memory of it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .migration import MigrationContext
from .migrations.m0007_scope_bare_person_names import ScopeBarePersonNamesMigration


def plan_for(corpus_root: Path, *, heal: bool) -> str:
    """m0007's own ``plan()`` at a given heal policy. Pure read; nothing is written."""
    migration = ScopeBarePersonNamesMigration()
    # `heal` is a CLASS constant on the migration (documented there: a migration runs against a
    # corpus directory, not a pipeline config, so there is no Config in scope). Overriding it on
    # the INSTANCE is the supported way to ask the other question — the migration's own docstring
    # says "Override on the instance if you want a conservative backfill".
    migration.heal = heal  # type: ignore[misc]
    ctx = MigrationContext(
        corpus_root=corpus_root,
        dry_run=True,
        logger=logging.getLogger("upgrade.dry-run"),
    )
    return migration.plan(ctx)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print what m0007 would do, at both heal policies. Returns 0 unless the corpus is missing."""
    parser = argparse.ArgumentParser(
        prog="python -m podcast_scraper.upgrade.dry_run_m0007",
        description="Read-only: what would the #1685 bare-name backfill change?",
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.corpus_root
    out: List[str] = []
    out.append("### m0007 dry run — what the #1685 backfill would change (READ-ONLY)")
    if not root.is_dir():
        out.append(f"- ⚠ corpus root does not exist: `{root}`")
        print("\n".join(out))
        return 1

    out.append(f"- corpus: `{root}`")
    out.append("")
    for heal, label in (
        (False, "heal=False — the agreed backfill policy, scopes everything"),
        (True, "heal=True — shown for comparison; NOT what we intend to run"),
    ):
        try:
            summary = plan_for(root, heal=heal)
        except Exception as exc:  # noqa: BLE001 — a dry run must never be the thing that breaks
            out.append(f"- **{label}**")
            out.append(f"    - ⚠ plan failed: `{type(exc).__name__}: {exc}`")
            continue
        out.append(f"- **{label}**")
        out.append(f"    - {summary}")
    out.append("")
    out.append(
        "- `healed` is the count that matters: it is the only branch that writes a REAL person's "
        "id onto content, and the only one with no cheap undo. At heal=False it must be 0."
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
