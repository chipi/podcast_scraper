"""Run the #1685 bare-name backfill against a corpus — plan, or apply in place.

Sibling of :mod:`dry_run_m0007`, and separate from the ``upgrade`` CLI for the same reason: that
CLI imports the whole migration registry (m0001 FAISS->Lance, m0002 two-tier reindex), which pulls
lancedb. ``scope-bare-names-prod.yml`` runs our code MOUNTED over the deployed image, and that
only holds while our code needs no dependency the image lacks. Asking about ONE migration should
not import every other one.

It also means this entry point deliberately does NOT touch the upgrade ledger. m0007 here is a
targeted data repair, not a schema-version step: the ledger records "this corpus has been taken to
version X", and writing that from a one-off repair would claim more than was done. If the corpus
is later taken through the real upgrade runner, m0007 is idempotent — a second pass plans an empty
map and writes nothing — so the ledger step stays truthful.

``--mode plan`` is the same read-only path as :mod:`dry_run_m0007`. ``--mode apply`` writes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .migration import MigrationContext
from .migrations.m0007_scope_bare_person_names import ScopeBarePersonNamesMigration


def run(corpus_root: Path, *, dry_run: bool, heal: bool) -> str:
    """Plan or apply m0007. Returns its summary line."""
    migration = ScopeBarePersonNamesMigration()
    # `heal` is a class constant on the migration by design (a migration runs against a corpus
    # directory, not a pipeline config). Its docstring names instance override as the supported
    # way to ask for a conservative backfill.
    migration.heal = heal  # type: ignore[misc]
    ctx = MigrationContext(
        corpus_root=corpus_root,
        dry_run=dry_run,
        logger=logging.getLogger("upgrade.m0007"),
    )
    if dry_run:
        return migration.plan(ctx)
    result = migration.apply(ctx)
    return getattr(result, "message", str(result))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Non-zero only when the corpus is missing or the migration raises."""
    parser = argparse.ArgumentParser(
        prog="python -m podcast_scraper.upgrade.apply_m0007",
        description="#1685 bare-name backfill: plan, or rewrite .gi.json/.kg.json in place.",
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("plan", "apply"), default="plan")
    parser.add_argument(
        "--heal",
        default="false",
        help="'true' rewrites a resolvable bare name to the real person's id. Default false.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.corpus_root
    dry_run = args.mode == "plan"
    heal = str(args.heal).strip().lower() in ("1", "true", "yes")

    if not root.is_dir():
        # Loud, and non-zero. A backfill pointed at the wrong path must never print a tidy
        # "0 episodes, nothing to do" and exit 0 — that reads exactly like success.
        print(f"ERROR: corpus root does not exist: {root}", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"m0007 bare-name scoping — mode={args.mode} heal={heal} corpus={root}")
    if not dry_run:
        print("WRITING IN PLACE — there is no reverse migration.")
    try:
        summary = run(root, dry_run=dry_run, heal=heal)
    except Exception as exc:  # noqa: BLE001 — surface it, never a tidy zero
        print(f"ERROR: m0007 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
