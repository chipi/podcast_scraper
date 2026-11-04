"""CLI for the database (ADR-169): ``podcast_scraper db upgrade | status | import``.

- ``upgrade --plane P``: bring plane ``P``'s schema to head. The prod migrate step; the API never
  does this at start. ``--sql`` prints the SQL it would run (from the plane's current revision, or
  ``--from REV``) and executes nothing.
- ``status --plane P``: current vs head. Exit **2** when behind (distinct from error=1), so a
  boot script can gate on it.
- ``import --from-files DIR --plane P [--only STORE] [--dry-run]``: read the file stores into the
  database through the registered importers and report every file under DIR by class. Exit 1 on
  any mismatch or any UNCLAIMED file (a dry run included).

The database is ``PODCAST_DATABASE_URL`` (the migrating role in the migrate step).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from sqlalchemy import create_engine

from podcast_scraper.db import inventory, migrate
from podcast_scraper.db.importers import registry
from podcast_scraper.db.settings import (
    DatabaseSettings,
    DEFAULT_PLANE,
    ENV_PLANE,
    ENV_URL,
    PLANES,
)


def _add_plane(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--plane",
        choices=PLANES,
        default=os.getenv(ENV_PLANE) or DEFAULT_PLANE,
        help=f"API plane (schema). Defaults to ${ENV_PLANE}, else {DEFAULT_PLANE}.",
    )


def parse_db_argv(argv: Sequence[str]) -> argparse.Namespace:
    """Parse ``db`` subcommand args into a namespace with ``command='db'``."""
    parser = argparse.ArgumentParser(prog="podcast_scraper db")
    sub = parser.add_subparsers(dest="db_subcommand", required=True, help="Command")
    up = sub.add_parser("upgrade", help="Bring a plane's schema to head")
    _add_plane(up)
    up.add_argument(
        "--sql",
        action="store_true",
        help="Print the SQL the upgrade would run and execute nothing (a dry run).",
    )
    up.add_argument(
        "--from",
        dest="from_revision",
        default=None,
        help="With --sql: start from this revision instead of the plane's current one.",
    )
    _add_plane(sub.add_parser("status", help="Show a plane's current vs head revision"))
    imp = sub.add_parser("import", help="Import file stores into the database and verify")
    _add_plane(imp)
    imp.add_argument("--from-files", required=True, type=Path, help="The APP_DATA_DIR to read.")
    imp.add_argument(
        "--only", default=None, help="Run only this store's importer (the inventory still runs)."
    )
    imp.add_argument("--dry-run", action="store_true", help="Write nothing; report and verify.")
    args = parser.parse_args(list(argv))
    args.command = "db"
    return args


def _upgrade(args: argparse.Namespace, log: logging.Logger) -> int:
    if args.sql:
        url = (os.getenv(ENV_URL) or "").strip() or None
        print(migrate.upgrade_sql(url, args.plane, from_revision=args.from_revision), end="")
        return 0
    url = DatabaseSettings.from_env().rw_url
    applied = migrate.upgrade(url, args.plane)
    current, head = migrate.status(url, args.plane)
    print(f"plane {args.plane}: {current} (head {head})")
    print(f"applied: {', '.join(applied) if applied else 'nothing'}")
    return 0 if current == head else 1


def _status(args: argparse.Namespace, log: logging.Logger) -> int:
    url = DatabaseSettings.from_env().rw_url
    current, head = migrate.status(url, args.plane)
    print(f"plane {args.plane}: current {current or 'none'}, head {head}")
    for row in migrate.history_rows(url, args.plane):
        release = f" release {row['release']} ({row['release_source']})" if row["release"] else ""
        print(
            f"  {row['applied_at']:%Y-%m-%d %H:%M:%S}Z  {row['revision']}  "
            f"{row['duration_ms']} ms{release}"
        )
    return 0 if current == head else 2


def _import(args: argparse.Namespace, log: logging.Logger) -> int:
    data_dir: Path = args.from_files
    if not data_dir.is_dir():
        log.error("not a directory: %s", data_dir)
        return 1
    importers = registry()
    if args.only is not None and args.only not in importers:
        log.error("unknown store %r; importers: %s", args.only, ", ".join(sorted(importers)))
        return 1
    selected = [importers[args.only]] if args.only else list(importers.values())

    engine = create_engine(
        DatabaseSettings.from_env().rw_url,
        execution_options={"schema_translate_map": {None: args.plane}},
    )
    failed = False
    try:
        for imp in selected:
            rep = imp.run(data_dir, engine, dry_run=args.dry_run)
            verb = "would insert" if args.dry_run else "inserted"
            print(
                f"store {rep.store}: read {rep.read}, {verb} {rep.inserted}, "
                f"already present {rep.already_present}, mismatched {len(rep.mismatches)}"
            )
            for m in rep.mismatches:
                print(f"  MISMATCH {m.record}: {m.detail}")
            failed = failed or bool(rep.mismatches)
    finally:
        engine.dispose()

    inv = inventory.scan(data_dir)
    print("inventory:")
    for kind, label in (
        ("imported", "imported by a store"),
        ("file", "still a file (not yet migrated)"),
        ("ignored", "ignored (lock files)"),
        ("retired", "retired (a removed store's data, not imported)"),
        ("ops", "ops backups and receipts (not app state)"),
    ):
        print(f"  {label}: {inv.by_class.get(kind, 0)}")
        for (k, owner), n in sorted(inv.by_owner.items()):
            if k == kind:
                print(f"    {owner}: {n}")
    print(f"  UNCLAIMED: {inv.by_class.get('unclaimed', 0)}")
    for rel in inv.unclaimed:
        print(f"    UNCLAIMED {rel}")
    return 1 if failed or inv.unclaimed else 0


def run_db_cli(args: argparse.Namespace, log: logging.Logger) -> int:
    """Run one ``db`` subcommand; 0 on success, 1 on error, 2 for ``status`` when behind."""
    handlers = {"upgrade": _upgrade, "status": _status, "import": _import}
    try:
        return handlers[args.db_subcommand](args, log)
    except Exception as exc:  # noqa: BLE001 — a CLI reports, with the cause, and exits 1
        log.error("db %s failed: %s", args.db_subcommand, exc)
        print(f"db {args.db_subcommand} failed: {exc}", file=sys.stderr)
        return 1
