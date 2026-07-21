"""CLI for the corpus upgrade framework (#862): ``podcast upgrade ...``.

Subcommands give both on-demand and automated operation:

- ``status``  — current vs target version + pending migrations. Exit **2** when
  upgrades are pending (distinct from error=1), so boot scripts / CI can gate on it.
- ``list``    — every registered migration with applied|pending state.
- ``run``     — apply pending migrations. Interactive confirmation by default;
  ``--yes`` for automation, ``--dry-run`` to preview the plan, ``--to`` to stop at a
  version. Exit 1 if any step fails.
- ``verify``  — run each applied migration's verification. Exit 1 on any failure.

``--json`` on any subcommand emits a machine-readable object for pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from .migration import MigrationContext
from .runner import UpgradeRunner, UpgradeStatus
from .state import FilesystemStateStore


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--corpus-dir",
        default=os.getenv("CORPUS_DIR") or os.getenv("OUTPUT_DIR"),
        help="Corpus root (parent of feeds/). Defaults to $CORPUS_DIR / $OUTPUT_DIR.",
    )
    sub.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def parse_upgrade_argv(argv: Sequence[str]) -> argparse.Namespace:
    """Parse ``upgrade`` subcommand args into a namespace with ``command='upgrade'``."""
    parser = argparse.ArgumentParser(prog="podcast_scraper upgrade")
    subparsers = parser.add_subparsers(dest="upgrade_subcommand", required=True, help="Command")

    _add_common(subparsers.add_parser("status", help="Show version + pending migrations"))
    _add_common(subparsers.add_parser("list", help="List all migrations + their state"))
    _add_common(subparsers.add_parser("verify", help="Verify applied migrations"))

    run_p = subparsers.add_parser("run", help="Apply pending migrations")
    _add_common(run_p)
    run_p.add_argument("--to", dest="to_version", default=None, help="Stop at this version.")
    run_p.add_argument("--dry-run", action="store_true", help="Preview the plan; write nothing.")
    run_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation (automation).")
    run_p.add_argument(
        "--snapshot-dir",
        default=None,
        help="Where to write the pre-upgrade corpus snapshot (default: a sibling of the "
        "corpus root). In a container, point this at a mounted/persistent path.",
    )
    run_p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip the pre-upgrade local snapshot. The migrations are in-place; without a "
        "snapshot, a partial failure can only be rolled back from an external backup.",
    )

    args = parser.parse_args(list(argv))
    args.command = "upgrade"
    return args


def _resolve_corpus_root(args: argparse.Namespace, log: logging.Logger) -> Optional[Path]:
    if not args.corpus_dir:
        log.error("--corpus-dir is required (or set $CORPUS_DIR / $OUTPUT_DIR).")
        return None
    return Path(args.corpus_dir).expanduser().resolve()


def _runner_for(corpus_root: Path) -> UpgradeRunner:
    return UpgradeRunner(FilesystemStateStore(corpus_root))


def _status_payload(status: UpgradeStatus) -> dict:
    return {
        "current_version": status.current_version,
        "target_version": status.target_version,
        "applied": status.applied,
        "pending": [m.id for m in status.pending],
        "up_to_date": status.up_to_date,
    }


def _cmd_status(runner: UpgradeRunner, as_json: bool, log: logging.Logger) -> int:
    status = runner.status()
    if as_json:
        print(json.dumps(_status_payload(status), indent=2))
    else:
        print(f"Current version: {status.current_version or 'unstamped'}")
        print(f"Target version:  {status.target_version or 'n/a'}")
        print(f"Applied:         {', '.join(status.applied) or 'none'}")
        if status.pending:
            print(f"Pending ({len(status.pending)}):")
            for m in status.pending:
                print(f"  - {m.id} → {m.to_version}: {m.description}")
        else:
            print("Up to date — no pending migrations.")
    return 0 if status.up_to_date else 2  # 2 = action needed (distinct from error)


def _cmd_list(runner: UpgradeRunner, as_json: bool) -> int:
    applied = set(runner.state.applied_migration_ids())
    rows = [
        {
            "id": m.id,
            "to_version": m.to_version,
            "description": m.description,
            "state": "applied" if m.id in applied else "pending",
        }
        for m in runner.migrations
    ]
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"  [{r['state']:7s}] {r['id']} → {r['to_version']}: {r['description']}")
    return 0


def _cmd_verify(runner: UpgradeRunner, ctx: MigrationContext, as_json: bool) -> int:
    results = runner.verify(ctx)
    ok_all = all(ok for _, ok, _ in results)
    if as_json:
        print(json.dumps([{"id": i, "ok": ok, "message": m} for i, ok, m in results], indent=2))
    else:
        if not results:
            print("No applied migrations to verify.")
        for mid, ok, msg in results:
            print(f"  [{'OK ' if ok else 'FAIL'}] {mid}: {msg}")
    return 0 if ok_all else 1


def _confirm(status: UpgradeStatus, log: logging.Logger) -> bool:
    if not sys.stdin.isatty():
        log.error("Refusing to apply without confirmation; pass --yes for automation.")
        return False
    names = ", ".join(m.id for m in status.pending)
    reply = input(f"Apply {len(status.pending)} migration(s) [{names}]? [y/N] ").strip().lower()
    return reply in ("y", "yes")


def _snapshot_corpus(
    corpus_root: Path,
    from_version: str,
    snapshot_dir: Optional[str],
    log: logging.Logger,
) -> Optional[Path]:
    """Copy the corpus to a pre-upgrade snapshot before the in-place migrations run.

    Placed OUTSIDE ``corpus_root`` (default: a sibling dir) so the migrations — which
    walk the corpus tree — never traverse or mutate the backup copy. In a container,
    pass ``--snapshot-dir`` pointing at a mounted/persistent path (a sibling of a
    volume mount is container-ephemeral). Returns the snapshot path, or ``None`` on
    failure so the caller aborts before mutating anything.
    """
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        frm = str(from_version or "unstamped").replace("/", "_")
        base = Path(snapshot_dir).expanduser() if snapshot_dir else corpus_root.parent
        dest = base / f".corpus-upgrade-backup-{frm}-{ts}"
        corpus_res = corpus_root.resolve()
        dest_res = dest.resolve()
        # The snapshot must not sit inside the corpus root — else the migrations walk it.
        if dest_res == corpus_res or corpus_res in dest_res.parents:
            log.error(
                "snapshot-dir must be OUTSIDE the corpus root (%s); refusing %s",
                corpus_root,
                dest,
            )
            return None
        base.mkdir(parents=True, exist_ok=True)
        shutil.copytree(corpus_root, dest, symlinks=True)
        return dest
    except Exception:  # noqa: BLE001 — a snapshot failure must abort, never mutate
        log.exception("pre-upgrade snapshot failed")
        return None


def _cmd_run(
    runner: UpgradeRunner, ctx: MigrationContext, args: argparse.Namespace, log: logging.Logger
) -> int:
    status = runner.status()
    if not status.pending:
        print("Up to date — nothing to run.")
        return 0
    if not args.dry_run and not args.yes and not _confirm(status, log):
        print("Aborted.")
        return 1

    # Pre-upgrade local snapshot (defense-in-depth). The migrations are IN-PLACE and
    # the runner STOPS at the first failure, so a bad migration can leave a partially
    # migrated corpus. Snapshot the corpus first (outside the corpus root, so the
    # migrations never walk it) for an instant local rollback. Skippable via
    # --no-snapshot (fresh external backup + tight disk).
    if not args.dry_run and not getattr(args, "no_snapshot", False):
        snap = _snapshot_corpus(
            ctx.corpus_root,
            runner.state.current_version() or "unstamped",
            getattr(args, "snapshot_dir", None),
            log,
        )
        if snap is None:
            log.error(
                "Pre-upgrade snapshot failed — aborting before any migration runs. "
                "Fix --snapshot-dir (must be writable + outside the corpus), or pass "
                "--no-snapshot if you have a fresh external backup."
            )
            return 1
        # To stderr, so ``--json`` stdout stays clean/parseable.
        print(f"Pre-upgrade snapshot: {snap}", file=sys.stderr)
        print(
            "  rollback = replace the corpus with this dir if the upgrade fails.",
            file=sys.stderr,
        )

    now = datetime.now(timezone.utc).isoformat()
    results = runner.run(ctx, to_version=args.to_version, now=now)
    payload: List[dict] = [
        {
            "id": r.migration_id,
            "applied": r.applied,
            "dry_run": r.dry_run,
            "message": r.message,
            "details": r.details,
        }
        for r in results
    ]
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            tag = "DRY-RUN" if r.dry_run else ("applied" if r.applied else "skipped")
            print(f"  [{tag}] {r.migration_id}: {r.message}")
        remaining = runner.pending()
        if not args.dry_run:
            print(
                f"\nNow at version {runner.state.current_version() or 'unstamped'}; "
                f"{len(remaining)} pending."
            )
    return 0


def run_upgrade_cli(args: argparse.Namespace, log: logging.Logger) -> int:
    """Dispatch a parsed ``upgrade`` namespace to its subcommand handler."""
    corpus_root = _resolve_corpus_root(args, log)
    if corpus_root is None:
        return 1
    runner = _runner_for(corpus_root)
    ctx = MigrationContext(
        corpus_root=corpus_root,
        dry_run=getattr(args, "dry_run", False),
        logger=log,
    )

    sub = args.upgrade_subcommand
    if sub == "status":
        return _cmd_status(runner, args.json, log)
    if sub == "list":
        return _cmd_list(runner, args.json)
    if sub == "verify":
        return _cmd_verify(runner, ctx, args.json)
    if sub == "run":
        return _cmd_run(runner, ctx, args, log)
    log.error("Unknown upgrade subcommand: %s", sub)
    return 1
