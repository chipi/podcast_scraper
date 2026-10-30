"""``podcast_scraper identity-ledger``: build and write the identity ledger (ADR-166).

The deterministic step that runs last on every corpus-writing path (ingest, migration apply,
relabel repair, enrich-edges), and once with the new image before an api deploy that reads the
ledger. Corpus text in, one derived file out; nothing else is written.
"""

from __future__ import annotations

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Sequence

from podcast_scraper.identity import ledger as L


def parse_identity_ledger_argv(argv: Sequence[str]) -> Namespace:
    """Parse ``identity-ledger`` args."""
    parser = argparse.ArgumentParser(prog="podcast_scraper identity-ledger")
    parser.add_argument("--output-dir", required=True, help="Corpus root (parent of feeds/).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report what would change; write nothing.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report the current ledger's state and staleness; build nothing.",
    )
    parser.add_argument(
        "--restore-previous",
        action="store_true",
        help="Put the previous valid ledger back (the undo of the last write); build nothing.",
    )
    args = parser.parse_args(list(argv))
    args.command = "identity-ledger"
    return args


def run_identity_ledger_cli(args: Namespace, log: logging.Logger) -> int:
    """Run ``identity-ledger``; prints one JSON object. Exit 0 on success, 2 on a bad root."""
    root = Path(args.output_dir)
    if not root.is_dir():
        log.error("identity-ledger: corpus root not found: %s", root)
        return 2
    if args.status:
        print(json.dumps(L.ledger_status(root), ensure_ascii=False, sort_keys=True))
        return 0
    if args.restore_previous:
        try:
            change = L.restore_previous(root)
        except (OSError, ValueError) as exc:
            log.error("identity-ledger: nothing restored: %s", exc)
            return 1
        print(json.dumps(change, ensure_ascii=False, sort_keys=True))
        return 0
    ledger = L.build_ledger(root)
    if args.dry_run:
        previous, state = L.read_ledger(root)
        change = L._change_entry(previous, ledger)
        change["dry_run"] = True
        change["previous_state"] = state
        print(json.dumps(change, ensure_ascii=False, sort_keys=True))
        return 0
    change = L.write_ledger(root, ledger)
    log.info(
        "identity-ledger: %d joins (%d added, %d removed, %d canonical changed) -> %s",
        change["joins"],
        change["joins_added"],
        change["joins_removed"],
        change["canonical_changed"],
        root / L.LEDGER_NAME,
    )
    print(json.dumps(change, ensure_ascii=False, sort_keys=True))
    return 0
