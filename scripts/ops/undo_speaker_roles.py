"""Roll back the role changes m0009 made, from the ledger it wrote (#2069).

    python scripts/ops/undo_speaker_roles.py --corpus-dir <corpus> [--show] [--run-id ID]

``--show`` prints the ledger and changes nothing. Without it, every recorded ``role_before`` is
restored — except on nodes something else has written since, which are REFUSED and reported. A
rollback that overwrites a later re-enrich is a regression wearing a rollback's clothes.

THE LEDGER CAN HOLD MORE THAN ONE RUN, and the default undoes only the NEWEST. That is the right
default and a trap worth naming: if m0009 crashed part-way and was re-run, the second run has its
own ``run_id``, and an undo then restores only what the second run touched — a PARTIAL rollback
that reports success. ``--show`` lists every run so the choice is visible, and ``--run-id`` makes
it. Undo the runs newest-first, one call each; there is no "undo everything" flag because the
order matters and a wrong order writes an older run's ``role_before`` over a newer one's.

Lives here rather than as ``python -m podcast_scraper.upgrade.role_ledger`` because running a
package submodule as a script double-imports it (RuntimeWarning), and the ops tools in this
directory are the established entry-point shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from podcast_scraper.upgrade.role_ledger import LEDGER_FILE, read_ledger, undo_from_ledger


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument(
        "--show", action="store_true", help="Print the ledger without changing anything."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Undo THIS migration run instead of the newest. Use --show to list them; a ledger "
            "with more than one run means m0009 ran more than once, and the default would roll "
            "back only the last."
        ),
    )
    args = parser.parse_args(argv)
    root = args.corpus_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    try:
        rows = read_ledger(root)
    except ValueError as exc:
        # A corrupt ledger is NOT the same fact as a missing one. The first version collapsed
        # them and printed "nothing to undo" over an unreadable file, exiting 0.
        print(f"ledger is unreadable: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"no role ledger at {root / LEDGER_FILE} — nothing to undo")
        return 0

    if args.show:
        print(f"{len(rows)} recorded change(s) in {root / LEDGER_FILE}")
        runs: dict[str, int] = {}
        for row in rows:
            runs[row.run_id] = runs.get(row.run_id, 0) + 1
        if len(runs) > 1:
            print(
                f"   !! {len(runs)} RUNS recorded — the default undoes only the newest. "
                "Pass --run-id to choose:"
            )
        for rid, n in runs.items():
            print(f"   run {rid or '<unset>'}: {n} change(s)")
        print()
        for row in rows:
            before = "<absent>" if row.role_before is None else row.role_before
            print(
                f"   {row.route:14} {row.name[:34]:36} {before:10} -> "
                f"{row.role_after:10} {row.feed_title[:30]}"
            )
        return 0

    try:
        restored, skipped, refused = undo_from_ledger(root, run_id=args.run_id)
    except RuntimeError as exc:
        # The corpus lock is held by a LIVE process — an ingest or a migration is running. The
        # undo is a read-modify-write across many artifacts, so running it now would race that
        # process and lose silently. Nothing has been changed; stop the other job and re-run.
        print(f"REFUSED: {exc}")
        return 1
    print(f"restored {restored} role(s); skipped {len(skipped)}; refused {len(refused)}")
    for line in skipped:
        print(f"   skipped {line}")
    for line in refused:
        print(f"   REFUSED {line}")
    # Refusals mean the rollback is PARTIAL. Exiting 0 on that would let a script — or an operator
    # reading only the exit code — treat a partial rollback as a complete one.
    return 1 if refused else 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
