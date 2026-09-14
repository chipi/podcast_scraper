"""Roll back the role changes m0009 made, from the ledger it wrote (#2069).

    python scripts/ops/undo_speaker_roles.py --corpus-dir <corpus> [--show]

``--show`` prints the ledger and changes nothing. Without it, every recorded ``role_before`` is
restored — except on nodes something else has written since, which are REFUSED and reported. A
rollback that overwrites a later re-enrich is a regression wearing a rollback's clothes.

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
    args = parser.parse_args(argv)
    root = args.corpus_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rows = read_ledger(root)
    if not rows:
        print(f"no role ledger at {root / LEDGER_FILE} — nothing to undo")
        return 0

    if args.show:
        print(f"{len(rows)} recorded change(s) in {root / LEDGER_FILE}")
        for row in rows:
            print(
                f"   {row.route:14} {row.name[:34]:36} {row.role_before:10} -> "
                f"{row.role_after:10} {row.feed_title[:30]}"
            )
        return 0

    restored, refused = undo_from_ledger(root)
    print(f"restored {restored} role(s); refused {len(refused)}")
    for line in refused:
        print(f"   REFUSED {line}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
