#!/usr/bin/env python3
"""Read-only: a hash of every file a migration's apply would write, built in memory as the apply
builds it (the same scan, the same JSON serialisation). Writes nothing.

    python scripts/measure/write_digest_ro.py --migration 0043_non_participant_names_removed \\
        --corpus /app/output

Run it on the gate's clone and on prod: equal ``writes_sha256`` means the apply would write the same
bytes on both, whatever an input hash in the plan says (2026-11-05: an identity ledger refreshed by
a nightly made every row of 0043's plan differ, and this proved the writes unchanged). For the
reviewed-list migrations (0029-shaped: ``_scan`` and ``json_files`` / ``text_files``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.upgrade.file_rewrite import dump_json  # noqa: E402
from podcast_scraper.upgrade.registry import get_migrations  # noqa: E402


def write_digest(mid: str, corpus: Path) -> Dict[str, Any]:
    """``{episodes, refused, files, writes_sha256}`` of what ``mid``'s apply would write."""
    mig: Any = next((m for m in get_migrations() if m.id == mid), None)
    if mig is None:
        raise SystemExit(f"not registered: {mid}")
    if not hasattr(mig, "_scan"):
        raise SystemExit(
            f"{mid}: not a 0029-shaped migration (no _scan); its writes cannot be built"
        )
    if hasattr(mig, "_reviewed"):
        mig._apply, _missing = (
            mig._reviewed()
        )  # as the reviewed-list migrations' apply() does first
    write, refused, _totals = mig._scan(corpus)
    rows: List[Tuple[str, str]] = []
    for ep in write:
        for path, payload in ep.json_files.items():
            data = dump_json(payload).encode()
            rows.append(
                (Path(path).relative_to(corpus).as_posix(), hashlib.sha256(data).hexdigest())
            )
        for path, text in ep.text_files.items():
            data = text.encode()
            rows.append(
                (Path(path).relative_to(corpus).as_posix(), hashlib.sha256(data).hexdigest())
            )
    rows.sort()
    joined = "\n".join(p + "\t" + h for p, h in rows).encode()
    return {
        "migration": mid,
        "episodes": len(write),
        "refused": len(refused),
        "files": len(rows),
        "writes_sha256": hashlib.sha256(joined).hexdigest(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the digest as one JSON line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--migration", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    args = ap.parse_args(argv)
    print(json.dumps(write_digest(args.migration, args.corpus)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
