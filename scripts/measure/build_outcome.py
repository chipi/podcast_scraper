#!/usr/bin/env python3
"""Write a migration's shipped outcome (``upgrade/outcomes/<id>.json``) from its gate report and
its change record, for the migration outcome hook (``upgrade/outcome_hook.py``).

    python scripts/measure/build_outcome.py --migration 0031_named_voices_shown
    python scripts/measure/build_outcome.py --migration <id> --root <an extension's checkout>

The record and gate report are read from ``--root``'s records dir (this checkout by default). The
outcome is written where the hook reads it, ``outcome_hook.outcome_path``: beside the package that
holds the migration, so a migration an installed extension adds gets its outcome in that package
(ADR-174); the extension must be installed (or on the path) for its migration to be found.

Refuses unless the gate report was measured on clean code and the record's last ``Verdict:`` line
in step 8 is GO / GO WITH CONDITIONS. Prints the outcome's sha256: the record's verdict line must
then name it (``outcome <sha256>``), which the CI landing test checks, so an outcome edited after
its verdict no longer matches. Copies, never computes: every number comes from the gate report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.quality.change_gate import load_config, parse_record  # noqa: E402
from podcast_scraper.upgrade.outcome_hook import outcome_path  # noqa: E402

VERDICT = re.compile(r"^Verdict:\s*(GO WITH CONDITIONS|GO|NO-GO)\b", re.MULTILINE)
INLINE = re.compile(r"\bConditions?:\s*(.+?)(?:\s+Open, unchanged:|$)")


def build(migration: str, root: Optional[Path] = None) -> dict:
    root = root or REPO
    cfg = load_config(root)
    rec_dir = root / cfg["records_dir"]
    gate = json.loads((rec_dir / f"{migration}.gate.json").read_text(encoding="utf-8"))
    rec = parse_record(rec_dir / f"{migration}.md")
    if rec is None:
        raise SystemExit(f"no change record for {migration}")
    step8 = rec.sections.get("8", "")
    verdicts = list(VERDICT.finditer(step8))
    if not verdicts or verdicts[-1].group(1) == "NO-GO":
        raise SystemExit(f"{migration}: the record's last verdict is not GO")
    if gate.get("code", {}).get("dirty"):
        raise SystemExit(f"{migration}: the gate report was measured on uncommitted code")
    tail = step8[verdicts[-1].end() :]
    conditions = [ln.strip() for ln in tail.splitlines() if ln.strip().startswith("- **C")]
    # A verdict written on one line carries its conditions inline ("Condition: ... . Open, ...").
    inline = INLINE.search(tail.splitlines()[0] if tail else "")
    if inline:
        conditions.insert(0, inline.group(1).strip())
    if verdicts[-1].group(1) == "GO WITH CONDITIONS" and not conditions:
        raise SystemExit(f"{migration}: GO WITH CONDITIONS, but no condition was found to carry")
    return {
        "migration": migration,
        "snapshot": gate["snapshot"],
        "applied_before": gate["applied_before"],
        "code_digest": gate["code_digest"],
        "code_dirty": False,
        "code_sha": gate["code"]["sha"],
        "verdicts_sha256": gate.get("verdicts_sha256"),
        "scoreboard_before": gate["scoreboard_before"],
        "scoreboard_after": gate["scoreboard_after"],
        "episode_deltas": gate["episode_deltas"],
        "plan": gate["plan"],
        "plan_sha256": gate["plan_sha256"],
        "computed_class": gate["computed_class"],
        "verdict": verdicts[-1].group(1),
        "conditions": conditions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--migration", required=True)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="the checkout holding the record and gate report (default: this one)",
    )
    args = ap.parse_args()
    raw = (json.dumps(build(args.migration, args.root), indent=1, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    out = outcome_path(args.migration)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"wrote {out}; outcome sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
