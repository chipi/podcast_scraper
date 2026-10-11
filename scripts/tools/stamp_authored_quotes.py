#!/usr/bin/env python3
"""Give the app validation corpus's authored quotes the time they are said (dry run by default).

``build_app_validation_corpus._inject_authored_claims`` stamped every authored quote at 0 ms with
character offsets 0..len, so the player could not play them and the Moments picker could not clip
them. The builder now anchors new ones (``authored_quote_anchor``); this applies the same anchor to
the COMMITTED corpus in place, because a rebuild does not reproduce it (later edits live only in the
committed files). Only quotes still at 0 ms are touched; one that no single transcript segment
contains is reported and left alone — a paraphrase must not be given a time.

    .venv/bin/python scripts/tools/stamp_authored_quotes.py            # report
    .venv/bin/python scripts/tools/stamp_authored_quotes.py --apply    # write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_app_validation_corpus import authored_quote_anchor  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("tests/fixtures/app-validation-corpus/v3"))
    p.add_argument("--apply", action="store_true", help="write the changes (default: report only)")
    args = p.parse_args()
    stamped = skipped = 0
    for gi_path in sorted((args.corpus / "feeds").rglob("*.gi.json")):
        gi = json.loads(gi_path.read_text(encoding="utf-8"))
        authored = [
            n
            for n in gi.get("nodes", [])
            if n.get("type") == "Quote"
            and str(n.get("id", "")).startswith("quote:authored-")
            and not (n.get("properties") or {}).get("timestamp_start_ms")
        ]
        if not authored:
            continue
        run = gi_path.parent.parent
        stem = gi_path.name[: -len(".gi.json")]
        segments = json.loads((run / "transcripts" / f"{stem}.segments.json").read_text("utf-8"))
        transcript = (run / "transcripts" / f"{stem}.txt").read_text(encoding="utf-8")
        changed = False
        for node in authored:
            props = node["properties"]
            anchor = authored_quote_anchor(str(props.get("text") or ""), segments, transcript)
            label = f"{gi_path.relative_to(args.corpus)} {node['id']}"
            if anchor is None:
                skipped += 1
                print(f"SKIP  {label}: not in one transcript segment — left at 0 ms")
                continue
            stamped += 1
            changed = True
            print(f"STAMP {label}: {anchor}")
            props.update(anchor)
        if changed and args.apply:
            gi_path.write_text(json.dumps(gi, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{'stamped' if args.apply else 'would stamp'} {stamped}, left {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
