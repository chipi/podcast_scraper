#!/usr/bin/env python3
"""Put the app-validation corpus's English renders on their source timeline (operator 2026-10-12).

The translated shows (p10-p14) carry English renders captured from a pipeline run on other audio,
so their timings disagree with the episode's own source segments and GI quotes. A short listen cut
its clip from the render at the quote's time and played the wrong line. This applies
``build_app_validation_corpus.retime_english_render`` (which the builder now runs after replaying
a render) to the committed corpus. Text is untouched; only ``start`` / ``end`` move.

Usage::

    .venv/bin/python scripts/tools/retime_fixture_english_renders.py          # dry run
    .venv/bin/python scripts/tools/retime_fixture_english_renders.py --apply  # rewrite

Idempotent: a second ``--apply`` moves nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import build_app_validation_corpus as builder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=REPO / "tests/fixtures/app-validation-corpus/v3")
    p.add_argument("--apply", action="store_true", help="rewrite the files (default: report)")
    args = p.parse_args()
    total = 0
    for ledger in sorted(args.corpus.glob("feeds/*/run_*/transcripts/*.translation.json")):
        ep_label = ledger.name[: -len(".translation.json")]
        moved = builder.retime_english_render(ledger.parent, ep_label, apply=args.apply)
        if not builder.english_render_times(ledger.parent, ep_label):
            print(f"  {ep_label}: cannot align, left as is")
            continue
        print(f"  {ep_label}: {moved} segments {'moved' if args.apply else 'would move'}")
        total += moved
    print(f"{'moved' if args.apply else 'would move'} {total} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
