#!/usr/bin/env python3
"""The naming shadow report over a corpus: side namers, talk floor, rescues (read only).

See :mod:`podcast_scraper.quality.naming_shadow` for what each table means. Runs on prod with the
checkout's source mounted, like the other audits; it reads sidecars and writes nothing but
``--json`` when given.

    python scripts/measure/naming_shadow_report.py --corpus /app/output [--json out.json] \\
        [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from podcast_scraper.quality.naming_shadow import render, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the report; ``--json`` also writes it whole."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--limit", type=int, help="rows listed per table (all when omitted)")
    a = ap.parse_args(argv)
    rep = report(a.corpus)
    sys.stdout.write(render(rep, a.limit))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
