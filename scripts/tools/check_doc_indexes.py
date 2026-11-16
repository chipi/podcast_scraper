#!/usr/bin/env python3
"""Every doc in a per-type docs directory is linked from that directory's index.md.

llms.txt sends agents to these indexes instead of listing hundreds of ADRs, RFCs and PRDs, so an
index that misses a doc hides it: on 2026-11-16 the ADR, RFC and PRD indexes each missed 10 docs,
the guides index 21. The indexes stay hand-written (status, relations, descriptions); this only
checks they are complete. Exit 1 lists each missing doc.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = ("docs/adr", "docs/rfc", "docs/prd", "docs/guides", "docs/architecture", "docs/api")
LINK = re.compile(r"\]\(([^)#\s]+\.md)")


def missing(folder: Path) -> list[str]:
    linked = {Path(m).name for m in LINK.findall((folder / "index.md").read_text(encoding="utf-8"))}
    return sorted(
        p.name for p in folder.glob("*.md") if p.name != "index.md" and p.name not in linked
    )


def main() -> int:
    gaps = {d: missing(ROOT / d) for d in DIRS}
    for d, names in gaps.items():
        for name in names:
            print(f"{d}/index.md does not link {name}")
    if any(gaps.values()):
        print(f"FAIL: {sum(map(len, gaps.values()))} doc(s) missing from their index")
        return 1
    print(f"doc indexes complete: {', '.join(DIRS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
