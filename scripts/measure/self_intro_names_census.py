"""Census of one-word self-introduced names on a corpus copy: every placed roster entry with
``source: self_intro`` and a one-token name, over the served episodes, with counts and an example
episode. Read-only. Used for 0033 (``docs/wip/changes/0033_hand_removed_names_refused.md``).

Usage: ``python scripts/measure/self_intro_names_census.py <corpus root>``
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from podcast_scraper.upgrade.corpus_selection import select_served_artifacts


def main(root: Path) -> None:
    """Print the served count, then one line per name: count, name, an example episode."""
    metas = select_served_artifacts(root, ".metadata.json")[0]
    counts: collections.Counter = collections.Counter()
    example = {}
    for meta in metas:
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in (data.get("content") or {}).get("speakers") or []:
            name = str(s.get("name") or "")
            if s.get("source") == "self_intro" and s.get("placed") and len(name.split()) == 1:
                counts[name] += 1
                example.setdefault(name, meta.name)
    print(
        f"served {len(metas)}; one-token placed self_intro names {sum(counts.values())}; "
        f"distinct {len(counts)}"
    )
    for name, n in counts.most_common():
        print(f"{n}\t{name}\t{example[name]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
