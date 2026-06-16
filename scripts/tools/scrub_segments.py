#!/usr/bin/env python3
"""Scrub network-name speaker_label from BOTH segment artifacts (raw + adfree).

Companion to scrub_network_speakers.py (which fixed gi.json + content.speakers +
adfree.segments). The raw <base>.segments.json was missed. Re-resolves the host the
same way (extract_self_introduced_host + feed-modal fallback) and renames any
speaker_label that is_known_network. Offset-safe (a label field, not the text).
Default DRY-RUN; --apply to write.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
)
from podcast_scraper.speaker_detectors.hosts import (
    extract_self_introduced_host,
    is_known_network,
)

CORPUS = Path(".test_outputs/manual/prod-v2/corpus")
APPLY = "--apply" in sys.argv


def _feed_dir(p: Path) -> str:
    for part in p.parts:
        if part.startswith("rss_"):
            return part
    return ""


def _seg_files(meta_path: Path, doc: dict) -> list[Path]:
    root = episode_root_from_metadata_path(meta_path)
    rel = ((doc.get("content") or {}).get("transcript_file_path") or "").strip()
    if not rel:
        return []
    base = (root / rel).with_suffix("")
    return [
        base.with_name(base.name + ".segments.json"),
        base.with_name(base.name + ".adfree.segments.json"),
    ]


def main() -> int:
    print(f"corpus: {CORPUS}  mode: {'APPLY' if APPLY else 'DRY-RUN'}\n")
    metas = list(discover_metadata_files(CORPUS))
    docs = {}
    per, votes = {}, {}
    for m in metas:
        try:
            doc = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        docs[m] = doc
        root = episode_root_from_metadata_path(m)
        rel = ((doc.get("content") or {}).get("transcript_file_path") or "").strip()
        txt = ""
        if rel:
            try:
                txt = (root / rel).read_text(encoding="utf-8")
            except OSError:
                txt = ""
        r = extract_self_introduced_host(txt)
        r = r if (r and not is_known_network(r)) else None
        per[m] = r
        if r:
            votes.setdefault(_feed_dir(m), Counter())[r] += 1
    feed_host = {f: c.most_common(1)[0][0] for f, c in votes.items() if c}

    total = 0
    for m, doc in docs.items():
        resolved = per[m] or feed_host.get(_feed_dir(m))
        for f in _seg_files(m, doc):
            if not f.is_file():
                continue
            try:
                segs = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            n = sum(1 for sg in segs if is_known_network(sg.get("speaker_label") or ""))
            if not n:
                continue
            if not resolved:
                print(f"SKIP (no host): {f.name}  ({n} network labels)")
                continue
            for sg in segs:
                if is_known_network(sg.get("speaker_label") or ""):
                    sg["speaker_label"] = resolved
            print(f"{f.name}: {n} labels -> {resolved!r}")
            total += 1
            if APPLY:
                f.write_text(json.dumps(segs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{'APPLIED' if APPLY else 'WOULD CHANGE'} {total} segment file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
