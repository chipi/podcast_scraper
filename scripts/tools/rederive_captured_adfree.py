#!/usr/bin/env python3
"""Re-derive the ad-free stack of the captured English renders with today's ad cut.

``tests/fixtures/pipeline-renders/<version>/<ep>.json`` holds what a real translation run wrote for
the app fixture corpus (``scripts/build_app_validation_corpus.py`` replays it). Its ``adfree.*``
files are frozen at the ad cut of that run, so an ad-detection fix (#2346: "Start at
strava.com/podcast") never reaches them without a new translation run. This rebuilds them from the
capture's own English ``segments.json`` with ``workflow.adfree_transcript.build_adfree_artifacts``,
the function the pipeline uses, keeping the capture's ``SPEAKER_xx`` labels (the run cut ads before
naming), and serialises them exactly as ``save_adfree_artifacts`` does.

The replay is checked before anything is written: every capture the new cut does not change must
come out byte-identical to what is committed. Then rebuild the corpus with the builder.

    .venv/bin/python scripts/tools/rederive_captured_adfree.py [--version v3] [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RENDER_DIR = Path("tests/fixtures/pipeline-renders")
KEYS = ("adfree.txt", "adfree.segments.json", "adfree.admap.json")


def _relabel(raw: List[Dict[str, Any]], old_adfree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The named segments with the run's pre-naming labels back: name -> SPEAKER_xx, learnt from
    the sentences both files hold."""
    by_sent = {s.get("sent_id"): s.get("speaker_label") for s in old_adfree if s.get("sent_id")}
    name_to_label: Dict[str, str] = {}
    for s in raw:
        label = by_sent.get(s.get("sent_id"))
        if label and s.get("speaker_label"):
            name_to_label.setdefault(str(s["speaker_label"]), str(label))
    out = []
    for s in raw:
        t = dict(s)
        t["speaker_label"] = name_to_label.get(str(s.get("speaker_label")), s.get("speaker_label"))
        out.append(t)
    return out


def rederive(stack: Dict[str, str]) -> Optional[Dict[str, str]]:
    """The three ad-free files for one capture, or ``None`` when it has none to rebuild."""
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_with_offsets,
    )
    from podcast_scraper.workflow.adfree_transcript import build_adfree_artifacts

    if not all(k in stack for k in KEYS + ("segments.json",)):
        return None
    raw = json.loads(stack["segments.json"])
    segments = _relabel(raw, json.loads(stack["adfree.segments.json"]))
    text, _ = format_diarized_screenplay_with_offsets(segments)
    art = build_adfree_artifacts(text, segments)
    if art is None:
        return None
    # The formatter keeps timing, speaker and text; the run's segments also carried their
    # translation ids, matched back by time and text.
    ids = {(s["start"], s["end"], s["text"]): s for s in raw}
    adfree = []
    for seg in art.segments:
        src = ids.get((seg.get("start"), seg.get("end"), seg.get("text")), {})
        extra = {k: src[k] for k in ("unit_id", "sent_id") if k in src}
        adfree.append({**seg, **extra})
    return {
        "adfree.txt": art.text,
        "adfree.segments.json": json.dumps(adfree, indent=0, allow_nan=False),
        "adfree.admap.json": json.dumps(art.ad_map, indent=2),
    }


def main(argv: Optional[list] = None) -> int:
    """Check every capture; with ``--write``, update the changed ones."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", default="v3")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    changed: List[Tuple[Path, Dict[str, Any], Dict[str, str]]] = []
    for path in sorted((RENDER_DIR / args.version).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        stack = ((doc.get("render") or {}).get("stack")) or {}
        new = rederive(stack)
        if new is None:
            print(f"skip       {path.name} (no ad-free stack)")
            continue
        diff = [k for k in KEYS if new[k] != stack[k]]
        if not diff:
            print(f"identical  {path.name}")
            continue
        removed = len(stack["adfree.txt"]) - len(new["adfree.txt"])
        print(f"changed    {path.name}: {', '.join(diff)} ({removed:+d} chars of ad-free text)")
        changed.append((path, doc, new))
    if not args.write:
        print(f"{len(changed)} capture(s) would change; nothing written (add --write)")
        return 0
    for path, doc, new in changed:
        doc["render"]["stack"].update(new)
        path.write_text(
            json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(changed)} capture(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
