#!/usr/bin/env python3
"""Reproject an existing corpus's GI quotes onto the #974 ad-free processing base.

For corpora whose ``gi.json`` was built before #974 (quote ``char_start`` in the unsaved
ad-excised space), this backfills the ad-free sidecars and rewrites each quote's
``char_start`` / ``char_end`` / ``speaker_id`` / ``timestamp_*`` / ``transcript_ref`` to
the ad-free coordinate space — WITHOUT re-running the (Gemini) GI extraction. The quote
TEXT is unchanged; we re-locate its verbatim span in the ad-free text and recompute
offsets/attribution exactly as ``build_artifact`` would. This is a deterministic
migration usable to validate the fix / serve a corrected corpus before a faithful
re-run.

Note: this is a data-layer fixup, not a full pipeline run — it does NOT update
``corpus_manifest.json``'s ``produced_by`` stamp, so the server may still warn "no
produced_by stamp" when serving the reprojected corpus. A subsequent ``make reindex``
or a faithful re-diarization stamps it. Usage::

    python scripts/migrate/reproject_gi_to_adfree.py <corpus_root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podcast_scraper.gi.pipeline import _char_range_to_ms, _speaker_id_for_char_range
from podcast_scraper.gi.speakers import person_id
from podcast_scraper.workflow.adfree_transcript import (
    adfree_transcript_relpath,
    build_adfree_artifacts,
    save_adfree_artifacts,
)


def _reproject_episode(gi_path: Path) -> dict:
    run_dir = gi_path.parent.parent  # <run>/metadata/<ep>.gi.json
    stem = gi_path.name[: -len(".gi.json")]
    rel = f"transcripts/{stem}.txt"
    txt = run_dir / rel
    seg = run_dir / "transcripts" / f"{stem}.segments.json"
    if not (txt.is_file() and seg.is_file()):
        return {"skipped": True}

    raw = txt.read_text(encoding="utf-8")
    segs = json.loads(seg.read_text(encoding="utf-8"))
    arts = build_adfree_artifacts(raw, segs)
    if arts is None:
        return {"skipped": True}
    save_adfree_artifacts(rel, str(run_dir), arts)
    adfree_ref = adfree_transcript_relpath(rel)

    gi = json.loads(gi_path.read_text(encoding="utf-8"))
    quotes = [n for n in gi.get("nodes", []) if n.get("type") == "Quote"]
    found = attributed = 0
    for q in quotes:
        p = q.setdefault("properties", {})
        text = (p.get("text") or "").strip()
        if not text:
            continue
        cs = arts.text.find(text)
        if cs < 0:
            continue  # not locatable (rare) — leave as-is
        ce = cs + len(text)
        found += 1
        p["char_start"] = cs
        p["char_end"] = ce
        label = _speaker_id_for_char_range(arts.text, cs, ce, arts.segments)
        p["speaker_id"] = person_id(label) if label else None
        if label:
            attributed += 1
        ts0, ts1 = _char_range_to_ms(arts.text, cs, ce, arts.segments)
        p["timestamp_start_ms"] = ts0
        p["timestamp_end_ms"] = ts1
        p["transcript_ref"] = adfree_ref
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    return {
        "skipped": False,
        "quotes": len(quotes),
        "found": found,
        "attributed": attributed,
        "ad_chars": arts.chars_removed,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    gi_files = sorted(root.glob("feeds/**/*.gi.json"))
    tot = dict(eps=0, quotes=0, found=0, attributed=0)
    for gi_path in gi_files:
        r = _reproject_episode(gi_path)
        if r.get("skipped"):
            continue
        tot["eps"] += 1
        tot["quotes"] += r["quotes"]
        tot["found"] += r["found"]
        tot["attributed"] += r["attributed"]
        print(
            f"  {gi_path.name[:42]:<42} quotes={r['quotes']:>3} "
            f"found={r['found']:>3} attributed={r['attributed']:>3} ad_chars={r['ad_chars']}"
        )
    q = max(tot["quotes"], 1)
    print(
        f"\nreprojected eps={tot['eps']} quotes={tot['quotes']} "
        f"found={tot['found']} ({100 * tot['found'] // q}%) "
        f"attributed={tot['attributed']} ({100 * tot['attributed'] // q}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
