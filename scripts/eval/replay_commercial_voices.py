#!/usr/bin/env python3
"""Which voices does ``classify_voices`` type commercial, episode by episode? (replay, 2026-10-15)

Runs the diarization cleaning's real code over a directory of segment files (the raw canonical
``<ep>.segments.json`` of a corpus, which carry each segment's voice id) and writes, per episode,
the commercial voices and each one's share of the episode's talk. Run it under the old code and
the new, and compare the two outputs (``--compare``).

Inputs it can rebuild from segments: the diarization, the ad regions (``ad_time_intervals``), the
per-voice texts and the ordered turns. It cannot rebuild the corpus-wide evidence the pipeline
also passes (``recurring_text``, ``ad_signatures``); those only add voices to the AD set, which is
skipped BEFORE the commercial rule — so leaving them out can only widen the set the rule sees,
never hide a change.

    PYTHONPATH=<checkout>/src python scripts/eval/replay_commercial_voices.py DIR --out old.json
    python scripts/eval/replay_commercial_voices.py --compare old.json new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def run(root: Path) -> dict[str, Any]:
    from podcast_scraper.gi.ad_regions import ad_time_intervals
    from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
    from podcast_scraper.providers.ml.diarization.roster import classify_voices

    out: dict[str, Any] = {}
    for path in sorted(root.rglob("*.segments.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
        segs = [s for s in segs if isinstance(s, dict) and s.get("speaker")]
        if not segs:
            continue
        turns = [
            DiarizationSegment(
                start=float(s.get("start") or 0.0),
                end=float(s.get("end") or 0.0),
                speaker=s["speaker"],
            )
            for s in segs
        ]
        dia = DiarizationResult(
            segments=turns, num_speakers=len({t.speaker for t in turns}), model_name="replay"
        )
        voice_texts: dict[str, str] = {}
        for s in segs:
            voice_texts[s["speaker"]] = (
                voice_texts.get(s["speaker"], "") + " " + str(s.get("text") or "")
            )
        ordered = [(s["speaker"], str(s.get("text") or "")) for s in segs]
        cleaning = classify_voices(
            dia, ad_time_intervals(segs), voice_texts=voice_texts, ordered_turns=ordered
        )
        talk: dict[str, float] = {}
        for t in turns:
            talk[t.speaker] = talk.get(t.speaker, 0.0) + max(0.0, t.end - t.start)
        total = sum(talk.values()) or 1.0
        out[str(path.relative_to(root))] = {
            v: round(talk.get(v, 0.0) / total, 4) for v in sorted(cleaning.commercial)
        }
    return out


def compare(old_path: Path, new_path: Path) -> int:
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    if set(old) != set(new):
        print(f"FAIL: episode sets differ ({len(old)} vs {len(new)})")
        return 1
    changed = {ep: (old[ep], new[ep]) for ep in old if old[ep] != new[ep]}
    n_old = sum(len(v) for v in old.values())
    n_new = sum(len(v) for v in new.values())
    print(f"episodes {len(old)} · commercial voices old {n_old}, new {n_new}")
    print(f"episodes whose commercial voices changed: {len(changed)}")
    for ep, (o, n) in sorted(changed.items())[:30]:
        print(f"  {ep}: {o} -> {n}")
    # The invariant: the fix only ever REMOVES a voice holding >= 25% of the talk; nothing else
    # moves.
    bad = [
        ep
        for ep, (o, n) in changed.items()
        if set(n) - set(o) or any(o[v] < 0.25 for v in set(o) - set(n))
    ]
    if bad:
        print(f"FAIL: changes outside the fix's reach in {len(bad)} episode(s): {bad[:5]}")
        return 1
    print(
        "PASS: every change is a voice with >= 25% of its episode's talk no longer typed commercial"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", nargs="?", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.root or not args.out:
        ap.error("root and --out, or --compare OLD NEW")
    result = run(args.root)
    if not result:
        print("FAIL: no episodes replayed (empty corpus)")
        return 1
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(f"replayed {len(result)} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
