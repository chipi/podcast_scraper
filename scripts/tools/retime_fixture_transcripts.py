#!/usr/bin/env python3
"""Put the app-validation corpus on its audio's real timeline (operator 2026-10-15).

Every fixture transcript line was stamped on a synthetic 6-second grid (``i * 6``), which the TTS
audio does not follow — a line takes as long as the voice takes to say it. A clip cut at a
segment's time played other words: ``scripts/eval/clip_alignment_check.py`` found 2 of 24 clips
playing what their card says, the offset growing to +41 s through an episode.

The real times are known exactly: ``tests/fixtures/scripts/transcripts_to_mp3.py`` renders each
turn with ``say``, joins the renders back to back, and writes their spans to the ``.rttm`` beside
the transcript. ``build_synthetic_validation_corpus.measured_line_times`` turns those into a
``(start, end)`` per line (a turn's span exact; split by text length inside a multi-line turn).

Per episode this tool builds the map from each segment's OLD span to its REAL one (linear inside
a segment, so a quote keeps its place within its sentence) and applies it to everything on that
timeline: the source segments (``<ep>.segments.json``, or ``<ep>.<lang>.segments.json`` for a
translated show), the ad-free segments and cut points, and the GI quotes' times. A translated
show's English render is then re-derived from the new source timeline
(``retime_english_render``). The search index is derived from these files; rebuild it afterwards
with ``make build-app-validation-index``.

    .venv/bin/python scripts/tools/retime_fixture_transcripts.py          # report
    .venv/bin/python scripts/tools/retime_fixture_transcripts.py --apply  # write

Idempotent: once applied, a second run moves nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import build_app_validation_corpus as builder  # noqa: E402
import build_synthetic_validation_corpus as synth  # noqa: E402

TRANSCRIPTS = REPO / "tests/fixtures/transcripts/v3"


def time_map(
    old: list[tuple[float, float]], new: list[tuple[float, float]]
) -> Callable[[float], float]:
    """Seconds on the old timeline -> seconds on the real one, piecewise linear by segment."""
    starts = [s for s, _ in old]

    def remap(t: float) -> float:
        i = max(0, bisect_right(starts, t) - 1)
        (os_, oe), (ns, ne) = old[i], new[i]
        if t >= oe and i + 1 < len(old):
            # Between segments (a gap on the old grid): pin to the next segment's start.
            return new[i + 1][0]
        span = oe - os_
        frac = (t - os_) / span if span > 0 else 0.0
        return round(ns + max(0.0, min(1.0, frac)) * (ne - ns), 3)

    return remap


def _source_segments_path(tr_dir: Path, ep: str) -> Path:
    tagged = sorted(tr_dir.glob(f"{ep}.[a-z][a-z].segments.json"))
    return tagged[0] if tagged else tr_dir / f"{ep}.segments.json"


def _load(path: Path) -> tuple[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return raw, json.loads(raw)


def retime_episode(tr_dir: Path, meta_dir: Path, ep: str, apply: bool) -> dict[str, int] | None:
    src_path = _source_segments_path(tr_dir, ep)
    transcript = TRANSCRIPTS / f"{ep}.txt"
    if not (src_path.is_file() and transcript.is_file()):
        return None
    lines, _ = synth.parse_diarized_segments(transcript)
    real = synth.measured_line_times(transcript, lines)
    raw_src, src_doc = _load(src_path)
    segs = builder._segments_of(src_doc)
    if not real or [s.get("text") for s in segs] != [line["text"] for line in lines]:
        print(f"  {ep}: lines do not match its segments, left as is")
        return None
    old = [(float(s.get("start") or 0.0), float(s.get("end") or 0.0)) for s in segs]
    remap = time_map(old, real)
    counts = {"segments": 0, "adfree": 0, "quotes": 0, "render": 0}

    for seg, (ns, ne) in zip(segs, real):
        if (seg.get("start"), seg.get("end")) != (ns, ne):
            seg["start"], seg["end"] = ns, ne
            counts["segments"] += 1
    if counts["segments"] and apply:
        src_path.write_text(builder._dump_like(raw_src, src_doc), encoding="utf-8")

    translated = src_path.name != f"{ep}.segments.json"
    if not translated:
        # An English show's ad-free body is on the same timeline as its source: map its times.
        adfree = tr_dir / f"{ep}.adfree.segments.json"
        if adfree.is_file():
            raw_a, doc_a = _load(adfree)
            for seg in builder._segments_of(doc_a):
                s, e = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
                ns, ne = remap(s), remap(e) if e else 0.0
                if (s, e) != (ns, ne):
                    seg["start"], seg["end"] = ns, ne
                    counts["adfree"] += 1
            if counts["adfree"] and apply:
                adfree.write_text(builder._dump_like(raw_a, doc_a), encoding="utf-8")
        admap = tr_dir / f"{ep}.adfree.admap.json"
        if admap.is_file():
            raw_m, doc_m = _load(admap)
            changed = False
            for key in ("preroll_cut_end", "postroll_cut_start"):
                v = doc_m.get(key)
                if isinstance(v, (int, float)) and remap(float(v)) != v:
                    doc_m[key] = remap(float(v))
                    changed = True
            if changed and apply:
                admap.write_text(builder._dump_like(raw_m, doc_m), encoding="utf-8")

    gi = meta_dir / f"{ep}.gi.json"
    if gi.is_file():
        raw_g, doc_g = _load(gi)
        for node in doc_g.get("nodes") or []:
            props = node.get("properties") if isinstance(node, dict) else None
            if node.get("type") != "Quote" or not isinstance(props, dict):
                continue
            for key in ("timestamp_start_ms", "timestamp_end_ms"):
                v = props.get(key)
                if isinstance(v, (int, float)):
                    nv = int(round(remap(v / 1000.0) * 1000))
                    if nv != v:
                        props[key] = nv
                        counts["quotes"] += 1
        if counts["quotes"] and apply:
            gi.write_text(builder._dump_like(raw_g, doc_g), encoding="utf-8")

    if translated:
        counts["render"] = builder.retime_english_render(tr_dir, ep, apply=apply)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=REPO / "tests/fixtures/app-validation-corpus/v3")
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: report)")
    args = ap.parse_args()
    totals = {"segments": 0, "adfree": 0, "quotes": 0, "render": 0}
    skipped = 0
    for tr_dir in sorted(args.corpus.glob("feeds/*/run_*/transcripts")):
        meta_dir = tr_dir.parent / "metadata"
        eps = sorted({p.name.split(".")[0] for p in tr_dir.glob("p*_e*.segments.json")})
        for ep in eps:
            counts = retime_episode(tr_dir, meta_dir, ep, args.apply)
            if counts is None:
                skipped += 1
                continue
            print(f"  {ep}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
            for k, v in counts.items():
                totals[k] += v
    verb = "moved" if args.apply else "would move"
    print(f"{verb}: " + ", ".join(f"{k} {v}" for k, v in totals.items()) + f" · skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
