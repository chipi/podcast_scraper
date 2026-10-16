#!/usr/bin/env python3
"""Map an episode's transcript times onto the audio its host serves NOW (#2359, option A).

Hosts that insert ads at fetch time change them, so the stream a listener gets no longer lines up
with the audio we transcribed (and did not keep). This listens to short samples of the stream as
served now, finds their words in our transcript, and records how far the stream has moved:

    {"<metadata_relative_path>": {"checked_at": "...", "anchors": [[transcript_s, offset_s], ...]}}

``offset_s`` applies from ``transcript_s`` until the next anchor: stream time = transcript time +
offset. Where two neighbouring samples disagree (an ad changed between them), more samples are
taken in between until the change is placed to within ``--precision`` seconds.

Episodes come from ``<corpus>/audio_timing_drift.json`` (``audio_drift_check.py``), or ``--slug``.
English only (Whisper ``base.en``). Read-only on the corpus unless ``--write``, which merges into
``<corpus>/audio_time_maps.json`` atomically; undo = delete that file or the episode's entry.

    .venv/bin/python scripts/ops/audio_retime.py --corpus <root> [--write] [--path <metadata path>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

VERBOSE = False
DRIFT_FILE = "audio_timing_drift.json"
_STOP = set(
    "a an and are as at be but by do for from has have i if in is it its me my no not of on or our"
    " so that the their them then there they this to was we were what when which who will with you"
    " your just like really yeah okay well um uh".split()
)


def words_of(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP and len(w) > 1]


def transcript_words(segments: Sequence[dict]) -> List[Tuple[str, float]]:
    """(content word, time) in order; word times when the transcript has them, else spread
    evenly across the segment."""
    out: List[Tuple[str, float]] = []
    for s in segments:
        start, end = float(s.get("start") or 0.0), float(s.get("end") or 0.0)
        ws = s.get("words") or []
        if ws:
            for w in ws:
                for cw in words_of(str(w.get("word") or "")):
                    out.append((cw, float(w.get("start") or start)))
        else:
            cws = words_of(str(s.get("text") or ""))
            for i, cw in enumerate(cws):
                out.append((cw, start + (end - start) * i / max(1, len(cws))))
    return out


def find(
    tw: List[Tuple[str, float]], heard: List[Tuple[str, float]], near: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """(transcript time, stream time) for one sample, by vote: every run of three heard words
    that occurs in the transcript proposes an offset; the largest group within 1.5 s wins, and
    needs three votes. ``near`` (an expected offset) breaks ties between groups."""
    index: dict = {}
    for i in range(len(tw) - 2):
        index.setdefault((tw[i][0], tw[i + 1][0], tw[i + 2][0]), []).append(tw[i][1])
    votes: List[Tuple[float, float]] = []  # (offset, stream t)
    for j in range(len(heard) - 2):
        hits = index.get((heard[j][0], heard[j + 1][0], heard[j + 2][0]), [])
        if 0 < len(hits) <= 3:
            votes += [(heard[j][1] - t, heard[j][1]) for t in hits]
    if not votes:
        return None
    votes.sort()
    best: Optional[Tuple[int, float, List[Tuple[float, float]]]] = None
    for a in range(len(votes)):
        group = [v for v in votes[a:] if v[0] - votes[a][0] <= 1.5]
        tie = -abs(group[len(group) // 2][0] - near) if near is not None else 0.0
        if best is None or (len(group), tie) > (best[0], best[1]):
            best = (len(group), tie, group)
    assert best is not None
    if best[0] < 3:
        return None
    off = sorted(v[0] for v in best[2])[len(best[2]) // 2]
    st = min(v[1] for v in best[2])
    return st - off, st


class Listener:
    """Whisper over a local WAV of the served stream."""

    def __init__(self, wav: Path, model: Any) -> None:
        self.wav, self.model = wav, model

    def heard(self, at: float, length: float = 20.0) -> List[Tuple[str, float]]:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, at):.2f}"]
            cmd += ["-t", f"{length:.2f}", "-i", str(self.wav), "-ac", "1", "-ar", "16000", f.name]
            subprocess.run(cmd, check=True)
            res = self.model.transcribe(f.name, language="en", fp16=False, word_timestamps=True)
        out: List[Tuple[str, float]] = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                for cw in words_of(w["word"]):
                    out.append((cw, at + float(w["start"])))
        return out


def build_map(
    tw: List[Tuple[str, float]],
    listener: Listener,
    served_len: float,
    step: float = 120.0,
    precision: float = 4.0,
) -> List[List[float]]:
    """Anchors ``[transcript_s, offset_s]``. One sample every ``step`` seconds of the stream; then,
    between two neighbours whose offsets differ (an ad was added, removed or changed between them),
    a binary search for where the talk resumes. A sample that matches nothing is inside the ad."""

    def sample(at: float) -> Optional[Tuple[float, float]]:
        hit = find(tw, listener.heard(at))
        if VERBOSE:
            print(
                f"    sample @{at:7.1f}s -> {hit and (round(hit[1], 1), round(hit[1] - hit[0], 1))}"
            )
        return None if hit is None else (hit[1], hit[1] - hit[0])

    samples = [s for s in (sample(t) for t in _frange(5.0, served_len - 10, step)) if s]
    samples.sort()
    anchors: List[List[float]] = []
    for k, (st, off) in enumerate(samples):
        if not anchors:
            anchors.append([0.0, round(off, 1)])
            continue
        prev_st, prev_off = samples[k - 1]
        if abs(off - prev_off) <= 2.0:
            continue
        lo, hi = prev_st, st  # talk at the new offset starts after lo and by hi
        while hi - lo > precision:
            mid = (lo + hi) / 2
            got = sample(mid)
            if got is not None and abs(got[1] - off) <= 2.0:
                hi = min(hi, got[0])
                if got[0] > mid:
                    lo = mid
            else:
                lo = mid if got is None else max(mid, got[0])
        anchors.append([round(hi - off, 1), round(off, 1)])
    return anchors


def _frange(a: float, b: float, step: float) -> List[float]:
    out = []
    while a < b:
        out.append(a)
        a += step
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--path", action="append", default=[], help="metadata path(s) to re-time")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--verbose", action="store_true", help="print every sample")
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    import whisper  # noqa: PLC0415 — heavy

    from podcast_scraper.server.app_audio_drift import MAPS_FILE
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_episode_search import _segment_dicts, episode_segments_relpath

    root = args.corpus
    paths = set(args.path)
    if not paths:
        drift = json.loads((root / DRIFT_FILE).read_text(encoding="utf-8"))
        paths = set(drift.get("episodes", {}))
    model = whisper.load_model(args.model)
    maps = {}
    for row in cached_catalog(root):
        if row.metadata_relative_path not in paths:
            continue
        doc = json.loads((root / row.metadata_relative_path).read_text(encoding="utf-8"))
        url = (doc.get("content") or {}).get("media_url")
        rel = episode_segments_relpath(root, row)
        if not url or rel is None:
            continue
        rel = rel.replace(".adfree.segments.json", ".segments.json")
        tw = transcript_words(_segment_dicts(root / rel))
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "episode.wav"
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", url, "-ac", "1", "-ar", "16000", str(wav)]
            if subprocess.run(cmd, check=False).returncode != 0:
                print(f"skip (download failed): {row.metadata_relative_path}")
                continue
            dur = float(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "csv=p=0",
                        str(wav),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                or 0
            )
            anchors = build_map(tw, Listener(wav, model), dur)
        if not anchors:
            print(f"skip (no sample matched): {row.metadata_relative_path}")
            continue
        maps[row.metadata_relative_path] = {
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "anchors": anchors,
        }
        print(f"{row.metadata_relative_path[:70]}  {anchors}")
    if not args.write:
        print(f"dry run: {len(maps)} maps, nothing written")
        return 0
    out = root / MAPS_FILE
    try:
        current = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    current.update(maps)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(current, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {len(maps)} maps to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
