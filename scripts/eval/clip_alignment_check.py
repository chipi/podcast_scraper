#!/usr/bin/env python3
"""Does a clip play what its card says? (operator 2026-10-15)

For each episode's Moments — the same picker, inputs and config the ``/episodes/{slug}/moments``
route uses — cut the clip's audio, transcribe it with Whisper, and compare:

* ``heard``: the share of the quote's content words the audio actually says (1.0 = all).
* ``offset_s``: where the quote's opening words really are, relative to the clip's start,
  found by transcribing a wider window with word timestamps (+3.0 = they come 3 s late; None =
  not found within the window).

A short listen and a learning day play the same clip windows (``app_goal_reel`` reuses
``score_candidates``), so checking Moments checks all three. Read-only: nothing is written to the
corpus. Audio comes from each episode's ``content.media_url`` unless ``--audio-dir`` holds
the media URL's file (the fixture corpus's TTS audio).

    .venv/bin/python scripts/eval/clip_alignment_check.py \\
        --corpus tests/fixtures/app-validation-corpus/v3 --audio-dir tests/fixtures/audio/v3 \\
        --limit 10 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from podcast_scraper.server.app_audio_drift import time_maps, to_stream
from podcast_scraper.server.app_episode_search import (
    episode_ad_spans,
    episode_segments,
    episode_turn_starts,
)
from podcast_scraper.server.app_moments import MomentsConfig, pick_moments
from podcast_scraper.server.catalog_cache import cached_catalog
from podcast_scraper.server.corpus_access import load_json_artifact
from podcast_scraper.server.slugs import slug_for_row

_STOP = set(
    "a an and are as at be but by do for from has have i if in is it its me my no not of on or our"
    " so that the their them then there they this to was we were what when which who will with you"
    " your just like really yeah okay well um uh".split()
)
WIDE_S = 40.0


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP and len(w) > 1]


@dataclass
class ClipCheck:
    slug: str
    episode_id: str | None
    start_s: float
    end_s: float
    clip_text: str
    heard_text: str
    heard: float
    offset_s: float | None


def media_source(root: Path, row: Any, audio_dir: Path | None) -> str | None:
    """The episode's ``content.media_url``; with ``--audio-dir``, the file of that name there."""
    try:
        doc = json.loads((root / row.metadata_relative_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = (doc.get("content") or {}).get("media_url") if isinstance(doc, dict) else None
    if not isinstance(url, str) or not url.strip():
        return None
    if audio_dir is not None:
        local = audio_dir / Path(url.split("?", 1)[0]).name
        return str(local) if local.is_file() else None
    return url.strip()


def cut(source: str, start: float, end: float, out: Path) -> bool:
    """Mono 16 kHz WAV of ``source`` from ``start`` for ``end - start`` seconds."""
    start = max(0.0, start)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{end - start:.3f}",
        "-i",
        source,
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out),
    ]
    return subprocess.run(cmd, check=False).returncode == 0 and out.is_file()


def locate(words: list[dict[str, Any]], target: list[str]) -> float | None:
    """Start time of the first content words of ``target`` in ``words``, matched IN ORDER.

    A candidate starts on a word equal to one of the target's first three words (tried in that
    order, so the target's own first word wins when it is heard) and scores how many of the
    target's words follow it in order within a short window. Without the anchor and the order,
    a window beginning a few words early still held every target word, scored the same and won as
    the earliest — every offset read about -2.5 s, fixture and prod alike (found 2026-10-15 against
    the fixture's RTTM ground truth, where segment starts are exact).
    """
    probe = target[:8]
    if len(probe) < 3:
        return None
    tokens = [(content_words(w["word"]), w["start"]) for w in words]
    flat = [(t, start) for ts, start in tokens for t in ts]
    for lead in range(3):
        best, at = 0.0, None
        for i in range(len(flat)):
            if flat[i][0] != probe[lead]:
                continue
            k, hits = lead, 0
            for t, _ in flat[i : i + len(probe) + 6]:
                if k < len(probe) and t == probe[k]:
                    hits += 1
                    k += 1
                elif k + 1 < len(probe) and t == probe[k + 1]:
                    # The word before it was misheard ("glorkeren" for "Glaucus", 2026-10-16):
                    # without the skip, the match stalls on it and the clip reads "not located".
                    hits += 1
                    k += 2
            score = hits / (len(probe) - lead)
            if score > best:
                best, at = score, flat[i][1]
        if at is not None and best >= 0.6:
            return float(at)
    return None


def check_episode(
    model: Any, root: Path, row: Any, source: str, config: MomentsConfig, mapped: bool = False
) -> list[ClipCheck]:
    picked = pick_moments(
        load_json_artifact(root, row.gi_relative_path),
        row.duration_seconds,
        config,
        episode_segments(root, row),
        episode_ad_spans(root, row),
        episode_turn_starts(root, row),
    )
    out: list[ClipCheck] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, m in enumerate(picked):
            start, end = m.start_ms / 1000, m.end_ms / 1000
            anchors = time_maps(root).get(row.metadata_relative_path) if mapped else None
            if anchors:
                start, end = to_stream(anchors, start), to_stream(anchors, end)
            clip_wav, wide_wav = Path(tmp) / f"c{i}.wav", Path(tmp) / f"w{i}.wav"
            if not cut(source, start, end, clip_wav):
                continue
            heard_text = model.transcribe(str(clip_wav), language="en", fp16=False)["text"].strip()
            # The QUOTE: the words the card shows and the clip is cut to start on. Not
            # `clip_text`, which runs from the start of the first segment the clip touches — a
            # clip starting mid-segment then "expects" words from before it, and the offset read
            # about -2.5 s on every clip, fixture and prod alike (found 2026-10-15 against the
            # fixture's RTTM ground truth, where segment starts are exact).
            expected = content_words(m.quote_text or m.text)
            got = set(content_words(heard_text))
            heard = (sum(1 for w in expected if w in got) / len(expected)) if expected else 0.0
            offset = None
            wide_start = max(0.0, start - WIDE_S)
            if cut(source, wide_start, end + WIDE_S, wide_wav):
                res = model.transcribe(
                    str(wide_wav), language="en", fp16=False, word_timestamps=True
                )
                words = [w for seg in res["segments"] for w in seg.get("words", [])]
                found = locate(words, expected)
                if found is not None:
                    offset = round(wide_start + found - start, 1)
            out.append(
                ClipCheck(
                    slug_for_row(row),
                    row.episode_id,
                    round(start, 1),
                    round(end, 1),
                    m.clip_text,
                    heard_text,
                    round(heard, 2),
                    offset,
                )
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument("--slug", action="append", default=[], help="Only these episodes (repeatable).")
    ap.add_argument("--limit", type=int, default=0, help="At most this many episodes (0 = all).")
    ap.add_argument(
        "--model", default="base.en", help="Whisper model name (cached in ~/.cache/whisper)."
    )
    ap.add_argument("--json", type=Path, default=None, help="Write every clip's result here.")
    ap.add_argument(
        "--mapped",
        action="store_true",
        help="Cut each clip at its time in the served stream, from <corpus>/audio_time_maps.json "
        "(python -m podcast_scraper.audio_sync build-maps), as the player will.",
    )
    ap.add_argument(
        "--exact",
        action="store_true",
        help="Download each episode's audio and decode it to WAV first, so every cut is "
        "sample-exact. Without it, ffmpeg seeks the (often VBR) MP3 directly, which can land "
        "seconds off — the same estimate a browser makes when it seeks.",
    )
    args = ap.parse_args()

    import whisper  # noqa: PLC0415 — heavy; only once the arguments are good

    model = whisper.load_model(args.model)
    config = MomentsConfig.from_env()
    rows = [
        r
        for r in cached_catalog(args.corpus)
        if r.has_gi and (not args.slug or slug_for_row(r) in args.slug)
    ]
    if args.limit:
        rows = rows[: args.limit]
    results: list[ClipCheck] = []
    for row in rows:
        source = media_source(args.corpus, row, args.audio_dir)
        if source is None:
            print(f"{slug_for_row(row)}: no audio", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory() as dl:
            if args.exact:
                wav = Path(dl) / "episode.wav"
                cmd = [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    source,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(wav),
                ]
                if subprocess.run(cmd, check=False).returncode != 0:
                    print(f"{slug_for_row(row)}: download failed", file=sys.stderr)
                    continue
                source = str(wav)
            checks = check_episode(model, args.corpus, row, source, config, args.mapped)
        for c in checks:
            results.append(c)
            off = "  n/a" if c.offset_s is None else f"{c.offset_s:+5.1f}"
            span = f"{c.start_s:7.1f}-{c.end_s:7.1f}s"
            print(f"{c.slug[:28]:28} {span}  heard {c.heard:4.2f}  offset {off}s")
    if not results:
        print("no clips checked")
        return 1
    ok = [
        c for c in results if c.heard >= 0.6 and c.offset_s is not None and abs(c.offset_s) <= 2.0
    ]
    offs = sorted(abs(c.offset_s) for c in results if c.offset_s is not None)
    median = offs[len(offs) // 2] if offs else "n/a"
    lost = sum(c.offset_s is None for c in results)
    print(
        f"\n{len(results)} clips · {len(ok)} play what they say (heard ≥ 0.6, |offset| ≤ 2 s) · "
        f"median |offset| {median} s · not located {lost}"
    )
    if args.json:
        args.json.write_text(json.dumps([asdict(c) for c in results], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
