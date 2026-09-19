#!/usr/bin/env python3
"""Turn a speaker-prefixed fixture transcript into WebVTT that NAMES EVERY TURN.

WHY THIS EXISTS
---------------
``tests/stack-test`` mounts only the two RSS fixtures that ship a ``<podcast:transcript>`` for
every item, deliberately, so Whisper never runs (see the comment in
``compose/docker-compose.stack-test.yml``). Those transcripts were **plain text**, which carries no
speaker turns — so every stack-test episode arrived as one undifferentiated voice, nothing was ever
``placed``, and after the #2075 operator decision ("an episode never diarized, or diarized with
nobody named, casts nobody") the KG correctly emitted zero ``Person`` nodes and the Person-rail
specs went red.

The fix is NOT to make the stack run a diarizer — that is the expensive processing the whole
transcript fast path exists to avoid. It is to hand the pipeline a transcript that already says who
is speaking, which is exactly what real publisher transcripts do. ``transcript_formats.cues``
already parses ``<v Speaker>`` voice spans; this script produces them.

The corpus also ships ``.rttm`` diarization ground truth, but nothing in ``src/podcast_scraper``
reads RTTM — it is scoring-only — and it is not needed here: the ``.txt`` fixtures already prefix
every line with the speaker, so the transcript is its own source of truth and the generated VTT
cannot disagree with the text the corpus already asserts.

TIMING
------
The fixtures mark blocks with ``[MM:SS]`` and give no per-turn times. Cue times are therefore
SYNTHESISED: each turn starts where the previous one ended and runs for a length proportional to
its text, seeded at each ``[MM:SS]`` marker. Two consequences worth knowing:

* Times are plausible, not measured. Nothing asserts them against the audio; they exist so the cue
  file is well-formed and segments carry ordering.
* Markers in the fixtures are not always monotonic (``p01_e01_fast`` goes 00:00 -> 03:30 -> 01:00).
  A cue that would start before the previous one ended is clamped forward, because a WebVTT file
  with overlapping/backwards cues is not one the parser should have to tolerate.

Usage::

    python tests/fixtures/scripts/transcripts_to_vtt.py p01_e01_fast p01_multi_e01
    python tests/fixtures/scripts/transcripts_to_vtt.py --all
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List, Optional, Tuple

#: ``[MM:SS]`` (or ``[HH:MM:SS]``) block marker on a line of its own.
_MARKER = re.compile(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\]\s*$")

#: ``Speaker: words`` — the speaker is a short human-ish label, never a sentence. Bounded to keep a
#: line like "Note: the trail was wet" from being read as a speaker turn.
_TURN = re.compile(r"^([A-Z][A-Za-z.'’\- ]{0,40}?):\s+(.*\S)\s*$")

#: Header labels that LOOK like a turn but describe the episode. ``p01_e01_fast.txt`` opens with
#: ``Host: Maya`` / ``Guest: Liam``; treating those as speech would invent a speaker called "Host".
_HEADER_LABELS = {"host", "guest", "hosts", "guests", "speaker", "speakers", "note", "summary"}

#: Labels that mark WHO IS TALKING as not-a-person. The fixtures prefix sponsor reads with ``Ad:``,
#: and the groundtruth agrees it is not a speaker: ``p07_e01`` lists ``speakers: ["Alex Morgan",
#: "Dr. Elena Fischer"]`` while carrying ``num_ad_voices: 1`` and putting ``Ad`` in ``voice_map``
#: only. So the ad is a VOICE, never a name.
#:
#: Their turns still become cues — the text is real, the ad-free sidecar is built from it, and
#: dropping it would silently shorten the transcript — but they carry NO ``<v>`` span. Emitting
#: ``<v Ad>`` would hand the roster a speaker literally named "Ad" and invite it to publish a
#: person nobody is, which is the #876 failure this corpus exists to catch, not to cause.
#:
#: MEASURED: across all 46 v3 fixtures, ``Ad`` is the ONLY label the generated VTT named that the
#: groundtruth's ``speakers`` list does not (30 files), and nothing was ever missing. The synonyms
#: below do not occur today; they are here so a new fixture cannot reintroduce the defect quietly.
_NON_PERSON_VOICES = {"ad", "ads", "advertisement", "sponsor", "announcer", "promo"}

#: Seconds per character, and the floor for a very short turn. Chosen so a typical fixture turn
#: lands in the 2-8s range that real speech occupies; nothing depends on the exact value.
_SEC_PER_CHAR = 0.06
_MIN_CUE_SEC = 1.5

_DEFAULT_DIR = pathlib.Path("tests/fixtures/transcripts/v3")


def _marker_seconds(line: str) -> Optional[float]:
    m = _MARKER.match(line)
    if not m:
        return None
    hours = int(m.group(1) or 0)
    return float(hours * 3600 + int(m.group(2)) * 60 + int(m.group(3)))


def parse_turns(text: str) -> List[Tuple[str, str, Optional[float]]]:
    """``(speaker, utterance, block_start_or_None)`` for every spoken line, in file order.

    Lines before the first ``[MM:SS]`` marker are episode headers, not speech: the fixtures put
    ``Host:``/``Guest:`` there, and admitting them would create speakers nobody said.
    """
    turns: List[Tuple[str, str, Optional[float]]] = []
    pending_start: Optional[float] = None
    seen_marker = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        marker = _marker_seconds(line)
        if marker is not None:
            pending_start = marker
            seen_marker = True
            continue
        if not seen_marker:
            continue
        m = _TURN.match(line)
        if not m:
            continue
        speaker = m.group(1).strip()
        if speaker.lower() in _HEADER_LABELS:
            continue
        if speaker.lower() in _NON_PERSON_VOICES:
            # Keep the words, drop the identity — see _NON_PERSON_VOICES.
            speaker = ""
        turns.append((speaker, m.group(2).strip(), pending_start))
        pending_start = None

    return turns


def _timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _escape(text: str) -> str:
    """WebVTT cue payload. ``<`` and ``&`` would otherwise read as markup — and a stray ``<``
    directly before a word is exactly the shape of the voice span this file relies on."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_vtt(text: str) -> Optional[str]:
    """The WebVTT body, or ``None`` when the transcript names no turns at all."""
    turns = parse_turns(text)
    if not turns:
        return None

    lines = ["WEBVTT", ""]
    cursor = 0.0
    for speaker, utterance, block_start in turns:
        start = cursor if block_start is None else max(block_start, cursor)
        end = start + max(_MIN_CUE_SEC, len(utterance) * _SEC_PER_CHAR)
        lines.append(f"{_timestamp(start)} --> {_timestamp(end)}")
        body = _escape(utterance)
        lines.append(f"<v {speaker}>{body}</v>" if speaker else body)
        lines.append("")
        cursor = end

    return "\n".join(lines)


def convert(base: str, directory: pathlib.Path) -> bool:
    src = directory / f"{base}.txt"
    if not src.is_file():
        print(f"  {base}: no {src} — skipped", file=sys.stderr)
        return False
    vtt = build_vtt(src.read_text(encoding="utf-8"))
    if vtt is None:
        print(f"  {base}: transcript names no turns — skipped", file=sys.stderr)
        return False
    dst = directory / f"{base}.vtt"
    dst.write_text(vtt, encoding="utf-8")
    speakers = sorted({s for s, _, _ in parse_turns(src.read_text(encoding="utf-8")) if s})
    print(f"  {base}.vtt: {vtt.count('-->')} cues, speakers={speakers}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bases", nargs="*", help="fixture basenames, e.g. p01_multi_e01")
    ap.add_argument(
        "--dir", default=str(_DEFAULT_DIR), help=f"fixture dir (default {_DEFAULT_DIR})"
    )
    ap.add_argument("--all", action="store_true", help="convert every .txt in --dir")
    args = ap.parse_args()

    directory = pathlib.Path(args.dir)
    bases = sorted(p.stem for p in directory.glob("*.txt")) if args.all else args.bases
    if not bases:
        ap.error("give at least one basename, or --all")

    written = sum(convert(b, directory) for b in bases)
    print(f"wrote {written}/{len(bases)} VTT file(s) to {directory}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
