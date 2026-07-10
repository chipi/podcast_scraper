#!/usr/bin/env python3
"""Generate per-episode ground-truth sidecars for the v3 fixture set.

For every ``tests/fixtures/transcripts/v3/<name>.txt`` this writes
``tests/fixtures/transcripts/v3/<name>.groundtruth.json`` declaring EXACTLY what is inside the
episode — the diarization/speaker eval's source of truth (not parsed at eval time):

- ``speakers``               distinct human speaker labels (excludes the synthetic ``Ad`` voice)
- ``num_human_speakers``     len(speakers)
- ``has_commercial`` / ``num_ad_voices``  whether an ``Ad:`` (mid-roll sponsor) voice is present
- ``expected_diarized_voices``  humans + ad voices — what a correct diarizer should DETECT
- ``type``                   monologue (1) | interview (2) | panel (>=3)
- ``failure_modes``          from the transcript's ``#fixture-v3: failure_modes=...`` annotation

Idempotent: derived purely from the transcript, so re-run after editing a transcript.

    python tests/fixtures/scripts/make_groundtruth.py            # all v3 fixtures
    python tests/fixtures/scripts/make_groundtruth.py --check    # verify sidecars are up to date
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

V3_DIR = os.path.join(os.path.dirname(__file__), "..", "transcripts", "v3")
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z '\-]{0,40}):\s+(.*)$")
HEADER_PREFIXES = ("podcast:", "episode:", "host:", "guest:", "guests:", "title:", "co-host:")
FAILURE_RE = re.compile(r"failure_modes\s*=\s*([^\s]+)")


def _episode_type(num_humans: int) -> str:
    if num_humans >= 3:
        return "panel"
    if num_humans == 2:
        return "interview"
    return "monologue"


def build_groundtruth(transcript_path: str) -> dict:
    podcast = episode = None
    speakers: list[str] = []
    seen = set()
    has_ad = False
    failure_modes: list[str] = []
    for line in open(transcript_path, encoding="utf-8").read().splitlines():
        s = line.strip()
        if s.startswith("# ") and podcast is None:
            podcast = s[2:].replace(" — Episode", "").strip()
            continue
        if s.startswith("## ") and episode is None:
            episode = s[3:].strip()
            continue
        if s.startswith("#fixture-v3:"):
            m = FAILURE_RE.search(s)
            if m:
                failure_modes.extend(x for x in m.group(1).split(",") if x)
            continue
        if s.startswith("#") or s.lower().startswith(HEADER_PREFIXES):
            continue
        m = SPEAKER_RE.match(s)
        if not m:
            continue
        name = m.group(1).strip()
        if name == "Ad":
            has_ad = True
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            speakers.append(name)

    num_ad = 1 if has_ad else 0
    return {
        "fixture": os.path.basename(transcript_path).replace(".txt", ""),
        "podcast": podcast,
        "episode": episode,
        "type": _episode_type(len(speakers)),
        "speakers": speakers,
        "num_human_speakers": len(speakers),
        "has_commercial": has_ad,
        "num_ad_voices": num_ad,
        "expected_diarized_voices": len(speakers) + num_ad,
        "failure_modes": sorted(set(failure_modes)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify sidecars match transcripts")
    args = ap.parse_args()

    stale = []
    written = 0
    for t in sorted(glob.glob(os.path.join(V3_DIR, "*.txt"))):
        gt = build_groundtruth(t)
        out = t.replace(".txt", ".groundtruth.json")
        payload = json.dumps(gt, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            current = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
            if current != payload:
                stale.append(os.path.basename(out))
            continue
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(payload)
        written += 1

    if args.check:
        if stale:
            print("STALE ground-truth sidecars (run make_groundtruth.py):", *stale, sep="\n  ")
            return 1
        print("all v3 ground-truth sidecars up to date")
        return 0
    print(f"wrote {written} ground-truth sidecars to {os.path.normpath(V3_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
