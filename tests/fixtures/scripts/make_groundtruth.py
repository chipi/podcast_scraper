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
- ``voice_map``              speaker (incl ``Ad``) -> the say voice it is rendered with, from the
                             ONE-VOICE-PER-PERSON map (FIXTURES_SPEC.md) — records EXACTLY who
                             sounds like what
- ``cameo``                  when ``cameo`` tagged: {speaker, voice, turns} of the brief 3rd voice
- ``transcript_sha256`` / ``audio_sha256``  reality-check hashes. ``--check`` recomputes both from
                             disk and fails if they drift, so audio/transcript can never change
                             without the sidecar being regenerated to match.

The sidecar is the full per-episode spec and the fixtures' reality check: whenever a transcript,
the voice map, or an audio file changes, regenerate the sidecars (they carry the new hashes).

Idempotent: derived purely from the transcript + voice map + audio file on disk.

    python tests/fixtures/scripts/make_groundtruth.py            # all v3 fixtures
    python tests/fixtures/scripts/make_groundtruth.py --check    # verify sidecars are up to date
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

V3_DIR = os.path.join(os.path.dirname(__file__), "..", "transcripts", "v3")
AUDIO_V3_DIR = os.path.join(os.path.dirname(__file__), "..", "audio", "v3")


def _load_voice_resolver():
    """Load ``get_voice_for_speaker`` from the sibling audio generator (single source
    of truth for the ONE-VOICE-PER-PERSON map; see tests/fixtures/FIXTURES_SPEC.md)."""
    path = os.path.join(os.path.dirname(__file__), "transcripts_to_mp3.py")
    spec = importlib.util.spec_from_file_location("transcripts_to_mp3", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.get_voice_for_speaker


_voice_for = _load_voice_resolver()


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Names may contain internal periods ("A. correspondent", "Dr. Elena Fischer");
# without '.' in the class those speakers are silently dropped, undercounting
# expected_diarized_voices and mislabelling type (#1170).
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z .'\-]{0,40}):\s+(.*)$")
HEADER_PREFIXES = ("podcast:", "episode:", "host:", "guest:", "guests:", "title:", "co-host:")
FAILURE_RE = re.compile(r"failure_modes\s*=\s*([^\s]+)")


def _episode_type(num_humans: int) -> str:
    if num_humans >= 3:
        return "panel"
    if num_humans == 2:
        return "interview"
    return "monologue"


def _parse_transcript(transcript_path: str) -> dict:
    """Extract podcast/episode headers, ordered distinct human speakers, per-speaker
    turn counts, ad presence, and failure-mode tags from a transcript."""
    podcast = episode = None
    speakers: list[str] = []
    seen: set[str] = set()
    has_ad = False
    failure_modes: list[str] = []
    turns: dict[str, int] = {}
    for line in open(transcript_path, encoding="utf-8").read().splitlines():
        s = line.strip()
        if s.startswith("# ") and podcast is None:
            podcast = s[2:].replace(" — Episode", "").strip()
        elif s.startswith("## ") and episode is None:
            episode = s[3:].strip()
        elif s.startswith("#fixture-v3:"):
            m = FAILURE_RE.search(s)
            if m:
                failure_modes.extend(x for x in m.group(1).split(",") if x)
        elif s.startswith("#") or s.lower().startswith(HEADER_PREFIXES):
            continue
        else:
            m = SPEAKER_RE.match(s)
            if not m:
                continue
            name = m.group(1).strip()
            turns[name] = turns.get(name, 0) + 1
            if name == "Ad":
                has_ad = True
            elif name.lower() not in seen:
                seen.add(name.lower())
                speakers.append(name)
    return {
        "podcast": podcast,
        "episode": episode,
        "speakers": speakers,
        "has_ad": has_ad,
        "failure_modes": failure_modes,
        "turns": turns,
    }


def build_groundtruth(transcript_path: str) -> dict:
    p = _parse_transcript(transcript_path)
    podcast, episode = p["podcast"], p["episode"]
    speakers, has_ad, turns = p["speakers"], p["has_ad"], p["turns"]
    num_ad = 1 if has_ad else 0
    modes = sorted(set(p["failure_modes"]))
    # Voice per voiced speaker (humans + Ad), from the ONE-VOICE-PER-PERSON map — so the
    # sidecar records EXACTLY which say voice each person is rendered with (FIXTURES_SPEC).
    voiced = list(speakers) + (["Ad"] if has_ad else [])
    voice_map = {spk: _voice_for(spk) for spk in voiced}
    # Cameo detail: when tagged ``cameo``, the cameo is the briefest human voice (one
    # short turn) — record who + which voice so evals know the brief-3rd-voice target.
    cameo = None
    if "cameo" in modes and speakers:
        cam = min(speakers, key=lambda spk: turns.get(spk, 0))
        cameo = {"speaker": cam, "voice": voice_map.get(cam), "turns": turns.get(cam, 0)}
    fixture = os.path.basename(transcript_path).replace(".txt", "")
    audio_path = os.path.join(AUDIO_V3_DIR, fixture + ".mp3")
    return {
        "fixture": fixture,
        "podcast": podcast,
        "episode": episode,
        "type": _episode_type(len(speakers)),
        "speakers": speakers,
        "num_human_speakers": len(speakers),
        "has_commercial": has_ad,
        "num_ad_voices": num_ad,
        "expected_diarized_voices": len(speakers) + num_ad,
        "failure_modes": modes,
        # --- fixture reality-check (#1170): the sidecar is the full per-episode spec ---
        "voice_map": voice_map,
        "cameo": cameo,
        "transcript_sha256": _sha256(transcript_path),
        "audio_sha256": _sha256(audio_path),
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
