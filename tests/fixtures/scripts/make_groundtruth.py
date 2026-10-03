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
- ``voice_pitch_shift``     speaker -> post-synthesis pitch factor, when one was applied. A
                            shifted voice is not the voice it came from, and this field is what
                            stops the ``voice_map`` above from implying it is.
- ``voice_map``              speaker (incl ``Ad``) -> the say voice it is rendered with, from the
                             ONE-VOICE-PER-PERSON map (FIXTURES_SPEC.md) — records EXACTLY who
                             sounds like what
- ``cameo``                  when ``cameo`` tagged: {speaker, voice, turns} of the brief 3rd voice
- ``transcript_sha256`` / ``audio_sha256`` / ``rttm_sha256``  reality-check hashes (rttm =
                             the per-turn diarization reference; null for the _fast fixture).
                             ``--check`` recomputes them and fails if they drift, so transcript,
                             audio, or RTTM cannot change without a matching sidecar regen.

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


def _load_audio_generator():
    """Load the sibling audio generator — the single source of truth for the
    ONE-VOICE-PER-PERSON(-PER-LANGUAGE) map; see tests/fixtures/FIXTURES_SPEC.md."""
    path = os.path.join(os.path.dirname(__file__), "transcripts_to_mp3.py")
    spec = importlib.util.spec_from_file_location("transcripts_to_mp3", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_t2m = _load_audio_generator()
_voice_for = _t2m.get_voice_for_speaker
_transcript_language = _t2m.transcript_language
_pitch_shift_for = _t2m.pitch_shift_for


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
#: Unicode-aware. `[A-Za-z]` cannot see the `í` in `Lucía Herrera`, so she stopped being a
#: speaker: her 26 turns vanished from the sidecar, `voice_map` came back with the guest and
#: the ad only, and `expected_diarized_voices` said 2 against an RTTM holding 3. Nothing
#: errored — an unmatched line is simply not a turn. Same fix as transcripts_to_vtt.py.
SPEAKER_RE = re.compile(r"^([A-Za-zÀ-ÖØ-Þ][\w .'\-]{0,40}):\s+(.*)$")
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
    # THE TRANSCRIPT'S LANGUAGE DECIDES THE VOICE. Resolving without it recorded the ENGLISH
    # voices for the Spanish fixture — `Samantha`/`Ralph` in the sidecar while the audio was
    # actually rendered with `Monica`/`Paulina`. This field's whole job is to record exactly who
    # sounds like what, so a language-blind lookup makes it a lie rather than a gap.
    language = _transcript_language(open(transcript_path, "r", encoding="utf-8").read())
    voice_map = {spk: _voice_for(spk, language) for spk in voiced}
    # A pitch-shifted voice is NOT the voice it was synthesised from, and this field's stated
    # job is to record exactly who sounds like what. `Paulina` alone would have a reader expect
    # her native 164.9 Hz when the fixture actually carries her at 77.4 Hz. See
    # `VOICE_PITCH_SHIFT` in transcripts_to_mp3.py for why the shift exists at all.
    pitch_shifts = {
        spk: shift for spk in voiced if (shift := _pitch_shift_for(spk, language)) is not None
    }
    # Cameo detail: when tagged ``cameo``, the cameo is the briefest human voice (one
    # short turn) — record who + which voice so evals know the brief-3rd-voice target.
    cameo = None
    if "cameo" in modes and speakers:
        cam = min(speakers, key=lambda spk: turns.get(spk, 0))
        cameo = {"speaker": cam, "voice": voice_map.get(cam), "turns": turns.get(cam, 0)}
    fixture = os.path.basename(transcript_path).replace(".txt", "")
    audio_path = os.path.join(AUDIO_V3_DIR, fixture + ".mp3")
    # Per-turn diarization reference (#1170). Co-located with the transcript, emitted by
    # transcripts_to_mp3.py --rttm-only from the deterministic aiff timeline. None for the
    # ffmpeg-truncated _fast fixture, which is not a diarization eval fixture.
    rttm_path = os.path.splitext(transcript_path)[0] + ".rttm"
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
        # Empty for every fixture that uses its voices natively, which is all of them but the
        # Spanish one.
        "voice_pitch_shift": pitch_shifts,
        "cameo": cameo,
        "transcript_sha256": _sha256(transcript_path),
        "audio_sha256": _sha256(audio_path),
        "rttm_sha256": _sha256(rttm_path),
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
