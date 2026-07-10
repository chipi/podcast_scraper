#!/usr/bin/env python3
"""
Generate MP3 files from .txt transcripts using either macOS ``say`` (default,
offline + deterministic) or **Gemini 2.5 multi-speaker TTS** (cloud +
naturalistic, #934).

Per-speaker voice mapping (RFC-059 §2 / issue #111) replaces the prior
binary host/guest scheme so diarization tests (RFC-058) can actually
distinguish speakers in fixture audio.

Backend selection: ``--backend say`` (default, macOS-only) or
``--backend gemini``.

For the ``say`` backend:
    Voice resolution order:
    1. Exact name match in SPEAKER_VOICE_MAP
    2. First-word match (e.g. "Alex Morgan" -> "Alex")
    3. Stable hash-based fallback (md5; deterministic across runs)
    Requires: /usr/bin/say, ffmpeg.

For the ``gemini`` backend:
    Each speaker maps to a prebuilt Gemini voice via
    SPEAKER_GEMINI_VOICE_MAP. Single-call multi-speaker TTS for transcripts
    with ≤ 2 distinct speakers; per-segment fallback for ≥ 3 speakers
    (Gemini multi-speaker mode is capped at 2 voices per call as of 2026).
    Output is non-deterministic — regen drifts byte-by-byte. Requires:
    GEMINI_API_KEY env var, ffmpeg.

Output version is derived from the input transcript's path
(``.../transcripts/v2/...`` -> ``.../audio/v2/...``); explicit
``--output-version`` overrides; otherwise FIXTURES_VERSION is read from
``tests/fixtures/FIXTURES_VERSION``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Update if you rename podcast IDs or hosts.
PODCAST_HOSTS: dict[str, str] = {
    "p01": "Maya",
    "p02": "Ethan",
    "p03": "Rina",
    "p04": "Leo",
    "p05": "Nora",
    "p06": "TBD",  # Edge cases podcast — v2 may name an actual host
    "p07": "Alex Morgan",  # The Long View - Sustainability
    "p08": "Alex Morgan",  # The Long View - Solar Energy
    "p09": "Alex Morgan",  # The Long View - Biohacking
}

# Per-speaker voice mapping (RFC-059 §2). Each fixture speaker gets a distinct
# macOS ``say`` voice with accent variety so diarization can actually separate
# them. Listing-order convention: hosts first, then guests, then synthetic.
SPEAKER_VOICE_MAP: dict[str, str] = {
    # Hosts (varied accents)
    "Maya": "Samantha",  # en_US female
    "Ethan": "Alex",  # en_US male
    "Rina": "Karen",  # en_AU female
    "Leo": "Daniel",  # en_GB male
    "Nora": "Moira",  # en_IE female
    "Alex": "Evan",  # en_US male (p07-p09 host: "Alex Morgan" -> first word)
    # Guests (varied accents for speaker distinction)
    "Liam": "Daniel",  # en_GB male (cross-accent from host Samantha so pyannote
    #                    reliably separates them on the short p01_multi_e01 clip)
    "Sophie": "Flo",  # en_GB female
    "Noah": "Tom",  # en_US male
    "Priya": "Isha",  # en_IN female
    "Jonas": "Eddy",  # en_US male
    "Camila": "Paulina",  # es_MX female
    "Marco": "Lee",  # en_AU male (it_IT "Luca" not installed; Lee keeps it male + distinct)
    "Hanna": "Anna",  # de_DE female
    "Owen": "Reed",  # en_US male
    "Ava": "Kathy",  # en_US female
    "Tariq": "Rishi",  # en_IN male
    "Elise": "Amélie",  # fr_CA female (exact installed name carries the accent)
    "Daniel": "Oliver",  # en_GB male
    "Isabel": "Mónica",  # es_ES female (exact installed name carries the accent)
    "Kasper": "Ralph",  # en_US male
    # p07-p09 recurring cast (#1170). Were falling to the novelty hash-fallback,
    # which collided two humans onto one voice (p09_e03: Sam+Skanda both -> Shelley,
    # so exp=3 was acoustically unreachable) and voiced the recurring guest
    # inconsistently across episodes (Skanda -> Trinoids in e02 but Shelley in e03).
    # Curated to distinct, natural, accent-appropriate voices matching each
    # sidecar's declared voice/host_voice intent.
    "Sam": "Nathan",  # en_US male host (host_voice=en-US)
    "Skanda": "Rishi",  # en_IN male guest (voice=en-IN); first-word rule covers
    #                     both "Skanda Amarnauth" (e02) and "Skanda Amarnath" (e03)
    "Renee": "Allison",  # en_US female guest (voice=en-US)
    "Scott": "Fred",  # en_US male panelist (was Whisper; distinct from Oliver/Moira)
    # Period-names — exact-match keys (first-word "A."/"Dr." would over-match). These
    # were regex-dropped before (#1170): host folded into hash voice / guest folded
    # into the host voice. Now they get their own distinct, accent-appropriate voice.
    "A. correspondent": "Tom",  # en_US male host (host_voice=en-US); non-NER name kept
    "Dr. Elena Fischer": "Anna",  # de_DE female guest (voice=de-DE)
    # p02/p04/p05/p06 guests+hosts that were falling to novelty hash-fallback
    # voices (Trinoids/Nicky/Bells/Whisper) — unrealistic for humans and fragile
    # to diarize. Curated to natural, accent-appropriate, in-fixture-distinct voices.
    "Joll": "Nathan",  # en_US male guest (voice=en-US), distinct from host Ethan
    "Tarek": "Majed",  # ar male guest (voice=ar-EG), multi-accent vs en_GB host Leo
    "Rich": "Tom",  # en_US male guest (voice=en-US), distinct from en_IE host Nora
    "Cam": "Evan",  # en_US host of p06 (host_voice=en-US)
    "Jordan": "Allison",  # en_US female guest of p06 (voice=en-US), distinct from Cam
    # #1170 cameo voices — a genuine brief 3rd voice, distinct from that episode's
    # host + guest, so the diarization squelch keeps a real ~3s cameo.
    "Caller": "Karen",  # en_AU F call-in (p02_e05) — max-distinct from Ethan/Priya
    "Nadia Sereni": "Anna",  # de_DE F archival clip (p05_e05) — max-distinct from Nora/Daniel
    # Synthetic / non-human speakers (RFC-059 §3 / issue #109).
    # Pre-recorded mid-roll ads use the ``Ad:`` speaker label; Zarvox is
    # deliberately robotic so listeners + diarization both flag it as
    # distinct from the host-read sponsor segments.
    "Ad": "Zarvox",
}

# Hash-based fallback for speakers not in SPEAKER_VOICE_MAP. Order matters
# for stability — appending is safe; reordering or deleting changes
# previously generated fallback assignments. Picked for clear distinction
# from each other and from the curated voices above.
FALLBACK_VOICES: list[str] = [
    "Albert",
    "Bruce",
    "Junior",
    "Nicky",
    "Shelley",
    "Trinoids",
    "Whisper",
    "Bells",
]

# Gemini 2.5 prebuilt voices per speaker (#934). Voice names are the
# canonical ones the Gemini TTS API recognises in 2026: Kore, Aoede, Puck,
# Charon, Fenrir, Leda, Orus, Zephyr. Each speaker label gets a distinct
# voice for diarization separability — the mapping intentionally varies
# across hosts so the gemini-backed fixtures behave like the say-backed
# ones (different speaker, different voice).
SPEAKER_GEMINI_VOICE_MAP: dict[str, str] = {
    # Hosts
    "Maya": "Kore",  # warm + clear, common host pick
    "Ethan": "Charon",  # lower-pitched male
    "Rina": "Leda",  # bright female
    "Leo": "Fenrir",  # mid male
    "Nora": "Zephyr",  # airy female
    "Alex": "Orus",  # neutral male
    # Guests — picked to contrast against each host they pair with
    "Liam": "Puck",  # higher-energy male (contrast vs Kore on p01)
    "Sophie": "Aoede",
    "Noah": "Charon",
    "Priya": "Leda",
    "Jonas": "Orus",
    "Camila": "Aoede",
    "Marco": "Puck",
    "Hanna": "Leda",
    "Owen": "Fenrir",
    "Ava": "Aoede",
    "Tariq": "Orus",
    "Elise": "Leda",
    "Daniel": "Charon",
    "Isabel": "Aoede",
    "Kasper": "Fenrir",
    # Ad reads — robotic-shaped pick (Charon is the closest to Zarvox-y).
    "Ad": "Charon",
}

GEMINI_FALLBACK_VOICES: list[str] = [
    "Kore",
    "Aoede",
    "Puck",
    "Charon",
    "Fenrir",
    "Leda",
    "Orus",
    "Zephyr",
]

TS_RE = re.compile(r"^\[\s*\d{1,2}:\d{2}(:\d{2})?\s*\]$")
# Names may contain internal periods ("A. correspondent", "Dr. Elena Fischer");
# without '.' those turns fall through to the host voice instead of their own (#1170).
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z .'\-]{0,40}):\s+(.*)$")
FILE_PREFIX_RE = re.compile(r"^(p\d{2})_", re.IGNORECASE)
VERSION_SEGMENT_RE = re.compile(r"^v\d+$")


def get_voice_for_speaker(name: str) -> str:
    """Resolve a speaker name to a macOS ``say`` voice.

    Lookup precedence: exact -> first word -> stable hash fallback.
    """
    name = name.strip()
    if name in SPEAKER_VOICE_MAP:
        return SPEAKER_VOICE_MAP[name]
    parts = name.split()
    if parts and parts[0] in SPEAKER_VOICE_MAP:
        return SPEAKER_VOICE_MAP[parts[0]]
    # Stable hash (Python's built-in hash() varies across runs). Not a security
    # boundary — just a deterministic bucket selector for voice assignment.
    digest = hashlib.md5(name.lower().encode("utf-8"), usedforsecurity=False).hexdigest()
    return FALLBACK_VOICES[int(digest, 16) % len(FALLBACK_VOICES)]


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def parse_segments(raw: str, host_name: str) -> list[tuple[str, str]]:
    """Parse transcript text into a list of (speaker_name, text) segments.

    Free-form narrative lines (no ``Speaker:`` prefix) are attributed to the
    host. Consecutive segments by the same speaker are merged so we don't
    re-invoke ``say`` for every sentence.
    """
    out: list[tuple[str, str]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if TS_RE.fullmatch(s):
            continue
        low = s.lower()
        if s.startswith("#") or low.startswith(
            ("podcast:", "episode:", "host:", "guest:", "title:")
        ):
            continue
        if s.startswith("*") and s.endswith("*"):
            continue

        m = SPEAKER_RE.match(s)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            if not text:
                continue
            out.append((speaker, text))
        else:
            out.append((host_name, s))

    # Merge consecutive segments from the same speaker.
    merged: list[tuple[str, str]] = []
    for speaker, text in out:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + "\n" + text)
        else:
            merged.append((speaker, text))
    return merged


def say_to_aiff(text: str, out_aiff: Path, voice: str, rate: int | None) -> None:
    cmd = ["say", "-o", str(out_aiff), "-v", voice]
    if rate is not None:
        cmd += ["-r", str(rate)]
    cmd += [text]
    run(cmd)


# ---------------------------------------------------------------- Gemini TTS


def get_gemini_voice_for_speaker(name: str) -> str:
    """Resolve a speaker name to a prebuilt Gemini voice (Gemini backend)."""
    name = name.strip()
    if name in SPEAKER_GEMINI_VOICE_MAP:
        return SPEAKER_GEMINI_VOICE_MAP[name]
    parts = name.split()
    if parts and parts[0] in SPEAKER_GEMINI_VOICE_MAP:
        return SPEAKER_GEMINI_VOICE_MAP[parts[0]]
    digest = hashlib.md5(name.lower().encode("utf-8"), usedforsecurity=False).hexdigest()
    return GEMINI_FALLBACK_VOICES[int(digest, 16) % len(GEMINI_FALLBACK_VOICES)]


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, out_wav: Path) -> None:
    """Wrap raw little-endian 16-bit PCM bytes into a WAV container."""
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,  # PCM
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_chunk = struct.pack("<4sI", b"data", data_size) + pcm_bytes
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    out_wav.write_bytes(riff_header + fmt_chunk + data_chunk)


def _gemini_tts_pcm(
    client: Any, model: str, script: str, speaker_voice_pairs: list[tuple[str, str]]
) -> tuple[bytes, int]:
    """Return (PCM bytes, sample_rate) for a Gemini TTS call.

    ``speaker_voice_pairs`` is (speaker_label, gemini_voice_name). Gemini's
    multi-speaker mode requires *exactly* 2 distinct voices; with 1 voice
    the script falls back to single-speaker mode (the speaker label is
    ignored by the API).
    """
    from google.genai import types

    if len(speaker_voice_pairs) == 2:
        speaker_configs = [
            types.SpeakerVoiceConfig(
                speaker=label,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
                ),
            )
            for label, voice in speaker_voice_pairs
        ]
        speech_config = types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=speaker_configs,
            ),
        )
    elif len(speaker_voice_pairs) == 1:
        _, voice = speaker_voice_pairs[0]
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
            ),
        )
    else:
        raise ValueError(
            f"Gemini TTS expects 1 or 2 speakers per call, got {len(speaker_voice_pairs)}"
        )

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=speech_config,
    )
    response = client.models.generate_content(
        model=model,
        contents=script,
        config=config,
    )
    parts = response.candidates[0].content.parts
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            mime = str(inline.mime_type or "")
            # Mime looks like 'audio/L16;codec=pcm;rate=24000'.
            sample_rate = 24000
            m = re.search(r"rate=(\d+)", mime)
            if m:
                sample_rate = int(m.group(1))
            return inline.data, sample_rate
    raise RuntimeError(
        f"Gemini TTS returned no audio part (got: {[type(p).__name__ for p in parts]})"
    )


def _gemini_render_script(
    segments: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """Build a Gemini-friendly TTS script + the speaker→voice list.

    Returns (script_text, speaker_voice_pairs). Speakers in script appear in
    alphabetical order of their first-seen index so the model receives them
    in a stable order.
    """
    distinct: list[str] = []
    for speaker, _ in segments:
        if speaker not in distinct:
            distinct.append(speaker)
    speaker_voice_pairs = [(s, get_gemini_voice_for_speaker(s)) for s in distinct]
    body_lines: list[str] = []
    for speaker, text in segments:
        body_lines.append(f"{speaker}: {text}")
    return "\n\n".join(body_lines), speaker_voice_pairs


def render_segments_via_gemini(
    segments: list[tuple[str, str]],
    out_mp3: Path,
    bitrate: str,
    model: str,
    api_key: str,
) -> None:
    """Render the full transcript via Gemini multi-speaker TTS in chunks.

    Gemini multi-speaker mode is capped at 2 distinct voices per call as of
    2026. Transcripts with more speakers fall back to **single-speaker mode
    per segment** — slower, costlier, and the voice for the same speaker may
    drift across segments. Still functional for fixture generation; flagged
    in the comparison memo (#934).
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    distinct_speakers = {s for s, _ in segments}

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav_chunks: list[Path] = []

        if len(distinct_speakers) <= 2:
            script, speaker_voice_pairs = _gemini_render_script(segments)
            pcm_bytes, sample_rate = _gemini_tts_pcm(client, model, script, speaker_voice_pairs)
            wav_path = td_path / "full.wav"
            _pcm_to_wav(pcm_bytes, sample_rate, wav_path)
            wav_chunks.append(wav_path)
        else:
            # > 2 speakers: per-segment single-speaker rendering.
            for idx, (speaker, text) in enumerate(segments, start=1):
                voice = get_gemini_voice_for_speaker(speaker)
                pcm_bytes, sample_rate = _gemini_tts_pcm(client, model, text, [(speaker, voice)])
                wav_path = td_path / f"chunk_{idx:03d}.wav"
                _pcm_to_wav(pcm_bytes, sample_rate, wav_path)
                wav_chunks.append(wav_path)

        # Concat WAV → MP3 via ffmpeg, mirroring the say backend's path.
        playlist = td_path / "playlist.txt"
        playlist.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in wav_chunks),
            encoding="utf-8",
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-ac",
                "1",
                "-b:a",
                bitrate,
                str(out_mp3),
            ]
        )


def concat_aiff_to_mp3(aiffs: list[Path], out_mp3: Path, bitrate: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        playlist = Path(f.name)
        for p in aiffs:
            f.write(f"file '{p.as_posix()}'\n")
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-ac",
                "1",
                "-b:a",
                bitrate,
                str(out_mp3),
            ]
        )
    finally:
        playlist.unlink(missing_ok=True)


def host_for_file(stem: str) -> str:
    """Pick host name from filename prefix: ``p01_e01`` -> ``Maya``."""
    m = FILE_PREFIX_RE.match(stem)
    if not m:
        raise ValueError(f"Filename does not start with 'pXX_': {stem}")
    pid = m.group(1).lower()
    host = PODCAST_HOSTS.get(pid)
    if not host:
        raise ValueError(f"No host mapping for podcast id '{pid}' (file: {stem})")
    return host


def derive_version_from_path(txt: Path) -> str | None:
    """Detect a ``v1``/``v2``/... segment inside the input transcript's path."""
    for part in txt.resolve().parts:
        if VERSION_SEGMENT_RE.fullmatch(part):
            return part
    return None


def read_fixtures_version_file() -> str | None:
    """Read the default fixture version from tests/fixtures/FIXTURES_VERSION."""
    candidate = Path(__file__).resolve().parent.parent / "FIXTURES_VERSION"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None


def list_voices() -> int:
    if not shutil.which("say"):
        print("say not found (macOS required)", file=sys.stderr)
        return 2
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    print(result.stdout)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "transcripts",
        type=Path,
        nargs="*",
        help="Transcript files (.txt) or folder containing transcripts (searched recursively)",
    )
    ap.add_argument(
        "--output-version",
        default=None,
        help=(
            "Output version segment under tests/fixtures/audio/<version>/. "
            "Default: detect from input path; fall back to FIXTURES_VERSION."
        ),
    )
    ap.add_argument("--rate", type=int, default=145, help="Speech rate for say (default: 145)")
    ap.add_argument("--bitrate", default="64k", help="MP3 bitrate (default: 64k)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing mp3 files")
    ap.add_argument(
        "--list-voices",
        action="store_true",
        help="List available macOS 'say' voices and exit",
    )
    ap.add_argument(
        "--backend",
        choices=("say", "gemini"),
        default="say",
        help="TTS backend (default: say). 'gemini' requires GEMINI_API_KEY.",
    )
    ap.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash-preview-tts",
        help="Gemini TTS model id (default: gemini-2.5-flash-preview-tts)",
    )
    args = ap.parse_args()

    if args.list_voices:
        return list_voices()

    if not args.transcripts:
        ap.error("transcripts argument required (unless --list-voices)")

    # Collect all transcript files
    txt_files: list[Path] = []
    for path in args.transcripts:
        if path.is_file() and path.suffix == ".txt":
            txt_files.append(path)
        elif path.is_dir():
            txt_files.extend(sorted(path.rglob("*.txt")))
        else:
            print(f"Skipping (not a .txt file or directory): {path}", file=sys.stderr)

    if args.backend == "say":
        if not shutil.which("say"):
            print("say not found (macOS required)", file=sys.stderr)
            return 2
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
        return 2

    gemini_api_key: str | None = None
    if args.backend == "gemini":
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
        if not gemini_api_key:
            print("GEMINI_API_KEY required for --backend gemini", file=sys.stderr)
            return 2

    scripts_dir = Path(__file__).resolve().parent
    fixtures_dir = scripts_dir.parent

    if not txt_files:
        print("No .txt transcripts found", file=sys.stderr)
        return 1

    # Output version: per-file (input path), CLI override, then FIXTURES_VERSION.
    cli_version = args.output_version
    fallback_version = read_fixtures_version_file()

    for txt in txt_files:
        if cli_version:
            version = cli_version
        else:
            version = derive_version_from_path(txt) or fallback_version
        if not version:
            print(
                f"Skipping {txt.name}: cannot determine output version "
                "(pass --output-version or place under tests/fixtures/transcripts/<version>/)",
                file=sys.stderr,
            )
            continue

        audio_dir = fixtures_dir / "audio" / version
        audio_dir.mkdir(parents=True, exist_ok=True)
        out_mp3 = audio_dir / f"{txt.stem}.mp3"
        if out_mp3.exists() and not args.overwrite:
            continue

        raw = txt.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        try:
            host_name = host_for_file(txt.stem)
        except ValueError as e:
            print(f"Skipping {txt.name}: {e}", file=sys.stderr)
            continue

        segments = parse_segments(raw, host_name=host_name)
        if not segments:
            continue

        if args.backend == "gemini":
            assert gemini_api_key is not None  # narrowed above
            render_segments_via_gemini(
                segments,
                out_mp3,
                bitrate=args.bitrate,
                model=args.gemini_model,
                api_key=gemini_api_key,
            )
        else:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                aiffs: list[Path] = []
                for i, (speaker, text) in enumerate(segments, start=1):
                    voice = get_voice_for_speaker(speaker)
                    chunk = text.strip()
                    safe_speaker = re.sub(r"[^A-Za-z0-9_]", "_", speaker)[:24] or "spk"
                    out_aiff = td_path / f"{txt.stem}_{i:03d}_{safe_speaker}.aiff"
                    say_to_aiff(chunk, out_aiff, voice=voice, rate=args.rate)
                    aiffs.append(out_aiff)
                concat_aiff_to_mp3(aiffs, out_mp3, bitrate=args.bitrate)

        print(f"Wrote {out_mp3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
