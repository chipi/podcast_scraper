#!/usr/bin/env python3
"""
macOS-only: Generate MP3 files from .txt transcripts using the system 'say'
TTS engine with a distinct voice per speaker.

Per-speaker voice mapping (RFC-059 §2 / issue #111) replaces the prior
binary host/guest scheme so diarization tests (RFC-058) can actually
distinguish speakers in fixture audio.

Voice resolution order:
1. Exact name match in SPEAKER_VOICE_MAP
2. First-word match (e.g. "Alex Morgan" -> "Alex")
3. Stable hash-based fallback (md5; deterministic across runs)

Output version is derived from the input transcript's path
(``.../transcripts/v2/...`` -> ``.../audio/v2/...``); explicit
``--output-version`` overrides; otherwise FIXTURES_VERSION is read from
``tests/fixtures/FIXTURES_VERSION``.

Requires: /usr/bin/say, ffmpeg
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
    "Liam": "Fred",  # en_US male
    "Sophie": "Flo",  # en_GB female
    "Noah": "Tom",  # en_US male
    "Priya": "Isha",  # en_IN female
    "Jonas": "Eddy",  # en_US male
    "Camila": "Paulina",  # es_MX female
    "Marco": "Luca",  # it_IT male
    "Hanna": "Anna",  # de_DE female
    "Owen": "Reed",  # en_US male
    "Ava": "Kathy",  # en_US female
    "Tariq": "Rishi",  # en_IN male
    "Elise": "Amelie",  # fr_CA female
    "Daniel": "Oliver",  # en_GB male
    "Isabel": "Monica",  # es_ES female
    "Kasper": "Ralph",  # en_US male
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

TS_RE = re.compile(r"^\[\s*\d{1,2}:\d{2}(:\d{2})?\s*\]$")
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z '\-]{0,40}):\s+(.*)$")
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
    # Stable hash (Python's built-in hash() varies across runs)
    digest = hashlib.md5(name.lower().encode("utf-8")).hexdigest()
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

    if not shutil.which("say"):
        print("say not found (macOS required)", file=sys.stderr)
        return 2
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
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
