#!/usr/bin/env python3
"""
macOS-only: Generate MP3 files from .txt transcripts using 'say' with two voices:
- Host voice for host lines
- Guest voice for guest (and any non-host speaker) lines

Input:  folder containing *.txt (recursively)
Output: fixtures/audio/<basename>.mp3 (audio folder in fixtures/)

Host detection (Option B):
- Expects transcript filenames like: p02_e03.txt
- Uses the pXX prefix to pick the host name from a fixed mapping below.

Requires: /usr/bin/say, ffmpeg
"""

from __future__ import annotations

import argparse
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
}

TS_RE = re.compile(r"^\[\s*\d{1,2}:\d{2}(:\d{2})?\s*\]$")
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z '\-]{0,40}):\s+(.*)$")
FILE_PREFIX_RE = re.compile(r"^(p\d{2})_", re.IGNORECASE)


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def parse_segments(raw: str, host_name: str) -> list[tuple[str, str]]:
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
            speaker_type = "host" if speaker.casefold() == host_name.casefold() else "guest"
            out.append((speaker_type, text))
        else:
            out.append(("host", s))

    # Merge consecutive segments of same type to reduce voice switches
    merged: list[tuple[str, str]] = []
    for typ, text in out:
        if merged and merged[-1][0] == typ:
            merged[-1] = (typ, merged[-1][1] + "\n" + text)
        else:
            merged.append((typ, text))
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
    """
    Determine host name based on filename prefix:
      p01_e01 -> p01 -> Maya
    """
    m = FILE_PREFIX_RE.match(stem)
    if not m:
        raise ValueError(f"Filename does not start with 'pXX_': {stem}")
    pid = m.group(1).lower()
    host = PODCAST_HOSTS.get(pid)
    if not host:
        raise ValueError(f"No host mapping for podcast id '{pid}' (file: {stem})")
    return host


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "transcripts_dir",
        type=Path,
        help="Folder containing .txt transcripts (searched recursively)",
    )
    ap.add_argument(
        "--host-voice", default="Samantha", help="macOS 'say' voice for host (default: Samantha)"
    )
    ap.add_argument(
        "--guest-voice", default="Daniel", help="macOS 'say' voice for guest (default: Daniel)"
    )
    ap.add_argument("--rate", type=int, default=145, help="Speech rate for say (default: 145)")
    ap.add_argument("--bitrate", default="64k", help="MP3 bitrate (default: 64k)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing mp3 files")
    args = ap.parse_args()

    if not args.transcripts_dir.is_dir():
        print(f"Not a directory: {args.transcripts_dir}", file=sys.stderr)
        return 2

    if not shutil.which("say"):
        print("say not found (macOS required)", file=sys.stderr)
        return 2
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
        return 2

    scripts_dir = Path(__file__).resolve().parent
    audio_dir = scripts_dir.parent / "fixtures" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.transcripts_dir.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt transcripts found under: {args.transcripts_dir}", file=sys.stderr)
        return 1

    for txt in txt_files:
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

            for i, (typ, text) in enumerate(segments, start=1):
                voice = args.host_voice if typ == "host" else args.guest_voice
                chunk = text.strip()
                if i != 1:
                    chunk = "...\n" + chunk
                out_aiff = td_path / f"{txt.stem}_{i:03d}_{typ}.aiff"
                say_to_aiff(chunk, out_aiff, voice=voice, rate=args.rate)
                aiffs.append(out_aiff)

            concat_aiff_to_mp3(aiffs, out_mp3, bitrate=args.bitrate)

        print(f"Wrote {out_mp3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
