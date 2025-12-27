#!/usr/bin/env python3
"""
Generate MP3 files from .txt transcripts.

- Input: folder containing *.txt (recursively)
- Output: audio/*.mp3 (audio folder next to scripts/)
- Sanitizes transcript text for TTS:
  - removes timestamps like [00:00]
  - removes headers/metadata lines (Podcast:, Episode:, Host:, Guest:, markdown headings)
  - removes stage directions like *Short break.*
  - strips speaker labels "Name:" and inserts a short pause instead

- Usage: python3 tests/scripts/transcripts_to_mp3.py tests/fixtures/transcripts \
  --prefer say \
  --voice Alex \
  --rate 175 \
  --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def detect_tts(prefer: str | None) -> str:
    engines = ["say", "piper", "espeak-ng"] if prefer is None else [prefer]
    for e in engines:
        if shutil.which(e):
            return e
    raise SystemExit("No supported TTS engine found (say | piper | espeak-ng).")


def prepare_for_tts(raw: str) -> str:
    """
    Convert transcript-ish text into something that sounds natural in TTS.
    """
    lines = raw.splitlines()
    out: list[str] = []

    ts_re = re.compile(r"^\[\s*\d{1,2}:\d{2}(:\d{2})?\s*\]$")
    speaker_re = re.compile(r"^[A-Za-z][A-Za-z '\-]{0,40}:\s+(.*)$")
    stage_dir_re = re.compile(r"^\*.*\*$")

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Drop timestamps like [00:00], [12:34], [1:02:03]
        if ts_re.fullmatch(s):
            continue

        # Drop markdown headings and common metadata lines
        low = s.lower()
        if s.startswith("#") or low.startswith(
            ("podcast:", "episode:", "host:", "guest:", "title:")
        ):
            continue

        # Drop stage directions like *Short break.*
        if stage_dir_re.fullmatch(s):
            continue

        # Convert speaker labels into a pause + the utterance
        m = speaker_re.match(s)
        if m:
            utterance = m.group(1).strip()
            if not utterance:
                continue
            out.append("...")  # short pause between speakers
            out.append(utterance)
            continue

        out.append(s)

    return "\n".join(out).strip()


def tts_to_wav(
    engine: str,
    text: str,
    wav_path: Path,
    *,
    voice: str | None,
    rate: int | None,
    piper_model: str | None,
) -> None:
    if engine == "say":
        # say outputs AIFF; convert to WAV for consistent ffmpeg behavior
        aiff = wav_path.with_suffix(".aiff")
        cmd = ["say", "-o", str(aiff)]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(rate)]
        cmd += [text]
        run(cmd)
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(aiff), str(wav_path)])
        aiff.unlink(missing_ok=True)
        return

    if engine == "piper":
        if not piper_model:
            raise SystemExit("piper requires --piper-model")
        p = subprocess.run(
            ["piper", "--model", piper_model, "--output_file", str(wav_path)],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if p.returncode != 0:
            raise RuntimeError(f"piper failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
        return

    if engine == "espeak-ng":
        cmd = ["espeak-ng", "-w", str(wav_path)]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-s", str(rate)]
        cmd += [text]
        run(cmd)
        return

    raise SystemExit(f"Unsupported engine: {engine}")


def wav_to_mp3(wav: Path, mp3: Path, bitrate: str) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav),
            "-ac",
            "1",
            "-b:a",
            bitrate,
            str(mp3),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "transcripts_dir",
        type=Path,
        help="Folder containing .txt transcripts (searched recursively)",
    )
    ap.add_argument(
        "--prefer", choices=["say", "piper", "espeak-ng"], help="Force a specific TTS engine"
    )
    ap.add_argument("--voice", help="Voice name (say/espeak-ng; ignored by piper)")
    ap.add_argument("--rate", type=int, help="Speech rate (say: -r, espeak-ng: -s)")
    ap.add_argument("--bitrate", default="64k", help="MP3 bitrate (default: 64k)")
    ap.add_argument("--piper-model", help="Path to piper .onnx model (required if using piper)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing mp3 files")
    args = ap.parse_args()

    if not args.transcripts_dir.is_dir():
        print(f"Not a directory: {args.transcripts_dir}", file=sys.stderr)
        return 2

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
        return 2

    engine = detect_tts(args.prefer)

    # audio/ folder next to scripts/
    scripts_dir = Path(__file__).resolve().parent
    audio_dir = scripts_dir.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.transcripts_dir.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt transcripts found under: {args.transcripts_dir}", file=sys.stderr)
        return 1

    for txt in txt_files:
        mp3 = audio_dir / f"{txt.stem}.mp3"
        if mp3.exists() and not args.overwrite:
            continue

        raw = txt.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        text = prepare_for_tts(raw)
        if not text:
            continue

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / f"{txt.stem}.wav"
            tts_to_wav(
                engine,
                text,
                wav,
                voice=args.voice,
                rate=args.rate,
                piper_model=args.piper_model,
            )
            wav_to_mp3(wav, mp3, args.bitrate)

        print(f"Wrote {mp3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
