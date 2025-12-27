#!/usr/bin/env python3
"""
Generate MP3 files from .txt transcripts.

- Input: folder containing *.txt
- Output: audio/*.mp3 (audio folder next to scripts/)
"""

from __future__ import annotations

import argparse
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
        aiff = wav_path.with_suffix(".aiff")
        cmd = ["say", "-o", str(aiff)]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(rate)]
        cmd += [text]
        run(cmd)
        run(["ffmpeg", "-y", "-i", str(aiff), str(wav_path)])
        aiff.unlink(missing_ok=True)
        return

    if engine == "piper":
        if not piper_model:
            raise SystemExit("piper requires --piper-model")
        p = subprocess.run(
            ["piper", "--model", piper_model, "--output_file", str(wav_path)],
            input=text,
            text=True,
        )
        if p.returncode != 0:
            raise RuntimeError("piper failed")
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
    ap.add_argument("transcripts_dir", type=Path)
    ap.add_argument("--prefer", choices=["say", "piper", "espeak-ng"])
    ap.add_argument("--voice")
    ap.add_argument("--rate", type=int)
    ap.add_argument("--bitrate", default="64k")
    ap.add_argument("--piper-model")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
        return 2

    engine = detect_tts(args.prefer)

    # audio/ folder next to scripts/
    scripts_dir = Path(__file__).resolve().parent
    audio_dir = scripts_dir.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for txt in sorted(args.transcripts_dir.glob("*.txt")):
        mp3 = audio_dir / f"{txt.stem}.mp3"
        if mp3.exists() and not args.overwrite:
            continue

        text = txt.read_text(encoding="utf-8").strip()
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
