#!/usr/bin/env python3
"""Head-to-head transcription WER: MOSS vs whisper-large-v3, ref = source .txt (#1174)."""

import glob
import os
import pathlib
import re
import sys

import jiwer

REPO = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{REPO}/tests/fixtures/scripts")
from transcripts_to_mp3 import host_for_file, parse_segments  # noqa: E402

REF_DIR = f"{REPO}/tests/fixtures/transcripts/v3"
MOSS_DIR = os.environ.get("MOSS_DIR", "/tmp/moss_out")
WHISPER_DIR = os.environ.get("WHISPER_DIR", "/tmp/whisper_out")

_PUNCT = re.compile(r"[^0-9a-z\s]")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def reference_text(stem: str) -> str:
    raw = open(f"{REF_DIR}/{stem}.txt").read().strip()
    turns = parse_segments(raw, host_name=host_for_file(stem))
    return " ".join(t for _spk, t in turns)


def wer(ref: str, hyp: str):
    ref_n, hyp_n = _norm(ref), _norm(hyp)
    if not hyp_n:
        return 1.0
    return jiwer.wer(ref_n, hyp_n)


def main():
    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{MOSS_DIR}/*.txt"))

    def _read(d, stem):
        p = f"{d}/{stem}.txt"
        return open(p).read() if os.path.exists(p) else ""

    rows, mw, ww = [], [], []
    for stem in stems:
        ref = reference_text(stem)
        moss, whi = _read(MOSS_DIR, stem), _read(WHISPER_DIR, stem)
        mwer, wwer = wer(ref, moss), wer(ref, whi)
        mw.append(mwer)
        ww.append(wwer)
        rows.append((stem, mwer * 100, wwer * 100))
    rows.sort(key=lambda r: -(r[1] - r[2]))
    print(f"{'fixture':16s} {'MOSS%':>7} {'whisp%':>7} {'delta':>7}")
    for stem, m, w in rows:
        print(f"{stem:16s} {m:7.1f} {w:7.1f} {m-w:+7.1f}")
    n = len(rows)
    print(f"\n=== transcription WER on {n} fixtures (ref = source text) ===")
    print(f"MOSS    mean WER: {sum(mw)/n*100:5.1f}%   median: {sorted(mw)[n//2]*100:.1f}%")
    print(f"whisper mean WER: {sum(ww)/n*100:5.1f}%   median: {sorted(ww)[n//2]*100:.1f}%")
    better = sum(1 for m, w in zip(mw, ww) if m < w)
    print(f"MOSS beats whisper on {better}/{n} fixtures")


if __name__ == "__main__":
    main()
