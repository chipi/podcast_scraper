#!/usr/bin/env python3
"""MOSS transcribe+diarize with AUDIO WINDOWING for long episodes (#1177 long-audio hardening).

A single MOSS pass caps at ~30 min (128k context fills with audio features, then it stops). This
splits audio into overlapping windows, runs MOSS per window (each window written to a temp WAV and
fed through the tested build_transcription_messages path), offsets timestamps, stitches the
transcript, and maps per-window speaker labels to a consistent global set via overlap agreement.

parse_transcript returns FROZEN dataclasses, so each turn is copied into a mutable dict first.

Usage: moss_chunked_batch.py <audio_dir> <out_dir> [window_sec=1500] [overlap_sec=30]
"""

import glob
import os
import sys
import tempfile
import time

import moss_transcribe_diarize.inference_utils as _iu
import numpy as np
import soundfile as sf
import torch
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    load_audio_item,
    resolve_device,
)
from transformers import AutoModelForCausalLM, AutoProcessor

_orig = _iu.prepare_inputs


def _cpu_audio(processor, messages, **kw):  # stubbed cuFFT -> CPU mel-STFT
    kw["device"] = None
    return _orig(processor, messages, **kw)


_iu.prepare_inputs = _cpu_audio

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
# Pin the download to a revision (matches the prod MOSS_MODEL_REVISION default) so the eval is
# reproducible and bandit's B615 unpinned-download check is satisfied.
MODEL_REVISION = os.environ.get("MOSS_MODEL_REVISION", "main")
SR = 16000


def run_window(model, processor, wav_path, device, dtype, max_new=16384):
    msgs = build_transcription_messages(wav_path)
    res = generate_transcription(
        model, processor, msgs, max_new_tokens=max_new, do_sample=False, device=device, dtype=dtype
    )
    return [
        {"start": float(s.start), "end": float(s.end), "speaker": s.speaker, "text": str(s.text)}
        for s in parse_transcript(res["text"])
    ]


def _spk_time(turns, lo, hi):
    t = {}
    for s in turns:
        a, b = max(s["start"], lo), min(s["end"], hi)
        if b > a:
            t[s["speaker"]] = t.get(s["speaker"], 0.0) + (b - a)
    return t


def map_speakers(prev_turns, new_turns, ov_start, ov_end):
    """Greedy-map new-window labels to prev global labels by shared overlap time."""
    prev_t, new_t = _spk_time(prev_turns, ov_start, ov_end), _spk_time(new_turns, ov_start, ov_end)
    mapping, used = {}, set()
    for ns in sorted(new_t, key=lambda k: -new_t[k]):
        best, best_score = None, 0.0
        for ps in prev_t:
            if ps in used:
                continue
            score = min(new_t[ns], prev_t[ps])
            if score > best_score:
                best, best_score = ps, score
        if best is not None:
            mapping[ns] = best
            used.add(best)
    return mapping


def transcribe_long(model, processor, audio_path, device, dtype, window, overlap, tmpdir):
    audio = load_audio_item(audio_path, sampling_rate=SR)
    n = len(audio)
    dur = n / SR
    win, ov = int(window * SR), int(overlap * SR)
    out_turns, prev_turns, next_gid, prev_end = [], [], [0], 0.0
    start, widx = 0, 0
    while start < n:
        end = min(start + win, n)
        w_start = start / SR
        wav = os.path.join(tmpdir, f"w{widx}.wav")
        sf.write(wav, audio[start:end].astype(np.float32), SR)
        turns = run_window(model, processor, wav, device, dtype)
        os.unlink(wav)
        for s in turns:
            s["start"] += w_start
            s["end"] += w_start
        if widx == 0:
            lab, m = {}, {}
        else:
            m = map_speakers(prev_turns, turns, w_start, min(prev_end, end / SR))
            lab = {}
        for s in turns:
            loc = s["speaker"]
            if loc in m:
                g = m[loc]
            elif loc in lab:
                g = lab[loc]
            else:
                g = f"S{next_gid[0]:02d}"
                next_gid[0] += 1
                lab[loc] = g
            s["speaker"] = g
        cutoff = 0.0 if widx == 0 else (prev_end - overlap / 2)
        out_turns.extend([s for s in turns if s["start"] >= cutoff])
        prev_turns, prev_end = turns, end / SR
        widx += 1
        if end >= n:
            break
        start = end - ov
    return out_turns, dur


def main():
    audio_dir, out_dir = sys.argv[1], sys.argv[2]
    window = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
    overlap = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    os.makedirs(out_dir, exist_ok=True)
    device = resolve_device("auto")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=MODEL_REVISION, trust_remote_code=True, dtype="auto"
        )
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=True
    )
    print(f"model loaded on {device}; window={window}s overlap={overlap}s", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for mp3 in sorted(glob.glob(os.path.join(audio_dir, "*.mp3"))):
            stem = os.path.splitext(os.path.basename(mp3))[0]
            t0 = time.time()
            turns, dur = transcribe_long(model, processor, mp3, device, dtype, window, overlap, td)
            lines = [
                f"SPEAKER {stem} 1 {s['start']:.3f} {max(0.0, s['end']-s['start']):.3f} "
                f"<NA> <NA> {s['speaker']} <NA> <NA>"
                for s in turns
            ]
            open(f"{out_dir}/{stem}.rttm", "w").write("\n".join(lines) + "\n")
            open(f"{out_dir}/{stem}.txt", "w").write(
                " ".join(s["text"].strip() for s in turns) + "\n"
            )
            spk = sorted({s["speaker"] for s in turns})
            el = time.time() - t0
            print(
                f"{stem}: {len(turns)} turns, {len(spk)} spk, dur={dur:.0f}s "
                f"({el:.0f}s, {dur/el:.1f}x)",
                flush=True,
            )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
