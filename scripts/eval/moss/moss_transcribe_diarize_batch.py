#!/usr/bin/env python3
"""Run MOSS on a dir of fixture mp3s, emit one hypothesis RTTM each (#1174 diarization eval)."""

import glob
import os
import sys
import time

import moss_transcribe_diarize.inference_utils as _iu
import torch
import transformers  # noqa: F401
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)
from transformers import AutoModelForCausalLM, AutoProcessor

# Stubbed cuFFT in the NGC vLLM image → force whisper mel-STFT to CPU (see validate.py).
_orig_prepare = _iu.prepare_inputs


def _cpu_audio_prepare(processor, messages, **kw):
    kw["device"] = None
    return _orig_prepare(processor, messages, **kw)


_iu.prepare_inputs = _cpu_audio_prepare

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_dir = sys.argv[1]
out_dir = sys.argv[2]
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
os.makedirs(out_dir, exist_ok=True)

device = resolve_device("auto")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = (
    AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype="auto")
    .to(dtype=dtype)
    .to(device)
    .eval()
)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print(f"model loaded on {device}", flush=True)

mp3s = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
for mp3 in mp3s:
    stem = os.path.splitext(os.path.basename(mp3))[0]
    t0 = time.time()
    messages = build_transcription_messages(mp3)
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=max_new,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    segs = list(parse_transcript(result["text"]))
    lines = []
    for s in segs:
        dur = max(0.0, float(s.end) - float(s.start))
        lines.append(
            f"SPEAKER {stem} 1 {float(s.start):.3f} {dur:.3f} " f"<NA> <NA> {s.speaker} <NA> <NA>"
        )
    with open(os.path.join(out_dir, f"{stem}.rttm"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # plain transcript text (for WER vs the source .txt) — segments joined in order.
    with open(os.path.join(out_dir, f"{stem}.txt"), "w") as fh:
        fh.write(" ".join(str(s.text).strip() for s in segs) + "\n")
    spk = sorted({s.speaker for s in segs})
    print(f"{stem}: {len(segs)} segs, {len(spk)} spk {spk} ({time.time()-t0:.1f}s)", flush=True)

print("DONE", flush=True)
