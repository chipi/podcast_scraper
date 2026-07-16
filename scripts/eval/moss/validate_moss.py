#!/usr/bin/env python3
"""Validate MOSS-Transcribe-Diarize on GB10: load, run one clip, print segments + timing."""

import os
import sys
import time

import torch
import transformers

print(
    f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
    f"transformers {transformers.__version__}",
    flush=True,
)

from moss_transcribe_diarize import parse_transcript  # noqa: E402
from moss_transcribe_diarize.inference_utils import (  # noqa: E402
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)
from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402

MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
audio = sys.argv[1]
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

# The NGC vLLM image ships a STUB cuFFT, so whisper's GPU mel-spectrogram (torch.stft)
# fails with "cuFFT error 50". Force feature extraction onto CPU (device=None → the moss
# processor skips the cuda audio_kwargs); STFT on CPU is cheap even for long audio, and the
# model still runs on GPU.
import moss_transcribe_diarize.inference_utils as _iu  # noqa: E402

_orig_prepare = _iu.prepare_inputs


def _cpu_audio_prepare(processor, messages, **kw):
    kw["device"] = None
    return _orig_prepare(processor, messages, **kw)


_iu.prepare_inputs = _cpu_audio_prepare

device = resolve_device("auto")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
print(f"device={device} dtype={dtype} loading {MODEL} ...", flush=True)

t0 = time.time()
model = (
    AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype="auto")
    .to(dtype=dtype)
    .to(device)
    .eval()
)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print(
    f"loaded in {time.time()-t0:.1f}s  " f"sr={processor.feature_extractor.sampling_rate}",
    flush=True,
)

# audio duration for realtime factor
import soundfile as sf  # noqa: E402

info = sf.info(audio)
dur = info.frames / info.samplerate
print(f"audio {audio}  dur={dur:.1f}s", flush=True)

messages = build_transcription_messages(audio)
t1 = time.time()
result = generate_transcription(
    model,
    processor,
    messages,
    max_new_tokens=max_new,
    do_sample=False,
    device=device,
    dtype=dtype,
)
elapsed = time.time() - t1
print(f"\n=== INFER {elapsed:.1f}s  ({dur/elapsed:.1f}x realtime) ===", flush=True)

segs = list(parse_transcript(result["text"]))
print(f"segments={len(segs)}  speakers={sorted({s.speaker for s in segs})}")
print("--- first 12 segments ---")
for s in segs[:12]:
    print(f"  [{s.start:7.2f}-{s.end:7.2f}] {s.speaker}: {s.text[:80]}")
print("--- raw head ---")
print(result["text"][:400])
