"""Dump per-turn diarization segments for every fixture (#1170 DER run).

DIAR_MODEL selects the model; OUT_DIR (default segments_<tag>) receives one
segments_<episode>.json = {"segments":[{start,end,speaker}]} per audio file, to be
scored locally against the exact RTTM ground truth (diarization_der_rttm_v1.py).
Run once per model (3.1 on the base image, community-1 on a pyannote-4 install).
Boilerplate (torch.load weights_only + version-string normalize) mirrors
score44_seg.py so both models load identically.
"""

# isort: skip_file  -- the torch.load weights_only patch MUST run between
# `import torch` and `from pyannote.audio import Pipeline`; do not reorder.

import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")
import torch  # noqa: E402

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
import re as _re  # noqa: E402

_m = _re.match(r"(\d+\.\d+\.\d+)", torch.__version__)
if _m:
    torch.__version__ = _m.group(1)  # type: ignore[assignment]  # pyannote version-string shim
from pyannote.audio import Pipeline  # noqa: E402
import torchaudio  # noqa: E402

WORK = "/work"
MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1")
TAG = "v4" if "community" in MODEL else "31"
OUT = os.path.join(WORK, os.environ.get("OUT_DIR", f"segments_{TAG}"))
os.makedirs(OUT, exist_ok=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — run this on the DGX GPU (docker run --gpus all).")
DEV = torch.device("cuda")

audio = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{WORK}/audio/*.mp3")}
fixtures = sorted(audio)
print(
    f"model={MODEL} fixtures={len(fixtures)} out={OUT} cuda={torch.cuda.is_available()}", flush=True
)

pipe = Pipeline.from_pretrained(MODEL)
if pipe is None:
    raise SystemExit(f"failed to load pipeline {MODEL} (check the mounted model cache + HF token).")
pipe = pipe.to(DEV)

for k in fixtures:
    w, sr = torchaudio.load(audio[k])
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    ann = pipe({"waveform": w.to(DEV), "sample_rate": sr})
    ann = getattr(ann, "speaker_diarization", ann)
    segs = [
        {"start": float(s.start), "end": float(s.end), "speaker": str(lab)}
        for s, _, lab in ann.itertracks(yield_label=True)
    ]
    with open(os.path.join(OUT, f"segments_{k}.json"), "w") as fh:
        json.dump({"model": MODEL, "segments": segs}, fh)
    nspk = len({x["speaker"] for x in segs})
    print(f"  {k}: {len(segs)} segments, {nspk} speakers", flush=True)

print("done", flush=True)
