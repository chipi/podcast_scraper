#!/usr/bin/env python3
"""#1273: is large-v3's last-place WER a serving artifact (int8 on the DGX) or real?

Re-transcribes the cached human-GT bake-off episodes with large-v3 two ways and scores WER against
the human transcript with the SAME normalization as the original bake-off (so the numbers are
directly comparable to the 16.3%):

  - int8    : the live DGX speaches service (default :8000, WHISPER__COMPUTE_TYPE=int8)
  - float16 : the parallel experiment service (default :8005, from
              infra/dgx/experiments/docker-compose.1273-largev3-fp16.yml)

turbo @ int8 is also run as an anchor — it should reproduce ~13.5% and confirms the harness is sane.

Reads the ALREADY-CACHED audio + ground truth from the eval-data repo (no re-fetch, byte-identical
speech_optimal_v1 audio). Run from this repo with the package importable (`.venv/bin/python`).

  python scripts/investigate/1273_largev3_compute_wer.py \
    --eval-data-root ../podcast-scraper-eval-data --lan 192.168.1.111 --episodes 10
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from podcast_scraper.config import Config
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)

# Exact normalization from the bake-off harness (run_asr_bakeoff.py) so WER is comparable.
_NORM = re.compile(r"[^a-z0-9']+")
_LABEL = re.compile(r"^[A-Z][A-Za-z .'-]{0,40}:", re.M)
_TS = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")

LARGE_V3 = "Systran/faster-whisper-large-v3"
TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"


def norm_words(t: str) -> list[str]:
    t = _LABEL.sub(" ", t)
    t = _TS.sub(" ", t)
    return [w for w in _NORM.sub(" ", t.lower()).split() if w]


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    return (Levenshtein.distance(r, h) / len(r)) if r else 0.0


def transcribe(model: str, port: int, lan: str, audio: str) -> str:
    cfg = Config.model_validate(
        {
            "rss_url": "https://x/f.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "dgx_whisper_model": model,
            "dgx_whisper_port": port,
            "dgx_tailnet_host": lan,
            "transcription_fallback_providers": ["whisper"],
            "dgx_request_timeout_sec": 3600.0,
        }
    )
    p = TailnetDgxWhisperTranscriptionProvider(cfg)
    p.initialize()
    res, _ = p.transcribe_with_segments(audio)
    return str(res.get("text", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-data-root", default="../podcast-scraper-eval-data")
    ap.add_argument("--lan", default="192.168.1.111", help="DGX LAN IP the harness reaches it by")
    ap.add_argument("--int8-port", type=int, default=8000)
    ap.add_argument("--fp16-port", type=int, default=8005)
    ap.add_argument("--episodes", type=int, default=0, help="cap episode count (0 = all)")
    ap.add_argument("--dataset", default="data/eval/datasets/asr_human_gt_v1.episodes.json")
    args = ap.parse_args()

    root = Path(args.eval_data_root)
    eps = json.loads((root / args.dataset).read_text())
    if args.episodes:
        eps = eps[: args.episodes]

    # (label, model, port) — turbo anchor first (fast) so a harness/serving fault surfaces early.
    arms = [
        ("turbo@int8 (anchor)", TURBO, args.int8_port),
        ("large-v3@int8", LARGE_V3, args.int8_port),
        ("large-v3@float16", LARGE_V3, args.fp16_port),
    ]
    results: dict[str, dict] = {label: {} for label, _, _ in arms}

    for label, model, port in arms:
        for e in eps:
            eid = e["id"]
            audio = root / "cache" / "audio" / f"{eid}.pre.mp3"
            gt = root / "cache" / "transcripts" / f"{eid}.txt"
            if not audio.exists() or not gt.exists():
                results[label][eid] = {"error": "cache miss (audio or GT not on disk)"}
                print(f"[{label}] {eid} SKIP — cache miss", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                text = transcribe(model, port, args.lan, str(audio))
                wall = time.perf_counter() - t0
                w = wer(gt.read_text(), text)
                results[label][eid] = {"wer": w, "wall_s": round(wall, 1)}
                print(f"[{label}] {eid} WER={w:.1%} wall={wall:.0f}s", flush=True)
            except Exception as ex:  # noqa: BLE001
                results[label][eid] = {"error": f"{type(ex).__name__}: {str(ex)[:140]}"}
                print(f"[{label}] {eid} FAILED: {type(ex).__name__}: {str(ex)[:140]}", flush=True)

    print("\n==== SUMMARY (mean WER over episodes with a score) ====")
    summary = {}
    for label in results:
        ws = [r["wer"] for r in results[label].values() if "wer" in r]
        mean = statistics.mean(ws) if ws else None
        summary[label] = {"mean_wer": mean, "n": len(ws)}
        if ws:
            print(f"  {label:24s} mean_WER={mean:.1%} (n={len(ws)})")
        else:
            print(f"  {label:24s} no scores")

    out = Path("docs/wip/EVAL_1273_largev3_int8_vs_fp16.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_episode": results}, indent=2))
    print(f"\nwrote {out}")
    # Verdict hint
    i8 = summary.get("large-v3@int8", {}).get("mean_wer")
    fp = summary.get("large-v3@float16", {}).get("mean_wer")
    if i8 is not None and fp is not None:
        delta = i8 - fp
        print(
            f"\nVERDICT: int8={i8:.1%} vs float16={fp:.1%} (Δ={delta:+.1%}). "
            + (
                "float16 materially better → int8 SERVING regression confirmed."
                if delta > 0.02
                else "no material gain → large-v3 genuinely worst; MOSS-failover was right."
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
