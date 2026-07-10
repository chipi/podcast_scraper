#!/usr/bin/env python3
"""Diarization tuning sweep — find pyannote params that match the v3 ground-truth voice counts.

Reads each ``tests/fixtures/transcripts/v3/<name>.groundtruth.json`` for its
``expected_diarized_voices`` (humans + ad voices), then sweeps a grid of
``clustering_threshold x max_speakers`` on the v3 audio, scoring ``detected == expected`` per
fixture. Reports a per-combo count-match rate + which fixtures/params miss, and the best combo.

Runs the in-process ``PyAnnoteDiarizationProvider`` (which now honours ``clustering_threshold``),
so it goes fast on a GPU box (``--device cuda`` on the DGX) and works — slowly — on CPU/MPS. The
default model is the non-gated ``speaker-diarization-community-1`` so no HF token is needed.

    # validate wiring without any diarization (GPU-free):
    python scripts/eval/score/diarization_tuning_sweep_v1.py --dry-run

    # real sweep on the DGX GPU, all fixtures, write results:
    python scripts/eval/score/diarization_tuning_sweep_v1.py --device cuda \\
        --out data/eval/runs/diarization_tuning_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GT_DIR = os.path.join(REPO, "tests", "fixtures", "transcripts", "v3")
AUDIO_DIR = os.path.join(REPO, "tests", "fixtures", "audio", "v3")
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
# Threshold: None = model default; higher merges more -> fewer speakers. max_speakers bounds it.
DEFAULT_THRESHOLDS = [None, 0.5, 0.6, 0.7, 0.8]
DEFAULT_MAX_SPEAKERS = [4, 6, 20]


def load_fixtures(only: Optional[list[str]]) -> list[dict]:
    out = []
    for gt in sorted(glob.glob(os.path.join(GT_DIR, "*.groundtruth.json"))):
        d = json.load(open(gt, encoding="utf-8"))
        name = d["fixture"]
        if only and name not in only:
            continue
        audio = os.path.join(AUDIO_DIR, f"{name}.mp3")
        if not os.path.exists(audio):
            continue
        out.append(
            {
                "name": name,
                "audio": audio,
                "expected": d["expected_diarized_voices"],
                "type": d["type"],
                "has_commercial": d["has_commercial"],
            }
        )
    return out


def sweep(fixtures, thresholds, max_speakers_list, *, device, model):
    from podcast_scraper.providers.ml.diarization.pyannote_provider import (
        PyAnnoteDiarizationProvider,
    )

    results = []
    for thr in thresholds:
        provider = PyAnnoteDiarizationProvider(
            hf_token=os.environ.get("HF_TOKEN", ""),
            device=device,
            model_name=model,
            clustering_threshold=thr,
        )
        for max_sp in max_speakers_list:
            per_fixture, matches = [], 0
            for fx in fixtures:
                t = time.time()
                res = provider.diarize(fx["audio"], min_speakers=1, max_speakers=max_sp)
                detected = res.num_speakers
                ok = detected == fx["expected"]
                matches += ok
                per_fixture.append(
                    {
                        **{k: fx[k] for k in ("name", "expected", "type")},
                        "detected": detected,
                        "match": ok,
                        "sec": round(time.time() - t, 1),
                    }
                )
            results.append(
                {
                    "clustering_threshold": thr,
                    "max_speakers": max_sp,
                    "match_rate": round(matches / len(fixtures), 3),
                    "matched": matches,
                    "total": len(fixtures),
                    "fixtures": per_fixture,
                }
            )
            print(
                f"  thr={str(thr):5s} max={max_sp:2d}: {matches}/{len(fixtures)} match "
                f"({100 * matches / len(fixtures):.0f}%)",
                flush=True,
            )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="clustering thresholds (default sweep incl. model default)",
    )
    ap.add_argument("--max-speakers", type=int, nargs="*", default=DEFAULT_MAX_SPEAKERS)
    ap.add_argument("--only", nargs="*", help="restrict to these fixture names")
    ap.add_argument("--out", help="dir to write results.json")
    ap.add_argument("--dry-run", action="store_true", help="validate fixtures/grid, no diarization")
    args = ap.parse_args()

    fixtures = load_fixtures(args.only)
    thresholds = args.thresholds if args.thresholds is not None else DEFAULT_THRESHOLDS
    print(
        f"fixtures: {len(fixtures)}  (panels: "
        f"{sum(f['type'] == 'panel' for f in fixtures)}, "
        f"commercial: {sum(f['has_commercial'] for f in fixtures)})"
    )
    print(
        f"grid: thresholds={thresholds} x max_speakers={args.max_speakers} "
        f"= {len(thresholds) * len(args.max_speakers)} combos x {len(fixtures)} fixtures"
    )

    if args.dry_run:
        print("dry-run: fixtures + expected counts loaded, grid built; skipping diarization.")
        for f in fixtures[:8]:
            print(f"  {f['name']:18s} expected={f['expected']} type={f['type']}")
        return 0

    results = sweep(fixtures, thresholds, args.max_speakers, device=args.device, model=args.model)
    best = max(results, key=lambda r: r["match_rate"])
    print(
        f"\nBEST: thr={best['clustering_threshold']} max_speakers={best['max_speakers']} "
        f"-> {best['matched']}/{best['total']} ({100 * best['match_rate']:.0f}%)"
    )
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "results": results, "best": best}, fh, indent=2)
        print(f"wrote {os.path.join(args.out, 'results.json')}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
