"""Diarization championship — pyannote/DGX vs pyannote/local CPU (#930).

Compares two diarization backends on the v2 audio fixtures:

- **pyannote/DGX**: the GPU-hosted FastAPI wrapper on the DGX (#926).
  Talked to via ``http://<dgx-host>:8001/v1/diarize``.
- **pyannote/local**: ``PyAnnoteDiarizationProvider`` (PyTorch CPU).

Gemini speech speaker-detection was a third candidate per the #930 issue
but the codebase doesn't currently have a Gemini speech provider —
that's wired separately and out of scope for this first cut. See
follow-up issue for adding it.

## Metrics

Proper Diarization Error Rate (DER) requires time-aligned speaker
ground truth (start/end timestamps per speaker turn). The v2 transcripts
have speaker labels per LINE but not per-second timestamps, so we
report:

- ``num_speakers_detected`` — should be 2 for all v2 episodes (each
  generated from a 2-voice macOS ``say`` transcript per RFC-059 §2).
- ``num_segments`` — total speaker turns the diarizer produced.
- ``ground_truth_turns`` — number of speaker-label changes in the
  reference transcript (a rough upper bound on the right answer).
- ``segments_per_turn_ratio`` — coarse measure of over-/under-segmentation.
- ``elapsed_seconds`` — wall-clock per episode.

Full DER computation is a follow-up that needs time-aligned ground truth
(generate via the materialized whisper outputs' word-level timestamps,
or use the macOS-``say`` per-line timing in the v2 generator).

## Usage

    python scripts/eval/score/diarization_dgx_vs_cloud_v1.py \\
        --backend dgx \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/runs/diarization_dgx_vs_cloud_v1/dgx

    # Local CPU (slower; use a smaller episode set for first pass)
    python scripts/eval/score/diarization_dgx_vs_cloud_v1.py \\
        --backend local \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 \\
        --output data/eval/runs/diarization_dgx_vs_cloud_v1/local
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Speaker-label regex tuned for v2 fixture transcripts ("Maya:" / "Liam:" /
# "Ad:" patterns at the start of dialog lines). Stricter than the more
# permissive whisper_accent_wer_v1.py pattern because we want clean turn
# detection here.
_SPEAKER_LINE_RE = re.compile(r"^([A-Z][A-Za-z .'\-]{0,40}):\s+")


def _count_ground_truth_turns(transcript_text: str) -> tuple[int, int]:
    """Return (num_distinct_speakers, num_turn_changes) from the transcript.

    Lines like ``Maya:`` / ``Liam:`` mark turns. Lines like ``Ad:`` are
    counted as another speaker (they're separate voices in v2 audio per
    RFC-059), but they're treated equivalently in turn-change counting.
    """
    speakers: list[str] = []
    for line in transcript_text.splitlines():
        m = _SPEAKER_LINE_RE.match(line.strip())
        if not m:
            continue
        speakers.append(m.group(1))
    distinct = sorted(set(speakers))
    changes = 0
    for i in range(1, len(speakers)):
        if speakers[i] != speakers[i - 1]:
            changes += 1
    # +1 because the first turn IS a turn even though no "change" preceded it.
    return len(distinct), changes + 1 if speakers else 0


# ---------------------------------------------------------------------------
# Backend dispatch


def _diarize_dgx(audio_path: Path) -> dict[str, Any]:
    """POST to the DGX pyannote-server /v1/diarize endpoint."""
    import requests

    url = os.environ.get(
        "DIARIZE_DGX_URL",
        "http://your-dgx.tailnet.ts.net:8001/v1/diarize",
    )
    t0 = time.time()
    with audio_path.open("rb") as fh:
        # Generous timeout — 90-min podcasts can take ~minutes on a
        # contended GPU. The diarization-resilience-gap (#954) covers
        # the production-grade timeout work; here we just want a
        # measurement, so we accept a long single-call wait.
        resp = requests.post(
            url,
            files={"file": (audio_path.name, fh, "audio/mpeg")},
            timeout=900,
        )
    elapsed = time.time() - t0
    resp.raise_for_status()
    payload = resp.json()
    return {
        "segments": payload.get("segments", []),
        "num_speakers": payload.get("num_speakers", 0),
        "model_name": payload.get("model_name", ""),
        "elapsed_seconds": elapsed,
    }


def _diarize_local(audio_path: Path) -> dict[str, Any]:
    """Local in-process pyannote. Device auto-picks: CUDA > MPS > CPU.

    Override with ``LOCAL_DIARIZE_DEVICE`` env var if you need to force a
    specific device (useful for the #930 honest comparison: setting
    ``LOCAL_DIARIZE_DEVICE=cpu`` measures pure CPU; the default lets pyannote
    pick the fastest available — typically MPS on Apple Silicon, which IS GPU
    acceleration via Metal and not actually a fair "CPU baseline").
    """
    from podcast_scraper.providers.ml.diarization.pyannote_provider import (
        PyAnnoteDiarizationProvider,
    )

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required for local pyannote (model is gated). "
            "Source ~/.env or infra/.env.dgx.local before running."
        )
    device = os.environ.get("LOCAL_DIARIZE_DEVICE", "auto").strip() or "auto"
    provider = PyAnnoteDiarizationProvider(hf_token=hf_token, device=device)
    t0 = time.time()
    result = provider.diarize(str(audio_path))
    elapsed = time.time() - t0
    segments = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in result.segments]
    return {
        "segments": segments,
        "num_speakers": result.num_speakers,
        "model_name": result.model_name,
        "elapsed_seconds": elapsed,
    }


_BACKENDS = {
    "dgx": _diarize_dgx,
    "local": _diarize_local,
}


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=sorted(_BACKENDS.keys()))
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--transcripts-dir", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for ep in args.episodes:
        audio_path = args.audio_dir / f"{ep}.mp3"
        transcript_path = args.transcripts_dir / f"{ep}.txt"
        if not audio_path.exists() or not transcript_path.exists():
            print(f"  SKIP {ep}: missing audio or transcript", file=sys.stderr)
            continue
        transcript_text = transcript_path.read_text(encoding="utf-8")
        gt_speakers, gt_turns = _count_ground_truth_turns(transcript_text)

        print(f"  {args.backend:5s} {ep}: diarizing…", file=sys.stderr)
        try:
            r = _BACKENDS[args.backend](audio_path)
            err = ""
        except Exception as exc:  # noqa: BLE001
            r = {"segments": [], "num_speakers": 0, "model_name": "", "elapsed_seconds": 0.0}
            err = str(exc)[:300]

        n_seg = len(r["segments"])
        ratio = (n_seg / gt_turns) if gt_turns else None
        row = {
            "backend": args.backend,
            "episode_id": ep,
            "num_speakers_detected": r["num_speakers"],
            "num_segments": n_seg,
            "ground_truth_speakers": gt_speakers,
            "ground_truth_turns": gt_turns,
            "segments_per_turn_ratio": round(ratio, 2) if ratio is not None else None,
            "speakers_match": r["num_speakers"] == gt_speakers,
            "elapsed_seconds": round(r["elapsed_seconds"], 2),
            "model_name": r["model_name"],
            "error": err,
        }
        rows.append(row)
        err_tail = f" ERROR={err}" if err else ""
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        print(
            f"    {ep}: detected_speakers={r['num_speakers']} "
            f"(gt={gt_speakers}) seg={n_seg} (gt_turns={gt_turns}, "
            f"ratio={ratio_str}) elapsed={r['elapsed_seconds']:.1f}s{err_tail}"
        )

    # Summary
    completed = [r for r in rows if not r["error"]]
    pct_speakers_correct = (
        100.0 * sum(1 for r in completed if r["speakers_match"]) / max(len(completed), 1)
    )
    mean_elapsed = sum(r["elapsed_seconds"] for r in completed) / max(len(completed), 1)
    mean_ratio = sum((r["segments_per_turn_ratio"] or 0.0) for r in completed) / max(
        len(completed), 1
    )
    summary = {
        "backend": args.backend,
        "episodes_attempted": len(rows),
        "episodes_completed": len(completed),
        "episodes_errored": len(rows) - len(completed),
        "pct_speakers_correct": round(pct_speakers_correct, 1),
        "mean_elapsed_seconds": round(mean_elapsed, 2),
        "mean_segments_per_turn_ratio": round(mean_ratio, 2),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "metrics_diarization_dgx_vs_cloud_v1",
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'backend':<6} {'episodes':>3} {'speakers_pct':>12} "
        f"{'mean_lat_s':>10} {'mean_ratio':>10}"
    )
    print(
        f"{summary['backend']:<6} "
        f"{summary['episodes_completed']:>3}/{summary['episodes_attempted']:<2} "
        f"{summary['pct_speakers_correct']:>11.1f}% "
        f"{summary['mean_elapsed_seconds']:>10.1f} "
        f"{summary['mean_segments_per_turn_ratio']:>10.2f}"
    )
    print(f"wrote {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
