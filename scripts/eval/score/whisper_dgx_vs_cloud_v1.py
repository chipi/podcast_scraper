"""Transcription championship: Speaches/DGX vs local CPU vs cloud Whisper (#929).

Extends ``whisper_accent_wer_v1.py`` (#906B) with two new transcription
backends so we can answer the question:

    Does Speaches on DGX beat local CPU base.en on the WER × latency × cost
    tradeoff, and how does either compare to cloud OpenAI Whisper API on
    the 90-min real-episode shape?

## Backends

- ``local``  — OpenAI's ``openai-whisper`` Python lib, model loaded in-process
  (matches the path the prod pipeline uses today). Cost = $0.
- ``dgx``    — Speaches HTTP endpoint on the DGX. POST a multipart audio file
  to ``/v1/audio/transcriptions``; OpenAI-compatible response. Cost = $0
  marginal (we own the DGX). The custom Blackwell build from #948 is what
  makes this actually GPU-accelerated.
- ``cloud``  — OpenAI Whisper API (same OpenAI-compatible response shape).
  Cost = $0.006 / minute audio (June 2026 public pricing).

## Metrics

Per (backend, model, episode):
- WER (Levenshtein word-edit over normalized text) vs ground-truth transcript.
- Single-request latency (wall-clock seconds for one transcribe call).
- Cost in USD.

Across the full sweep:
- Burst-latency check (5 concurrent transcribe calls of the same episode).
  Tells us whether the backend keeps up under concurrency, or queues / dies.

## Inputs

Reuses #906B's v2 audio bed (``tests/fixtures/audio/v2/``) for accent coverage
(15 episodes × 9 macOS ``say`` voice combos). Add one real 90-min episode
(``manual-run-10`` corpus) for the production-shape latency signal — short
v2 clips are overhead-dominated and don't surface the DGX advantage.

## Usage

    # Local CPU baseline
    python scripts/eval/score/whisper_dgx_vs_cloud_v1.py \\
        --backend local --models tiny.en base.en small.en \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/runs/whisper_dgx_vs_cloud_v1/local

    # DGX (requires WHISPER_DGX_URL pointing at Speaches /v1/audio/transcriptions)
    WHISPER_DGX_URL=http://your-dgx.tailnet.ts.net:8000/v1/audio/transcriptions \\
    python scripts/eval/score/whisper_dgx_vs_cloud_v1.py \\
        --backend dgx \\
        --models Systran/faster-whisper-base.en \\
                 Systran/faster-whisper-small.en \\
                 Systran/faster-whisper-medium.en \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/runs/whisper_dgx_vs_cloud_v1/dgx

    # Cloud (requires AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY in env)
    python scripts/eval/score/whisper_dgx_vs_cloud_v1.py \\
        --backend cloud --models whisper-1 \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/runs/whisper_dgx_vs_cloud_v1/cloud

    # Burst-latency check on a single backend (--burst 5 → 5 concurrent jobs)
    python scripts/eval/score/whisper_dgx_vs_cloud_v1.py \\
        --backend dgx --models Systran/faster-whisper-base.en \\
        --episodes p01_e01 --burst 5 ...
"""

from __future__ import annotations

import argparse
import concurrent.futures as _futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Cloud pricing — public OpenAI Whisper API rate, June 2026. Edit if upstream
# changes (we own no fallback; if pricing moves the cost column moves with it).
CLOUD_PRICE_USD_PER_MIN_AUDIO = 0.006

# ---------------------------------------------------------------------------
# WER + transcript normalization (mirrors whisper_accent_wer_v1.py so the
# two harnesses' numbers are directly comparable — copy is intentional,
# 20 lines isn't worth a shared module for these one-shot scripts).

_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = _NORMALIZE_RE.sub(" ", text)
    return text.split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate via Levenshtein. Returns 0.0 for empty ref+hyp."""
    r = _normalize(ref)
    h = _normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    nr, nh = len(r), len(h)
    dp = [[0] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(nh + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[nr][nh] / nr


def _strip_transcript_metadata(text: str) -> str:
    """Drop header / speaker tags / timestamps so WER measures spoken content only."""
    lines = text.splitlines()
    body: list[str] = []
    speaker_re = re.compile(r"^[A-Z][A-Za-z .'\-]{0,40}:\s+")
    ts_re = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]$")
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or ts_re.fullmatch(s):
            continue
        m = speaker_re.match(s)
        body.append(m.group(0).join(s.split(m.group(0))[1:]) if m else s)
    return " ".join(body)


# ---------------------------------------------------------------------------
# Backend dispatch.
#
# All three backends return ``(transcript_text, audio_duration_seconds)``.
# The audio duration is needed both for cost calc (cloud) and for
# realtime-multiple reporting. We probe duration with ffprobe once per
# episode and pass it through.


def _audio_duration_seconds(path: Path) -> float:
    """Use ffprobe to read audio duration. Fallback: 0.0 (cost won't be computed)."""
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
            timeout=30,
        )
        return float(out.strip())
    except Exception as exc:
        print(f"  ffprobe failed on {path.name}: {exc}", file=sys.stderr)
        return 0.0


def _transcribe_local(audio_path: Path, model_name: str, model_cache: dict[str, Any]) -> str:
    """Local in-process openai-whisper.

    Device auto-picks: CUDA > MPS > CPU. Override with the
    ``LOCAL_WHISPER_DEVICE`` env var (``cuda`` / ``mps`` / ``cpu``) for
    the #929 honest comparison — without that, this picks the fastest
    available, which is MPS on Apple Silicon (GPU acceleration via Metal,
    NOT a fair pure-CPU baseline).
    """
    import whisper

    cache_dir = PROJECT_ROOT / ".cache" / "whisper"
    device = os.environ.get("LOCAL_WHISPER_DEVICE", "").strip() or None
    cache_key = f"{model_name}|{device or 'auto'}"
    if cache_key not in model_cache:
        print(f"  loading whisper {model_name} (device={device or 'auto'})...", file=sys.stderr)
        t0 = time.time()
        kwargs: dict[str, Any] = {"download_root": str(cache_dir)}
        if device:
            kwargs["device"] = device
        model_cache[cache_key] = whisper.load_model(model_name, **kwargs)
        print(f"    loaded in {time.time()-t0:.1f}s", file=sys.stderr)
    model = model_cache[cache_key]
    # fp16=True on GPU (MPS/CUDA), False on CPU. openai-whisper warns +
    # falls back when fp16 is wrong; we set explicitly to avoid the warn.
    use_fp16 = device != "cpu"
    result = model.transcribe(str(audio_path), verbose=False, fp16=use_fp16)
    return result["text"]


def _transcribe_dgx(audio_path: Path, model_name: str, _model_cache: dict[str, Any]) -> str:
    """DGX Speaches via HTTP /v1/audio/transcriptions."""
    import requests

    url = os.environ.get("WHISPER_DGX_URL")
    if not url:
        raise RuntimeError(
            "WHISPER_DGX_URL unset — e.g. "
            "http://your-dgx.tailnet.ts.net:8000/v1/audio/transcriptions"
        )
    with audio_path.open("rb") as fh:
        # Timeout sized for a 90-min episode on a slow DGX day. The actual
        # transcription should take seconds on the custom-build #948 image.
        # 1500s (25 min) covers the worst case observed during #957 — the
        # speaches faster-whisper int8 path needs ~700s for a 11-min audio
        # on a busy DGX, ~3× what the int8 single-ep ping suggested.
        resp = requests.post(
            url,
            files={"file": (audio_path.name, fh, "audio/mpeg")},
            data={"model": model_name, "response_format": "json"},
            timeout=1500,
        )
    resp.raise_for_status()
    return resp.json().get("text", "")


def _transcribe_cloud(audio_path: Path, model_name: str, model_cache: dict[str, Any]) -> str:
    """OpenAI Whisper API."""
    if "_openai_client" not in model_cache:
        from openai import OpenAI

        # Mirror the autoresearch keying convention from the finale judge
        # clients: never fall through to the plain prod ``OPENAI_API_KEY`` —
        # autoresearch evals must spend on the autoresearch budget.
        api_key = (
            os.environ.get("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", "").strip()
            or os.environ.get("AUTORESEARCH_JUDGE_OPENAI_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError(
                "AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY unset; refuse to charge prod key"
            )
        model_cache["_openai_client"] = OpenAI(api_key=api_key)
    client = model_cache["_openai_client"]
    with audio_path.open("rb") as fh:
        resp = client.audio.transcriptions.create(
            model=model_name,
            file=fh,
            response_format="json",
        )
    return resp.text


_BACKENDS = {
    "local": _transcribe_local,
    "dgx": _transcribe_dgx,
    "cloud": _transcribe_cloud,
}


def _cost_usd(backend: str, audio_seconds: float) -> float:
    """USD cost for one transcription call."""
    if backend == "cloud":
        return audio_seconds / 60.0 * CLOUD_PRICE_USD_PER_MIN_AUDIO
    return 0.0  # local + dgx have no marginal cost


# ---------------------------------------------------------------------------
# Sweep + burst harnesses


def _one_call(
    *,
    backend: str,
    model_name: str,
    audio_path: Path,
    transcript_path: Path,
    audio_seconds: float,
    model_cache: dict[str, Any],
) -> dict[str, Any]:
    """Single (backend, model, episode) measurement."""
    reference = _strip_transcript_metadata(transcript_path.read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        hyp = _BACKENDS[backend](audio_path, model_name, model_cache)
        err = ""
    except Exception as exc:  # noqa: BLE001
        hyp = ""
        err = str(exc)[:300]
    elapsed = time.time() - t0
    ep_wer = wer(reference, hyp) if hyp else 1.0
    return {
        "backend": backend,
        "model": model_name,
        "episode_id": audio_path.stem,
        "wer": round(ep_wer, 4),
        "elapsed_s": round(elapsed, 2),
        "audio_seconds": round(audio_seconds, 2),
        "realtime_multiple": (
            round(audio_seconds / elapsed, 1) if elapsed > 0 and audio_seconds > 0 else None
        ),
        "cost_usd": round(_cost_usd(backend, audio_seconds), 6),
        "ref_word_count": len(_normalize(reference)),
        "hyp_word_count": len(_normalize(hyp)),
        "error": err,
    }


def _burst(
    *,
    backend: str,
    model_name: str,
    audio_path: Path,
    transcript_path: Path,
    audio_seconds: float,
    concurrency: int,
    model_cache: dict[str, Any],
) -> dict[str, Any]:
    """Fire ``concurrency`` parallel transcribe calls on the same episode.

    Reports the slowest call and the mean — what we care about is "does the
    backend keep up under concurrency, or does it queue / fall over?".
    """
    if backend == "local":
        # The local whisper lib is not thread-safe and would just serialize on
        # the GIL. Burst on local is meaningless; skip and report a sentinel.
        return {
            "backend": backend,
            "model": model_name,
            "episode_id": audio_path.stem,
            "concurrency": concurrency,
            "skipped_reason": "local backend serializes via GIL",
        }
    rows: list[dict[str, Any]] = []
    with _futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _one_call,
                backend=backend,
                model_name=model_name,
                audio_path=audio_path,
                transcript_path=transcript_path,
                audio_seconds=audio_seconds,
                model_cache=model_cache,
            )
            for _ in range(concurrency)
        ]
        for f in _futures.as_completed(futures):
            rows.append(f.result())
    elapsed = [r["elapsed_s"] for r in rows if not r.get("error")]
    return {
        "backend": backend,
        "model": model_name,
        "episode_id": audio_path.stem,
        "concurrency": concurrency,
        "n_completed": len(elapsed),
        "n_errored": len(rows) - len(elapsed),
        "elapsed_s_max": max(elapsed) if elapsed else None,
        "elapsed_s_mean": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
        "audio_seconds": round(audio_seconds, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=sorted(_BACKENDS.keys()))
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--transcripts-dir", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--burst",
        type=int,
        default=0,
        help="If >0, also run a burst test with this many concurrent jobs per (model, episode).",
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    model_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    burst_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        for ep in args.episodes:
            audio_path = args.audio_dir / f"{ep}.mp3"
            transcript_path = args.transcripts_dir / f"{ep}.txt"
            if not audio_path.exists() or not transcript_path.exists():
                print(f"  SKIP {ep}: missing audio or transcript", file=sys.stderr)
                continue
            audio_seconds = _audio_duration_seconds(audio_path)
            row = _one_call(
                backend=args.backend,
                model_name=model_name,
                audio_path=audio_path,
                transcript_path=transcript_path,
                audio_seconds=audio_seconds,
                model_cache=model_cache,
            )
            rows.append(row)
            rt = row["realtime_multiple"]
            rt_s = f"{rt:.1f}×" if rt is not None else "n/a"
            err_tail = f" ERROR={row['error']}" if row["error"] else ""
            print(
                f"  {args.backend:5s} {model_name:50s} {ep:8s} "
                f"WER={row['wer']:.4f} elapsed={row['elapsed_s']:.1f}s "
                f"rt={rt_s} cost=${row['cost_usd']:.4f}{err_tail}"
            )
            if args.burst > 0:
                br = _burst(
                    backend=args.backend,
                    model_name=model_name,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    audio_seconds=audio_seconds,
                    concurrency=args.burst,
                    model_cache=model_cache,
                )
                burst_rows.append(br)
                if br.get("skipped_reason"):
                    print(f"    burst skipped: {br['skipped_reason']}")
                else:
                    print(
                        f"    burst×{args.burst}: completed={br['n_completed']} "
                        f"errored={br['n_errored']} "
                        f"max={br['elapsed_s_max']}s mean={br['elapsed_s_mean']}s"
                    )

    # Per-(backend, model) summary
    from collections import defaultdict

    by_bm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        by_bm[(r["backend"], r["model"])].append(r)
    summary: list[dict[str, Any]] = []
    for (backend, model_name), rs in sorted(by_bm.items()):
        wer_values = [r["wer"] for r in rs]
        elapsed_values = [r["elapsed_s"] for r in rs]
        audio_sec_total = sum(r["audio_seconds"] for r in rs)
        cost_total = sum(r["cost_usd"] for r in rs)
        summary.append(
            {
                "backend": backend,
                "model": model_name,
                "episodes": len(rs),
                "mean_wer": round(sum(wer_values) / len(wer_values), 4),
                "max_wer": max(wer_values),
                "min_wer": min(wer_values),
                "mean_elapsed_s": round(sum(elapsed_values) / len(elapsed_values), 2),
                "mean_realtime_multiple": (
                    round(audio_sec_total / sum(elapsed_values), 1)
                    if sum(elapsed_values) > 0
                    else None
                ),
                "total_audio_seconds": round(audio_sec_total, 1),
                "total_cost_usd": round(cost_total, 4),
            }
        )

    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "metrics_whisper_dgx_vs_cloud_v1",
                "backend": args.backend,
                "summary": summary,
                "rows": rows,
                "burst": burst_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'backend':<6} {'model':<48} eps  mean_wer   max_wer   "
        f"mean_lat_s   mean_rt  total_$"
    )
    for s in summary:
        rt_s = f"{s['mean_realtime_multiple']}×" if s["mean_realtime_multiple"] else "n/a"
        print(
            f"{s['backend']:<6} {s['model']:<48} {s['episodes']:>3}  "
            f"{s['mean_wer']:>8.4f}  {s['max_wer']:>7.4f}  "
            f"{s['mean_elapsed_s']:>10.1f}  {rt_s:>6}  ${s['total_cost_usd']:>6.4f}"
        )
    print(f"wrote {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
