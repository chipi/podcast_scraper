"""Bitrate sweep: how low can we go before Whisper WER degrades? (#577)

Re-encodes each episode at a range of MP3 bitrates, transcribes each variant,
scores WER against a reference transcript. Finds the knee-point below which
quality drops materially — that's the cost-optimal upload bitrate for the
Whisper API path (and the cost-optimal preprocessing bitrate for local).

Two phases:
  1. Local Whisper (small.en by default) — zero API cost, answers the
     knee-point question because the knee is a Whisper-model property.
  2. Optional `--api` validation — re-encodes at a single bitrate and sends
     to OpenAI Whisper API. Uses AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY
     with OPENAI_API_KEY fallback (per autoresearch convention).

Usage (Phase 1 — local, free):
    python scripts/eval/transcription_bitrate_sweep.py \\
        --audio-dir tests/fixtures/audio \\
        --reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \\
        --episodes p01_e03,p02_e03,p03_e03,p04_e03,p05_e03 \\
        --bitrates 32,48,64,96,128

Usage (Phase 2 — API validation at winning bitrate):
    export AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY=sk-...
    python scripts/eval/transcription_bitrate_sweep.py \\
        --audio-dir tests/fixtures/audio \\
        --reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \\
        --episodes p01_e03,p02_e03,p03_e03,p04_e03,p05_e03 \\
        --bitrates 48 \\
        --api
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".tmp" / "bitrate_sweep"
_RESULTS_DIR = _REPO_ROOT / "data" / "eval" / "runs" / "_bitrate_sweep"

DEFAULT_BITRATES = [32, 48, 64, 96, 128]
DEFAULT_MODEL = "small.en"
DEFAULT_SAMPLE_RATE = 16000  # 16kHz is Whisper's native; downsampling is safe


def _resolve_openai_key() -> Optional[str]:
    """AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY → OPENAI_API_KEY (autoresearch pattern)."""
    return os.environ.get("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )


def reencode(src: Path, dst: Path, bitrate_kbps: int, sample_rate: int) -> None:
    """Re-encode src → dst at given bitrate, mono, target sample rate. Skips if cached."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-b:a",
        f"{bitrate_kbps}k",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "libmp3lame",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def transcribe_local(audio_path: Path, model: Any) -> Dict[str, Any]:
    t0 = time.time()
    result = model.transcribe(str(audio_path), language="en")
    dt = time.time() - t0
    return {"text": result.get("text", ""), "transcribe_time": round(dt, 1)}


def transcribe_api(audio_path: Path, api_key: str) -> Dict[str, Any]:
    """OpenAI Whisper API (whisper-1). Cost: ~$0.006/min."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    t0 = time.time()
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
    dt = time.time() - t0
    return {"text": resp.text, "transcribe_time": round(dt, 1)}


def compute_wer(hypothesis: str, reference: str) -> float:
    """Levenshtein word-level WER, lowercased + whitespace-tokenized."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitrate sweep for Whisper (#577)")
    parser.add_argument("--audio-dir", required=True, help="Dir with source .mp3 files")
    parser.add_argument(
        "--reference-dir", required=True, help="Dir with reference .txt transcripts"
    )
    parser.add_argument("--episodes", required=True, help="Comma-separated episode IDs")
    parser.add_argument(
        "--bitrates",
        default=",".join(str(b) for b in DEFAULT_BITRATES),
        help=f"Comma-separated kbps (default: {','.join(str(b) for b in DEFAULT_BITRATES)})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Local Whisper model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--api", action="store_true", help="Use OpenAI Whisper API instead of local"
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ref_dir = Path(args.reference_dir)
    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    bitrates = [int(b) for b in args.bitrates.split(",") if b.strip()]

    api_key: Optional[str] = None
    if args.api:
        api_key = _resolve_openai_key()
        if not api_key:
            sys.exit(
                "--api set but no key found. Export AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY "
                "or OPENAI_API_KEY."
            )

    print(f"Audio dir:     {audio_dir}")
    print(f"Reference dir: {ref_dir}")
    print(f"Episodes:      {episodes}")
    print(f"Bitrates:      {bitrates} kbps")
    print(f"Mode:          {'OpenAI Whisper API' if args.api else 'local ' + args.model}")
    print(f"Sample rate:   {args.sample_rate} Hz, mono")
    print()

    refs: Dict[str, str] = {}
    for ep in episodes:
        ref_path = ref_dir / f"{ep}.txt"
        if ref_path.exists():
            refs[ep] = ref_path.read_text()
        else:
            print(f"  WARNING: no reference for {ep} — will skip WER")

    # Load local model once, reuse across all (ep, bitrate) combos.
    local_model = None
    if not args.api:
        import whisper

        t0 = time.time()
        print(f"Loading local Whisper model {args.model}...", flush=True)
        local_model = whisper.load_model(args.model)
        print(f"  loaded in {time.time() - t0:.1f}s\n")

    results: List[Dict[str, Any]] = []
    total_cost_usd = 0.0

    for bitrate in bitrates:
        print(f"{'='*70}")
        print(f"Bitrate: {bitrate} kbps")
        print(f"{'='*70}")
        for ep in episodes:
            src = audio_dir / f"{ep}.mp3"
            if not src.exists():
                print(f"  {ep}: SKIP (no source audio)")
                continue
            dst = _CACHE_DIR / f"{ep}_{bitrate}k_{args.sample_rate}hz_mono.mp3"
            print(f"  {ep}: encoding @ {bitrate}k...", end=" ", flush=True)
            try:
                reencode(src, dst, bitrate, args.sample_rate)
            except subprocess.CalledProcessError as e:
                print(f"FAIL ({e})")
                continue
            size_bytes = dst.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            print(f"{size_mb:.2f} MB, transcribing...", end=" ", flush=True)

            try:
                if args.api:
                    assert api_key is not None
                    t = transcribe_api(dst, api_key)
                    # Duration approx: API charges per second of audio. Use ffprobe.
                    dur_s = _probe_duration(dst)
                    cost = dur_s / 60.0 * 0.006
                    total_cost_usd += cost
                else:
                    assert local_model is not None
                    t = transcribe_local(dst, local_model)
                    cost = 0.0
                    dur_s = None
            except Exception as e:
                print(f"FAIL ({e})")
                continue

            hyp = t["text"]
            ref = refs.get(ep, "")
            wer = compute_wer(hyp, ref) if ref else None
            wer_str = f"WER={wer:.1%}" if wer is not None else "no ref"
            extra = f", ${cost:.4f}" if args.api else ""
            print(f"{t['transcribe_time']:.1f}s, {wer_str}{extra}")

            results.append(
                {
                    "bitrate_kbps": bitrate,
                    "episode": ep,
                    "model": "whisper-1" if args.api else args.model,
                    "mode": "api" if args.api else "local",
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 2),
                    "transcribe_time_s": t["transcribe_time"],
                    "wer": round(wer, 4) if wer is not None else None,
                    "chars": len(hyp),
                    "audio_duration_s": dur_s,
                    "estimated_cost_usd": round(cost, 6) if args.api else None,
                }
            )
        print()

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Bitrate':>9s} {'Avg WER':>8s} {'Avg Size':>10s} {'Avg Time':>10s} {'Eps':>4s}")
    print("-" * 55)
    by_br: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_br[r["bitrate_kbps"]].append(r)
    for br in sorted(by_br.keys()):
        rows = by_br[br]
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        sizes = [r["size_mb"] for r in rows]
        times = [r["transcribe_time_s"] for r in rows]
        wer_avg = statistics.mean(wers) if wers else 0.0
        print(
            f"{br:>6d} k  {wer_avg:>7.1%}  {statistics.mean(sizes):>7.2f} MB  "
            f"{statistics.mean(times):>8.1f}s  {len(rows):>4d}"
        )

    if args.api:
        print(f"\nTotal API cost (this run): ${total_cost_usd:.4f}")

    # Write results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = "api" if args.api else "local"
    out_path = _RESULTS_DIR / f"bitrate_sweep_{suffix}_{ts}.json"
    payload = {
        "sweep_mode": "api" if args.api else "local",
        "model": "whisper-1" if args.api else args.model,
        "bitrates_kbps": bitrates,
        "episodes": episodes,
        "sample_rate_hz": args.sample_rate,
        "total_cost_usd": round(total_cost_usd, 6) if args.api else 0.0,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults: {out_path}")


def _probe_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


if __name__ == "__main__":
    main()
