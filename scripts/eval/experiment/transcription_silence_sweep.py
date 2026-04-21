"""Silence-trim aggressiveness sweep: how much audio duration can we cut without hurting WER? (#577)

Silence trim is THE real cost lever for the Whisper API path. API pricing is
duration-based, so every second trimmed = proportional $ saved. Bitrate doesn't
reduce API $ directly; duration does.

Audio fixed at 32 kbps / 16 kHz / mono (Exp 1 winner). Sweeps
(silence_threshold_dB, silence_min_duration_s) and measures:

  - Audio duration before vs after trim (% reduction)
  - WER on trimmed audio vs original reference transcript (local small.en oracle)
  - Transcript char count (red flag if dropping > 3% — suggests we cut real speech)

Usage:
    python scripts/eval/transcription_silence_sweep.py \\
        --audio-dir tests/fixtures/audio \\
        --reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \\
        --episodes p01_e03,p02_e03,p03_e03,p04_e03,p05_e03

Configs swept by default (threshold_dB, duration_s):
  (-50, 2.0)   baseline — current pipeline default
  (-45, 1.5)   mild
  (-40, 1.0)   moderate (risky — may cut short inter-sentence pauses)
  (-55, 3.0)   conservative (safer, smaller duration cut)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".tmp" / "silence_sweep"
_RESULTS_DIR = _REPO_ROOT / "data" / "eval" / "runs" / "_silence_sweep"

# (threshold_dB, min_silence_duration_s). Negative dB; more-negative = quieter cutoff.
DEFAULT_CONFIGS: List[Tuple[int, float]] = [
    (-50, 2.0),  # baseline (current default)
    (-45, 1.5),  # mild
    (-40, 1.0),  # moderate (risky)
    (-55, 3.0),  # conservative
]
DEFAULT_MODEL = "small.en"
DEFAULT_BITRATE_KBPS = 32
DEFAULT_SAMPLE_RATE_HZ = 16000

# Safety thresholds — if exceeded, flag the config as suspect.
WER_DELTA_MAX_PP = 1.0  # max acceptable WER increase vs baseline (percentage points)
CHAR_DROP_MAX_PCT = 3.0  # max acceptable transcript char-count drop vs baseline


def reencode_with_silence_trim(
    src: Path,
    dst: Path,
    bitrate_kbps: int,
    sample_rate_hz: int,
    silence_threshold_db: int,
    silence_duration_s: float,
) -> None:
    """ffmpeg: resample + mono + silenceremove at given config + mp3 encode. Skip if cached."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # stop_periods=-1 removes all silences meeting criteria (not just trailing).
    silence_filter = (
        f"silenceremove="
        f"stop_periods=-1:"
        f"stop_threshold={silence_threshold_db}dB:"
        f"stop_duration={silence_duration_s}"
    )
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-af",
        silence_filter,
        "-ar",
        str(sample_rate_hz),
        "-ac",
        "1",
        "-b:a",
        f"{bitrate_kbps}k",
        "-c:a",
        "libmp3lame",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def probe_duration_s(path: Path) -> float:
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


def transcribe_local(audio_path: Path, model: Any) -> Dict[str, Any]:
    t0 = time.time()
    result = model.transcribe(str(audio_path), language="en")
    dt = time.time() - t0
    return {"text": result.get("text", ""), "transcribe_time": round(dt, 1)}


def compute_wer(hypothesis: str, reference: str) -> float:
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


def parse_configs(s: str) -> List[Tuple[int, float]]:
    """Parse 'threshold_dB:duration_s,threshold_dB:duration_s,...' into list of tuples."""
    configs: List[Tuple[int, float]] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        th, dur = chunk.split(":")
        configs.append((int(th), float(dur)))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Silence-trim sweep (#577)")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--configs",
        default=",".join(f"{t}:{d}" for t, d in DEFAULT_CONFIGS),
        help=(
            "Comma-separated silence configs as <threshold_dB>:<duration_s>. "
            f"Default: {','.join(f'{t}:{d}' for t, d in DEFAULT_CONFIGS)}"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE_KBPS)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ref_dir = Path(args.reference_dir)
    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    configs = parse_configs(args.configs)

    print(f"Audio dir:     {audio_dir}")
    print(f"Reference dir: {ref_dir}")
    print(f"Episodes:      {episodes}")
    print(f"Silence cfgs:  {configs} (threshold_dB, duration_s)")
    print(f"Fixed:         {args.bitrate} kbps, {args.sample_rate} Hz, mono, model={args.model}")
    print()

    refs: Dict[str, str] = {}
    for ep in episodes:
        ref_path = ref_dir / f"{ep}.txt"
        if ref_path.exists():
            refs[ep] = ref_path.read_text()

    # Load Whisper once.
    import whisper

    t0 = time.time()
    print(f"Loading {args.model}...", flush=True)
    model = whisper.load_model(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s\n", flush=True)

    # Probe original duration per episode (once) for reduction computation.
    original_durations: Dict[str, float] = {}
    for ep in episodes:
        src = audio_dir / f"{ep}.mp3"
        if src.exists():
            original_durations[ep] = probe_duration_s(src)

    results: List[Dict[str, Any]] = []

    for th_db, dur_s in configs:
        print(f"{'='*70}")
        print(f"Silence config: threshold={th_db}dB, duration={dur_s}s")
        print(f"{'='*70}", flush=True)
        for ep in episodes:
            src = audio_dir / f"{ep}.mp3"
            if not src.exists():
                print(f"  {ep}: SKIP (no source)", flush=True)
                continue
            tag = f"{ep}_b{args.bitrate}k_th{abs(th_db)}_d{str(dur_s).replace('.', 'p')}"
            dst = _CACHE_DIR / f"{tag}.mp3"
            print(f"  {ep}: trimming + encoding...", end=" ", flush=True)
            try:
                reencode_with_silence_trim(src, dst, args.bitrate, args.sample_rate, th_db, dur_s)
            except subprocess.CalledProcessError as e:
                print(f"FFMPEG FAIL ({e})", flush=True)
                continue

            trimmed_dur = probe_duration_s(dst)
            orig_dur = original_durations.get(ep, 0.0)
            dur_reduction_pct = (orig_dur - trimmed_dur) / orig_dur * 100.0 if orig_dur > 0 else 0.0
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(
                f"orig={orig_dur:.0f}s → trimmed={trimmed_dur:.0f}s "
                f"(-{dur_reduction_pct:.1f}%), {size_mb:.2f} MB, transcribing...",
                end=" ",
                flush=True,
            )

            try:
                t = transcribe_local(dst, model)
            except Exception as e:
                print(f"WHISPER FAIL ({e})", flush=True)
                continue

            hyp = t["text"]
            ref = refs.get(ep, "")
            wer = compute_wer(hyp, ref) if ref else None
            wer_str = f"WER={wer:.1%}" if wer is not None else "no ref"
            print(f"{t['transcribe_time']:.1f}s, {wer_str}", flush=True)

            results.append(
                {
                    "silence_threshold_db": th_db,
                    "silence_duration_s": dur_s,
                    "episode": ep,
                    "model": args.model,
                    "bitrate_kbps": args.bitrate,
                    "sample_rate_hz": args.sample_rate,
                    "orig_duration_s": round(orig_dur, 1),
                    "trimmed_duration_s": round(trimmed_dur, 1),
                    "duration_reduction_pct": round(dur_reduction_pct, 2),
                    "size_mb": round(size_mb, 2),
                    "transcribe_time_s": t["transcribe_time"],
                    "wer": round(wer, 4) if wer is not None else None,
                    "chars": len(hyp),
                }
            )
        print(flush=True)

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    by_cfg: Dict[Tuple[int, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_cfg[(r["silence_threshold_db"], r["silence_duration_s"])].append(r)

    print(
        f"{'Config':>14s}  {'Avg dur↓':>9s}  {'Avg WER':>8s}  "
        f"{'Avg chars':>10s}  {'Size':>7s}  {'Eps':>4s}"
    )
    print("-" * 66)

    baseline_wer: float = 0.0
    baseline_chars: float = 0.0
    for (th, dur), rows in sorted(by_cfg.items(), key=lambda kv: -kv[0][0]):
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        drs = [r["duration_reduction_pct"] for r in rows]
        chars = [r["chars"] for r in rows]
        sizes = [r["size_mb"] for r in rows]
        wer_avg = statistics.mean(wers) if wers else 0.0
        dr_avg = statistics.mean(drs) if drs else 0.0
        ch_avg = statistics.mean(chars) if chars else 0.0
        sz_avg = statistics.mean(sizes) if sizes else 0.0
        if (th, dur) == (-50, 2.0):  # baseline
            baseline_wer = wer_avg
            baseline_chars = ch_avg
        print(
            f"  th={th:>3d}dB d={dur:>3.1f}s  {dr_avg:>7.1f}%  "
            f"{wer_avg:>7.1%}  {ch_avg:>10.0f}  {sz_avg:>5.2f}MB  {len(rows):>4d}"
        )

    # Safety flags
    print()
    print("Safety flags (WER Δ > 1pp or chars Δ > 3% vs baseline -50dB/2.0s):")
    for (th, dur), rows in sorted(by_cfg.items(), key=lambda kv: -kv[0][0]):
        if (th, dur) == (-50, 2.0):
            continue
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        chars = [r["chars"] for r in rows]
        if not wers or not chars:
            continue
        wer_avg = statistics.mean(wers)
        ch_avg = statistics.mean(chars)
        wer_delta_pp = (wer_avg - baseline_wer) * 100.0
        char_drop_pct = (
            (baseline_chars - ch_avg) / baseline_chars * 100.0 if baseline_chars else 0.0
        )
        flags = []
        if wer_delta_pp > WER_DELTA_MAX_PP:
            flags.append(f"WER +{wer_delta_pp:.2f}pp")
        if char_drop_pct > CHAR_DROP_MAX_PCT:
            flags.append(f"chars -{char_drop_pct:.2f}%")
        status = "FLAG: " + ", ".join(flags) if flags else "ok"
        print(f"  th={th:>3d}dB d={dur:>3.1f}s → {status}")

    # Write results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = _RESULTS_DIR / f"silence_sweep_{ts}.json"
    payload = {
        "model": args.model,
        "bitrate_kbps": args.bitrate,
        "sample_rate_hz": args.sample_rate,
        "configs": configs,
        "episodes": episodes,
        "results": results,
        "safety": {
            "wer_delta_max_pp": WER_DELTA_MAX_PP,
            "char_drop_max_pct": CHAR_DROP_MAX_PCT,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
