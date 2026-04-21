"""Transcription model sweep: compare local Whisper models on quality + speed.

Transcribes held-out audio with each Whisper model size, then measures:
1. WER against a reference transcript (from API Whisper or existing materialized)
2. Wall-clock time per episode
3. Optionally: downstream summary quality (chain test)

Usage:
    # Sweep local models against existing materialized transcripts as reference:
    python scripts/eval/transcription_sweep.py \
        --audio-dir tests/fixtures/audio \
        --reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \
        --episodes p01_e03,p02_e03,p03_e03,p04_e03,p05_e03

    # Single model:
    python scripts/eval/transcription_sweep.py \
        --audio-dir tests/fixtures/audio \
        --reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \
        --episodes p01_e03 \
        --models base.en
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Models to sweep: ordered small → large
DEFAULT_MODELS = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    # "large-v3",  # Very slow on MPS; uncomment for overnight runs
]


def transcribe_episode(
    audio_path: Path,
    model_name: str,
) -> Dict[str, Any]:
    """Transcribe one audio file with local Whisper."""
    import whisper

    t0 = time.time()
    model = whisper.load_model(model_name)
    load_time = time.time() - t0

    t1 = time.time()
    result = model.transcribe(str(audio_path), language="en")
    transcribe_time = time.time() - t1

    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "load_time": round(load_time, 1),
        "transcribe_time": round(transcribe_time, 1),
        "total_time": round(load_time + transcribe_time, 1),
    }


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate between hypothesis and reference."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Simple Levenshtein on word level
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


def main():
    parser = argparse.ArgumentParser(description="Transcription model sweep")
    parser.add_argument("--audio-dir", required=True, help="Dir with .mp3 files")
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Dir with reference .txt transcripts",
    )
    parser.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated episode IDs (e.g. p01_e03,p02_e03)",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated Whisper models (default: {','.join(DEFAULT_MODELS)})",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ref_dir = Path(args.reference_dir)
    episodes = [e.strip() for e in args.episodes.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    print(f"Audio dir:     {audio_dir}")
    print(f"Reference dir: {ref_dir}")
    print(f"Episodes:      {episodes}")
    print(f"Models:        {models}")
    print()

    # Load reference transcripts
    refs: Dict[str, str] = {}
    for ep in episodes:
        ref_path = ref_dir / f"{ep}.txt"
        if ref_path.exists():
            refs[ep] = ref_path.read_text()
        else:
            print(f"  WARNING: no reference for {ep}")

    results: List[Dict[str, Any]] = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model_wers = []
        model_times = []

        for ep in episodes:
            audio_path = audio_dir / f"{ep}.mp3"
            if not audio_path.exists():
                print(f"  {ep}: SKIP (no audio)")
                continue

            print(f"  {ep}: transcribing...", end="", flush=True)
            try:
                result = transcribe_episode(audio_path, model_name)
            except Exception as e:
                print(f" ERROR: {e}")
                continue

            transcript = result["text"]
            ref = refs.get(ep, "")

            if ref:
                wer = compute_wer(transcript, ref)
                model_wers.append(wer)
            else:
                wer = None

            model_times.append(result["transcribe_time"])

            wer_str = f"WER={wer:.1%}" if wer is not None else "no ref"
            print(f" {len(transcript)} chars, " f"{result['transcribe_time']:.1f}s, " f"{wer_str}")

            results.append(
                {
                    "model": model_name,
                    "episode": ep,
                    "chars": len(transcript),
                    "wer": round(wer, 4) if wer is not None else None,
                    "transcribe_time": result["transcribe_time"],
                    "load_time": result["load_time"],
                }
            )

        if model_wers:
            avg_wer = statistics.mean(model_wers)
            avg_time = statistics.mean(model_times)
            print(f"\n  AVG: WER={avg_wer:.1%}, " f"time={avg_time:.1f}s/ep")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15s} {'Avg WER':>8s} {'Avg Time':>10s} " f"{'Episodes':>9s}")
    print("-" * 45)
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    for model_name in models:
        rows = by_model.get(model_name, [])
        if not rows:
            continue
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        times = [r["transcribe_time"] for r in rows]

        avg_wer = statistics.mean(wers) if wers else 0
        avg_time = statistics.mean(times) if times else 0
        print(f"{model_name:<15s} {avg_wer:>7.1%} {avg_time:>9.1f}s " f"{len(rows):>9d}")

    # Save results
    out_dir = Path("data/eval/runs/_transcription_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sweep_{ts}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
