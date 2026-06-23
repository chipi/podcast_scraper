"""#1046 measurement pass 2 — corpus-level small.en vs large-v3 timing + transcripts.

Runs ON DGX (``ssh dgx-llm-1 'python3 /path/to/this'``). Hits speaches at
127.0.0.1:8000 per-request with both models, serially (to avoid GPU
contention skewing latency). Writes raw transcripts and a timings CSV
to /tmp/1046_results/. The laptop-side analyzer (1046_corpus_analyze.py)
consumes these to compute entity-count agreement + r.

To re-run on a future corpus:
  1. rsync the audio files to dgx:/tmp/1046_audio/
  2. ssh dgx-llm-1 'nohup python3 /tmp/measure_dgx.py > /tmp/1046_results/run.log 2>&1 &'
  3. Poll until ``pgrep -f measure_dgx`` returns DEAD.
  4. rsync -a dgx-llm-1:/tmp/1046_results/ data/eval/runs/1046-measurement-pass-N/
  5. .venv/bin/python scripts/measure/1046_corpus_analyze.py --out-dir ...

This is a measurement artifact — write once, never mutate (per ADR-095
convention). Resume mode skips episodes whose ``timing_<stem>.json`` is
already present.

The skip-deep gate that this script measured was REJECTED for the
intelligence-extraction goal (see docs/wip/1046-WHISPER-MULTI-MODEL-DESIGN.md
§ 13). This script is preserved for future measurement passes of the same
shape (e.g. dual-pass reconciliation cost vs gain).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000/v1/audio/transcriptions"
AUDIO_DIR = Path("/tmp/1046_audio")
OUT_DIR = Path("/tmp/1046_results")
SMALL = "Systran/faster-whisper-small.en"
LARGE = "Systran/faster-whisper-large-v3"


def transcribe(audio: Path, model: str) -> tuple[str, float, int]:
    """Returns (text, wall_clock_seconds, http_status)."""
    with audio.open("rb") as f:
        files = {"file": (audio.name, f, "audio/mpeg")}
        data = {"model": model, "language": "en", "response_format": "json"}
        t = time.monotonic()
        r = httpx.post(API, files=files, data=data, timeout=2400.0)
        elapsed = time.monotonic() - t
    if r.status_code != 200:
        return (r.text, elapsed, r.status_code)
    return (r.json()["text"], elapsed, r.status_code)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "transcripts_small").mkdir(exist_ok=True)
    (OUT_DIR / "transcripts_large").mkdir(exist_ok=True)

    rows = []
    audio_files = sorted(AUDIO_DIR.glob("*.mp3"))
    print(f"[1046] Processing {len(audio_files)} audio files serially.")

    for i, audio in enumerate(audio_files, 1):
        stem = audio.stem
        small_path = OUT_DIR / "transcripts_small" / f"{stem}.txt"
        large_path = OUT_DIR / "transcripts_large" / f"{stem}.txt"
        timing_path = OUT_DIR / f"timing_{stem}.json"

        if timing_path.exists():
            # Resume — already done in a prior run.
            row = json.loads(timing_path.read_text())
            rows.append(row)
            print(f"[{i:>2}/{len(audio_files)}] {stem} ... skip (already done)")
            continue

        print(f"[{i:>2}/{len(audio_files)}] {stem} ... ", end="", flush=True)

        # small.en first (likely warm cache for second).
        if small_path.exists() and small_path.stat().st_size > 0:
            small_text = small_path.read_text()
            small_s, small_code = 0.0, 200  # already done; latency not re-measured
        else:
            small_text, small_s, small_code = transcribe(audio, SMALL)
            small_path.write_text(small_text)

        large_text, large_s, large_code = transcribe(audio, LARGE)
        large_path.write_text(large_text)

        audio_bytes = audio.stat().st_size
        row = {
            "episode": stem,
            "audio_bytes": audio_bytes,
            "small_status": small_code,
            "small_seconds": round(small_s, 3),
            "small_chars": len(small_text),
            "large_status": large_code,
            "large_seconds": round(large_s, 3),
            "large_chars": len(large_text),
            "ratio_large_over_small": round(large_s / small_s, 3) if small_s > 0 else None,
        }
        rows.append(row)
        timing_path.write_text(json.dumps(row))
        print(
            f"small={small_s:.2f}s ({small_code}) large={large_s:.2f}s "
            f"({large_code}) ratio={large_s / max(small_s, 0.001):.2f}x"
        )

    with (OUT_DIR / "timings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (OUT_DIR / "summary.json").open("w") as f:
        small_total = sum(r["small_seconds"] for r in rows)
        large_total = sum(r["large_seconds"] for r in rows)
        # geo mean ratio so big-episode outliers don't dominate. Filter rows
        # whose small_seconds is 0 (resumed without re-measuring).
        import math

        ratios = [r["ratio_large_over_small"] for r in rows if r["ratio_large_over_small"]]
        if ratios:
            geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        else:
            geomean = float("nan")
        json.dump(
            {
                "n_episodes": len(rows),
                "small_total_s": round(small_total, 2),
                "large_total_s": round(large_total, 2),
                "ratio_arithmetic_mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
                "ratio_geomean": round(geomean, 3) if ratios else None,
                "ratio_min": round(min(ratios), 3) if ratios else None,
                "ratio_max": round(max(ratios), 3) if ratios else None,
            },
            f,
            indent=2,
        )

    print(f"\n[1046] Done. Results in {OUT_DIR}.")


if __name__ == "__main__":
    main()
