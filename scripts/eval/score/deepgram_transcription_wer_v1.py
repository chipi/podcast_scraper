"""Deepgram nova-3 transcription-WER on v2 fixtures (#979/#980/#981 follow-up).

Closes the registry-materialization gap for the ``cloud_quality`` profile by
filling in the missing eval: Deepgram nova-3 vs OpenAI Whisper / MPS / DGX
on the same 5 v2 episodes that ``EVAL_TRANSCRIPTION_3WAY_2026_06.md`` covered.

Output schema mirrors ``whisper_accent_wer_v1.py`` so the headline tables stay
comparable across models. Uses the same WER routine + transcript normalisation
+ EPISODE_VOICES map.

Usage:
    DEEPGRAM_API_KEY=... python scripts/eval/score/deepgram_transcription_wer_v1.py \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --model nova-3 \\
        --output  data/eval/runs/baseline_deepgram_transcription_wer_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse the proven WER + normalisation + transcript-cleaning helpers.
from scripts.eval.score.whisper_accent_wer_v1 import (  # noqa: E402
    _normalize,
    _strip_transcript_metadata,
    EPISODE_VOICES,
    wer,
)


def _transcribe_deepgram(client: Any, audio_path: Path, model: str) -> tuple[str, float]:
    """Return (hypothesis_text, elapsed_seconds) from a single Deepgram call."""
    from podcast_scraper.providers.deepgram.deepgram_provider import (
        parse_deepgram_transcript,
    )

    with audio_path.open("rb") as f:
        audio_bytes = f.read()
    kwargs: dict[str, Any] = {
        "request": audio_bytes,
        "model": model,
        "smart_format": False,
        "punctuate": True,
        "language": "en",
    }
    t0 = time.time()
    response = client.listen.v1.media.transcribe_file(**kwargs)
    elapsed = time.time() - t0
    parsed = parse_deepgram_transcript(response)
    return str(parsed.get("text") or ""), elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--transcripts-dir", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--model", default="nova-3")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY env var not set", file=sys.stderr)
        return 2

    from deepgram import DeepgramClient

    client = DeepgramClient(api_key=api_key)

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for ep in args.episodes:
        audio_path = args.audio_dir / f"{ep}.mp3"
        transcript_path = args.transcripts_dir / f"{ep}.txt"
        if not audio_path.exists() or not transcript_path.exists():
            print(f"  SKIP {ep}: missing audio or transcript", file=sys.stderr)
            continue

        reference = _strip_transcript_metadata(transcript_path.read_text(encoding="utf-8"))
        hyp, elapsed = _transcribe_deepgram(client, audio_path, args.model)
        ep_wer = wer(reference, hyp)
        voices = EPISODE_VOICES.get(ep, {})
        row = {
            "model": args.model,
            "episode_id": ep,
            "host_voice": voices.get("host"),
            "guest_voice": voices.get("guest"),
            "wer": round(ep_wer, 4),
            "ref_word_count": len(_normalize(reference)),
            "hyp_word_count": len(_normalize(hyp)),
            "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"  {args.model:8s} {ep:8s} WER={ep_wer:.4f} "
            f"({len(_normalize(reference))}w ref / {len(_normalize(hyp))}w hyp) "
            f"voices={voices.get('host','?')}+{voices.get('guest','?')} "
            f"elapsed={elapsed:.1f}s"
        )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    summary = []
    for model_name in sorted(by_model):
        rs = by_model[model_name]
        wer_values = [r["wer"] for r in rs]
        summary.append(
            {
                "model": model_name,
                "episodes": len(rs),
                "mean_wer": round(sum(wer_values) / len(wer_values), 4),
                "max_wer": max(wer_values),
                "min_wer": min(wer_values),
                "mean_elapsed_s": round(sum(r["elapsed_s"] for r in rs) / len(rs), 1),
            }
        )

    (args.output / "metrics.json").write_text(
        json.dumps(
            {"schema": "metrics_deepgram_transcription_wer_v1", "summary": summary, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'model':<10} eps  mean_wer  min_wer  max_wer  mean_lat_s")
    for s in summary:
        print(
            f"{s['model']:<10} {s['episodes']:>3}  {s['mean_wer']:>8.4f} "
            f"{s['min_wer']:>7.4f}  {s['max_wer']:>7.4f}  {s['mean_elapsed_s']:>10.1f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
