"""Whisper accent-WER sweep for #906 Phase B.

Runs each Whisper model variant on v2 audio fixtures and computes WER against
the source transcript (which is ground truth since v2 audio was generated FROM
the transcript via macOS `say` with per-speaker voices). Surfaces whether
non-US-English voices cause meaningful WER degradation per model tier.

The v2 voice map (RFC-059 §2) assigns 21 distinct macOS `say` voices,
including Indian English (Isha, Rishi), Italian (Luca), Mexican Spanish
(Paulina), Irish (Moira), Australian (Karen). The current Whisper default is
`tiny.en` — was chosen against (probably) en_US-distributed test inputs.

Usage:
    python scripts/eval/score/whisper_accent_wer_v1.py \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --models tiny.en base.en \\
        --output  data/eval/runs/baseline_whisper_accent_wer_v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Per-episode primary voices (host + guest) for cross-referencing — pulled from
# RFC-059 § voice map / generator spec. Used to correlate WER with accent tier.
EPISODE_VOICES: dict[str, dict[str, str]] = {
    "p01_e01": {"host": "Samantha (US-en)", "guest": "Fred (US-en)"},
    "p01_e02": {"host": "Samantha (US-en)", "guest": "Flo (UK-en)"},
    "p01_e03": {"host": "Samantha (US-en)", "guest": "Tom (US-en)"},
    "p02_e01": {"host": "Alex (US-en)", "guest": "Isha (en-IN)"},
    "p02_e02": {"host": "Alex (US-en)", "guest": "Eddy (en-US)"},
    "p02_e03": {"host": "Alex (US-en)", "guest": "Paulina (es-MX)"},
    "p03_e01": {"host": "Karen (en-AU)", "guest": "Luca (it-IT)"},
    "p03_e02": {"host": "Karen (en-AU)", "guest": "Anna (de-DE)"},
    "p03_e03": {"host": "Karen (en-AU)", "guest": "Reed (en-US)"},
    "p04_e01": {"host": "Daniel (en-GB)", "guest": "Kathy (fr-CA)"},
    "p04_e02": {"host": "Daniel (en-GB)", "guest": "Rishi (en-IN)"},
    "p04_e03": {"host": "Daniel (en-GB)", "guest": "Amelie (fr-CA)"},
    "p05_e01": {"host": "Moira (en-IE)", "guest": "Oliver (en-GB)"},
    "p05_e02": {"host": "Moira (en-IE)", "guest": "Monica (es-ES)"},
    "p05_e03": {"host": "Moira (en-IE)", "guest": "Ralph (en-US)"},
}


_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(text: str) -> list[str]:
    """Lowercase, strip non-alphanum, split into words for WER."""
    text = text.lower()
    text = _NORMALIZE_RE.sub(" ", text)
    return text.split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate between reference and hypothesis. Levenshtein over words."""
    r = _normalize(ref)
    h = _normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # DP for Levenshtein
    nr, nh = len(r), len(h)
    dp = [[0] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(nh + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # del
                dp[i][j - 1] + 1,  # ins
                dp[i - 1][j - 1] + cost,  # sub
            )
    return dp[nr][nh] / nr


def _strip_transcript_metadata(text: str) -> str:
    """Drop header/format lines so WER measures spoken content only."""
    lines = text.splitlines()
    body: list[str] = []
    speaker_re = re.compile(r"^[A-Z][A-Za-z .'\-]{0,40}:\s+")
    ts_re = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]$")
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or ts_re.fullmatch(s):
            continue
        # Drop the leading "Speaker:" prefix; keep the spoken text only
        m = speaker_re.match(s)
        body.append(m.group(0).join(s.split(m.group(0))[1:]) if m else s)
    return " ".join(body)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--transcripts-dir", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["tiny.en", "base.en"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import whisper

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    cache_dir = PROJECT_ROOT / ".cache" / "whisper"

    for model_name in args.models:
        t_model_start = time.time()
        print(f"loading {model_name}...", file=sys.stderr)
        model = whisper.load_model(model_name, download_root=str(cache_dir))
        print(f"  loaded in {time.time()-t_model_start:.1f}s", file=sys.stderr)

        for ep in args.episodes:
            audio_path = args.audio_dir / f"{ep}.mp3"
            transcript_path = args.transcripts_dir / f"{ep}.txt"
            if not audio_path.exists() or not transcript_path.exists():
                print(f"  SKIP {ep}: missing audio or transcript", file=sys.stderr)
                continue

            reference = _strip_transcript_metadata(transcript_path.read_text(encoding="utf-8"))
            t_ep = time.time()
            result = model.transcribe(str(audio_path), verbose=False, fp16=False)
            elapsed = time.time() - t_ep
            hyp = result["text"]
            ep_wer = wer(reference, hyp)
            voices = EPISODE_VOICES.get(ep, {})
            row = {
                "model": model_name,
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
                f"  {model_name:8s} {ep:8s} WER={ep_wer:.4f} "
                f"({len(_normalize(reference))}w ref / {len(_normalize(hyp))}w hyp) "
                f"voices={voices.get('host','?')}+{voices.get('guest','?')} "
                f"elapsed={elapsed:.1f}s"
            )

    # Aggregate
    from collections import defaultdict

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    summary = []
    for model_name in args.models:
        rs = by_model.get(model_name, [])
        if not rs:
            continue
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
            {"schema": "metrics_whisper_accent_wer_v1", "summary": summary, "rows": rows},
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
