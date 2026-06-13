"""Real Diarization Error Rate (DER) on v2 fixtures (#992).

Phase 3 of the diarization championship. Closes the speaker-confusion blind
spot that ``segments_per_turn_ratio`` couldn't measure.

## Approach (Path A from #992)

1. **Word-level timestamps from Deepgram nova-3**: the highest-WER backend on
   v2 (2.48% per ``EVAL_DEEPGRAM_TRANSCRIPTION_2026_06_13.md``). One call per
   episode, ~5 minutes audio, ~$0.022 each.
2. **Reference transcript** has per-line ``Speaker:`` markers but no
   timestamps. We do word-level Levenshtein DP alignment between the reference
   text and the Deepgram hypothesis text. Each reference word that aligns
   onto a hypothesis word inherits that word's start/end seconds.
3. Reference words within the same ``Speaker:`` line carry the line's speaker
   label. Contiguous same-speaker words collapse into ``(start, end, speaker)``
   ground-truth utterance segments.
4. Diarization output (already saved per backend in
   ``data/eval/runs/diarization_3way_v1/<backend>/segments_<ep>.json`` after
   the #992 instrumentation patch) is compared against the ground truth via
   ``pyannote.metrics.diarization.DiarizationErrorRate``. The metric internally
   solves the optimal speaker mapping (Hungarian) so reference labels like
   ``Maya`` and hypothesis labels like ``SPEAKER_00`` don't need to match
   by name.

## What this measures that ``segments_per_turn_ratio`` did not

Per-backend DER split into the three additive components:

- **missed_detection_time**: ground truth has speech, hypothesis says silence.
- **false_alarm_time**: hypothesis has speech, ground truth says silence.
- **confusion_time**: both agree speech is happening but disagree on speaker.
- **total_speech_time**: reference duration the rates are normalized against.

``DER = (missed + false_alarm + confusion) / total_speech``.

For podcast diarization downstream (SPOKEN_BY metadata), the
``confusion_time`` component is the one that turns "Maya said X" into
"Liam said X" in artifacts.

## Usage

    DEEPGRAM_API_KEY=...  PYTHONPATH=. .venv/bin/python \\
        scripts/eval/score/diarization_der_v1.py \\
        --audio-dir tests/fixtures/audio/v2 \\
        --transcripts-dir tests/fixtures/transcripts/v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --segments-run pyannote/MPS:data/eval/runs/diarization_3way_v1/local-mps \\
        --segments-run Gemini-2.5-flash:data/eval/runs/diarization_3way_v1/gemini \\
        --output data/eval/runs/diarization_der_v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_WORD_RE = re.compile(r"[a-z0-9']+")
_SPEAKER_LINE_RE = re.compile(r"^([A-Z][A-Za-z .'\-]{0,40}):\s+(.*)$")


# ----------------------------------------------------------------- ref parsing


def _normalize_word(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalnum() or c == "'").strip("'")


def _ref_words_with_speakers(transcript_path: Path) -> List[Tuple[str, str]]:
    """Return (normalized_word, speaker) tuples for every word in the transcript.

    Order is preserved; speaker carries forward until the next ``Speaker:``
    line. Header lines, timestamps, and empty lines are skipped.
    """
    words: List[Tuple[str, str]] = []
    current_speaker = "UNKNOWN"
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if m:
            current_speaker = m.group(1).strip()
            rest = m.group(2)
        else:
            rest = line
        for token in _WORD_RE.findall(rest.lower()):
            normalized = _normalize_word(token)
            if normalized:
                words.append((normalized, current_speaker))
    return words


# ------------------------------------------------------------ DP word align


def _align(
    ref_words: List[str],
    hyp_words: List[str],
) -> List[Tuple[int, Optional[int]]]:
    """Return list of (ref_idx, hyp_idx_or_None) for matched ref words.

    Skip ref words that don't align to any hyp word (deletions). Insertions
    (hyp words with no ref alignment) are silently dropped.
    """
    nr, nh = len(ref_words), len(hyp_words)
    # Cost table for Levenshtein (sub=1, ins=1, del=1, match=0).
    dp = [[0] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(nh + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    # Backtrack to recover the alignment path.
    pairs: List[Tuple[int, Optional[int]]] = []
    i, j = nr, nh
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Ref-word deletion: ref has a word that hyp didn't transcribe.
            pairs.append((i - 1, None))
            i -= 1
            continue
        # Insertion: hyp word with no ref counterpart.
        j -= 1
    pairs.reverse()
    return pairs


# ----------------------------------------------------- ground truth assembly


def _interpolate_timestamps(
    pairs: List[Tuple[int, Optional[int]]],
    hyp_word_times: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return per-ref-word (start, end) seconds. Linearly interpolate gaps."""
    n_ref = len(pairs)
    times: List[Optional[Tuple[float, float]]] = [None] * n_ref
    for ref_idx, hyp_idx in pairs:
        if hyp_idx is not None:
            start, end = hyp_word_times[hyp_idx]
            times[ref_idx] = (start, end)
    # Linear-interpolate ref words that didn't align to a hyp word.
    last_anchor = (0, (0.0, 0.0))
    next_anchor_idx = None
    for i in range(n_ref):
        if times[i] is not None:
            continue
        # Find the next aligned ref word.
        if next_anchor_idx is None or next_anchor_idx <= i:
            next_anchor_idx = None
            for k in range(i + 1, n_ref):
                if times[k] is not None:
                    next_anchor_idx = k
                    break
        if next_anchor_idx is None:
            # End-of-list deletion — extend the last anchor's end.
            start, end = last_anchor[1][1], last_anchor[1][1]
            times[i] = (start, end)
            continue
        # Linear interpolate between last_anchor and next anchor.
        next_times = times[next_anchor_idx]
        assert next_times is not None
        # mypy-friendly local alias
        next_start, next_end = next_times
        prev_end = last_anchor[1][1]
        gap_words = next_anchor_idx - last_anchor[0]
        if gap_words <= 0:
            gap_words = 1
        position = i - last_anchor[0]
        span = next_start - prev_end
        if span < 0:
            span = 0.0
        word_dur = span / gap_words
        start = prev_end + word_dur * (position - 1) if position > 1 else prev_end
        end = start + word_dur if word_dur > 0 else start
        times[i] = (start, end)
    final: List[Tuple[float, float]] = []
    for t in times:
        if t is None:
            final.append((0.0, 0.0))
        else:
            final.append(t)
        if t is not None:
            last_anchor = (len(final) - 1, t)
    return final


def _build_reference_segments(
    ref_words_with_speakers: List[Tuple[str, str]],
    per_word_times: List[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Collapse contiguous same-speaker ref words into (start, end, speaker) segments."""
    segments: List[Dict[str, Any]] = []
    if not ref_words_with_speakers:
        return segments
    current_speaker = ref_words_with_speakers[0][1]
    seg_start = per_word_times[0][0]
    seg_end = per_word_times[0][1]
    for i in range(1, len(ref_words_with_speakers)):
        speaker = ref_words_with_speakers[i][1]
        wstart, wend = per_word_times[i]
        if speaker == current_speaker:
            seg_end = max(seg_end, wend)
            continue
        if seg_end > seg_start:
            segments.append({"start": seg_start, "end": seg_end, "speaker": current_speaker})
        current_speaker = speaker
        seg_start, seg_end = wstart, wend
    if seg_end > seg_start:
        segments.append({"start": seg_start, "end": seg_end, "speaker": current_speaker})
    return segments


# ---------------------------------------------------------------- Deepgram


def _deepgram_word_timestamps(audio_path: Path) -> List[Dict[str, Any]]:
    """Return [{word, start, end}, ...] from Deepgram nova-3 with smart_format off."""
    from deepgram import DeepgramClient

    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY required for the word-timestamp transcription.")
    client = DeepgramClient(api_key=api_key)
    with audio_path.open("rb") as f:
        audio_bytes = f.read()
    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        smart_format=False,
        punctuate=False,
        language="en",
    )
    data = response.model_dump() if hasattr(response, "model_dump") else response.__dict__
    channels = data.get("results", {}).get("channels", [])
    if not channels:
        return []
    alts = channels[0].get("alternatives", [])
    if not alts:
        return []
    raw_words = alts[0].get("words", []) or []
    return [
        {"word": _normalize_word(w["word"]), "start": float(w["start"]), "end": float(w["end"])}
        for w in raw_words
        if w.get("word")
    ]


# ----------------------------------------------------------------- DER


def _segments_to_annotation(segments: List[Dict[str, Any]]) -> Any:
    """Wrap a list of segment dicts into a pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        ann[Segment(start, end)] = str(seg["speaker"])
    return ann


def _compute_der(
    reference_segments: List[Dict[str, Any]], hyp_segments: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Return DER + the three sub-rate seconds + total reference speech seconds."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_ann = _segments_to_annotation(reference_segments)
    hyp_ann = _segments_to_annotation(hyp_segments)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    detail = metric(ref_ann, hyp_ann, detailed=True)
    total = float(detail.get("total", 0.0) or 0.0)
    confusion = float(detail.get("confusion", 0.0) or 0.0)
    missed = float(detail.get("missed detection", 0.0) or 0.0)
    false_alarm = float(detail.get("false alarm", 0.0) or 0.0)
    der = float(detail.get("diarization error rate", 0.0) or 0.0)
    return {
        "der": der,
        "confusion_s": confusion,
        "missed_s": missed,
        "false_alarm_s": false_alarm,
        "total_speech_s": total,
    }


# ----------------------------------------------------------------- main


def _load_segments_file(segments_dir: Path, episode: str) -> Optional[List[Dict[str, Any]]]:
    path = segments_dir / f"segments_{episode}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("segments", []))


def _parse_segments_run_arg(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "--segments-run must be NAME:PATH (e.g. 'pyannote/MPS:data/eval/runs/.../local-mps')"
        )
    name, _, path = value.partition(":")
    return name.strip(), Path(path.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--transcripts-dir", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument(
        "--segments-run",
        action="append",
        type=_parse_segments_run_arg,
        required=True,
        help="Repeatable. NAME:PATH where PATH contains segments_<episode>.json files.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    gt_dir = args.output / "ground_truth"
    gt_dir.mkdir(exist_ok=True)
    dg_cache_dir = args.output / "deepgram_words"
    dg_cache_dir.mkdir(exist_ok=True)

    per_episode_results: List[Dict[str, Any]] = []

    for ep in args.episodes:
        audio_path = args.audio_dir / f"{ep}.mp3"
        transcript_path = args.transcripts_dir / f"{ep}.txt"
        if not audio_path.exists() or not transcript_path.exists():
            print(f"  SKIP {ep}: missing audio or transcript", file=sys.stderr)
            continue

        print(f"  {ep}: building ground truth…", file=sys.stderr)
        ref_pairs = _ref_words_with_speakers(transcript_path)
        ref_words = [w for w, _ in ref_pairs]

        # Cache Deepgram per-episode so the script is idempotent + cheap to rerun.
        dg_cache = dg_cache_dir / f"{ep}.json"
        if dg_cache.is_file():
            dg_words = json.loads(dg_cache.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            dg_words = _deepgram_word_timestamps(audio_path)
            elapsed = time.time() - t0
            dg_cache.write_text(json.dumps(dg_words), encoding="utf-8")
            print(f"    Deepgram: {len(dg_words)} words in {elapsed:.1f}s", file=sys.stderr)

        hyp_words = [w["word"] for w in dg_words]
        hyp_word_times = [(w["start"], w["end"]) for w in dg_words]
        pairs = _align(ref_words, hyp_words)
        per_word_times = _interpolate_timestamps(pairs, hyp_word_times)
        ref_segments = _build_reference_segments(ref_pairs, per_word_times)
        (gt_dir / f"{ep}.json").write_text(
            json.dumps(
                {
                    "episode_id": ep,
                    "ref_word_count": len(ref_words),
                    "hyp_word_count": len(hyp_words),
                    "aligned_word_count": sum(1 for _, h in pairs if h is not None),
                    "segments": ref_segments,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"    GT: {len(ref_segments)} ref segments / aligned "
            f"{sum(1 for _, h in pairs if h is not None)}/{len(ref_words)} words",
            file=sys.stderr,
        )

        ep_row: Dict[str, Any] = {
            "episode_id": ep,
            "ref_segments": len(ref_segments),
            "by_backend": {},
        }
        for backend_name, segments_dir in args.segments_run:
            hyp_segments = _load_segments_file(segments_dir, ep)
            if not hyp_segments:
                print(
                    f"    {backend_name}: no segments file for {ep} at {segments_dir}",
                    file=sys.stderr,
                )
                continue
            metrics = _compute_der(ref_segments, hyp_segments)
            ep_row["by_backend"][backend_name] = metrics
            print(
                f"    {backend_name}: DER={metrics['der']:.3f} "
                f"(conf={metrics['confusion_s']:.1f}s / miss={metrics['missed_s']:.1f}s / "
                f"fa={metrics['false_alarm_s']:.1f}s of {metrics['total_speech_s']:.1f}s)",
                file=sys.stderr,
            )
        per_episode_results.append(ep_row)

    # Aggregate per backend (micro-average: pool seconds across episodes, then divide).
    by_backend_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"confusion_s": 0.0, "missed_s": 0.0, "false_alarm_s": 0.0, "total_speech_s": 0.0}
    )
    for ep_row in per_episode_results:
        for backend_name, metrics in ep_row["by_backend"].items():
            for key in ("confusion_s", "missed_s", "false_alarm_s", "total_speech_s"):
                by_backend_totals[backend_name][key] += metrics[key]

    summary = {}
    for backend_name, totals in by_backend_totals.items():
        total = totals["total_speech_s"] or 1.0
        der = (totals["confusion_s"] + totals["missed_s"] + totals["false_alarm_s"]) / total
        summary[backend_name] = {
            "der": der,
            "confusion_rate": totals["confusion_s"] / total,
            "missed_rate": totals["missed_s"] / total,
            "false_alarm_rate": totals["false_alarm_s"] / total,
            "total_speech_s": totals["total_speech_s"],
        }

    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "metrics_diarization_der_v1",
                "summary": summary,
                "rows": per_episode_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"{'Backend':<24} {'DER':>7} {'Conf':>7} {'Miss':>7} {'FA':>7} {'Total':>8}")
    for backend_name, s in summary.items():
        print(
            f"{backend_name:<24} "
            f"{s['der']:>6.2%} "
            f"{s['confusion_rate']:>6.2%} "
            f"{s['missed_rate']:>6.2%} "
            f"{s['false_alarm_rate']:>6.2%} "
            f"{s['total_speech_s']:>7.1f}s"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
