"""Count + DER for the v3 fixtures against EXACT RTTM ground truth (#1170).

Unlike ``diarization_der_v1.py`` (which reconstructs v2 ground truth from a paid
Deepgram transcription + Levenshtein alignment), the v3 fixtures carry an exact
per-turn reference RTTM emitted from the deterministic ``say`` timeline
(``transcripts_to_mp3.py --rttm-only``). DER here needs no transcription and no
alignment — it is free and exact.

Reports, per episode and pooled across a run, for each diarization backend:

- **count**: detected distinct speakers vs ``expected_diarized_voices`` (the
  brittle count metric — a 1s cameo miss costs a full -1);
- **DER** + components (confusion / missed / false-alarm), which weight error by
  mis-attributed TIME, so a good-but-cameo-missing run is not punished as harshly.

Count and DER are COMPLEMENTARY — report both, never swap one for the other.

The hypothesis for each backend is a directory of ``segments_<episode>.json``
files (``{"segments": [{"start", "end", "speaker"}, ...]}``) — the same shape
``diarization_der_v1.py`` consumes, produced by the run harness.

Usage::

    .venv/bin/python scripts/eval/score/diarization_der_rttm_v1.py \\
        --rttm-dir tests/fixtures/transcripts/v3 \\
        --segments-run 3.1:data/eval/runs/diar_v3/pyannote-3.1 \\
        --segments-run community-1:data/eval/runs/diar_v3/community-1 \\
        --output data/eval/runs/diar_v3_der
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _rttm_to_segments(path: Path) -> List[Dict[str, Any]]:
    """Parse a NIST RTTM into ``[{start, end, speaker}]`` segments."""
    segments: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER" or len(parts) < 8:
            continue
        start = float(parts[3])
        end = start + float(parts[4])
        segments.append({"start": start, "end": end, "speaker": parts[7]})
    return segments


def _segments_to_annotation(segments: List[Dict[str, Any]]) -> Any:
    """Wrap ``[{start, end, speaker}]`` into a pyannote Annotation (skip empty spans)."""
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        # Unique key per row so adjacent same-label turns are both retained.
        ann[Segment(start, end), i] = str(seg["speaker"])
    return ann


def _compute_der(
    reference_segments: List[Dict[str, Any]], hyp_segments: List[Dict[str, Any]]
) -> Dict[str, float]:
    """DER + the three sub-rate seconds + total reference speech seconds.

    ``DiarizationErrorRate`` solves the optimal speaker mapping internally, so
    reference voice labels and hypothesis ``SPEAKER_NN`` labels need not match.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_ann = _segments_to_annotation(reference_segments)
    hyp_ann = _segments_to_annotation(hyp_segments)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    detail = metric(ref_ann, hyp_ann, detailed=True)
    total = float(detail.get("total", 0.0) or 0.0)
    return {
        "der": float(detail.get("diarization error rate", 0.0) or 0.0),
        "confusion_s": float(detail.get("confusion", 0.0) or 0.0),
        "missed_s": float(detail.get("missed detection", 0.0) or 0.0),
        "false_alarm_s": float(detail.get("false alarm", 0.0) or 0.0),
        "total_speech_s": total,
    }


def _distinct_speakers(segments: List[Dict[str, Any]]) -> int:
    return len({str(s["speaker"]) for s in segments if float(s["end"]) > float(s["start"])})


def _load_hyp(segments_dir: Path, episode: str) -> Optional[List[Dict[str, Any]]]:
    path = segments_dir / f"segments_{episode}.json"
    if not path.is_file():
        return None
    return list(json.loads(path.read_text(encoding="utf-8")).get("segments", []))


def _expected_count(rttm_dir: Path, episode: str) -> Optional[int]:
    sidecar = rttm_dir / f"{episode}.groundtruth.json"
    if not sidecar.is_file():
        return None
    return int(json.loads(sidecar.read_text(encoding="utf-8"))["expected_diarized_voices"])


def _parse_segments_run_arg(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("--segments-run must be NAME:PATH")
    name, _, path = value.partition(":")
    return name.strip(), Path(path.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rttm-dir", type=Path, required=True, help="dir with <ep>.rttm + sidecars")
    p.add_argument("--episodes", nargs="*", help="default: every <ep>.rttm in --rttm-dir")
    p.add_argument(
        "--segments-run",
        action="append",
        type=_parse_segments_run_arg,
        required=True,
        help="Repeatable NAME:PATH; PATH holds segments_<episode>.json files.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    episodes = args.episodes or sorted(f.stem for f in args.rttm_dir.glob("*.rttm"))
    if not episodes:
        print(f"no episodes (no *.rttm in {args.rttm_dir})", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for ep in episodes:
        rttm = args.rttm_dir / f"{ep}.rttm"
        if not rttm.is_file():
            print(f"  SKIP {ep}: no RTTM", file=sys.stderr)
            continue
        ref = _rttm_to_segments(rttm)
        expected = _expected_count(args.rttm_dir, ep)
        row: Dict[str, Any] = {"episode_id": ep, "expected": expected, "by_backend": {}}
        for name, seg_dir in args.segments_run:
            hyp = _load_hyp(seg_dir, ep)
            if hyp is None:
                print(f"    {name}: no segments for {ep} in {seg_dir}", file=sys.stderr)
                continue
            detected = _distinct_speakers(hyp)
            metrics = _compute_der(ref, hyp)
            metrics["detected"] = detected
            metrics["count_match"] = expected is not None and detected == expected
            row["by_backend"][name] = metrics
            print(
                f"  {ep} [{name}]: count {detected}/{expected} "
                f"{'OK' if metrics['count_match'] else 'X'}  DER={metrics['der']:.3f}",
                file=sys.stderr,
            )
        rows.append(row)

    # Aggregate: micro-average DER (pool seconds) + count-match rate per backend.
    totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"confusion_s": 0.0, "missed_s": 0.0, "false_alarm_s": 0.0, "total_speech_s": 0.0}
    )
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"match": 0, "scored": 0})
    for row in rows:
        for name, m in row["by_backend"].items():
            for k in ("confusion_s", "missed_s", "false_alarm_s", "total_speech_s"):
                totals[name][k] += m[k]
            counts[name]["scored"] += 1
            counts[name]["match"] += 1 if m["count_match"] else 0

    summary: Dict[str, Any] = {}
    for name, t in totals.items():
        denom = t["total_speech_s"] or 1.0
        scored = counts[name]["scored"] or 1
        summary[name] = {
            "der": (t["confusion_s"] + t["missed_s"] + t["false_alarm_s"]) / denom,
            "confusion_rate": t["confusion_s"] / denom,
            "missed_rate": t["missed_s"] / denom,
            "false_alarm_rate": t["false_alarm_s"] / denom,
            "count_match": counts[name]["match"],
            "count_scored": counts[name]["scored"],
            "count_match_rate": counts[name]["match"] / scored,
            "total_speech_s": t["total_speech_s"],
        }

    (args.output / "metrics.json").write_text(
        json.dumps(
            {"schema": "metrics_diarization_der_rttm_v1", "summary": summary, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"{'Backend':<16} {'Count':>10} {'DER':>7} {'Conf':>7} {'Miss':>7} {'FA':>7}")
    for name, s in summary.items():
        count_str = f"{s['count_match']}/{s['count_scored']}"
        print(
            f"{name:<16} "
            f"{count_str:>10} "
            f"{s['der']:>6.2%} "
            f"{s['confusion_rate']:>6.2%} "
            f"{s['missed_rate']:>6.2%} "
            f"{s['false_alarm_rate']:>6.2%}"
        )
    print(f"wrote {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
