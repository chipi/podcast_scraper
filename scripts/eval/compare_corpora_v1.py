#!/usr/bin/env python3
"""Compare two corpora built by different stacks — e.g. cloud (prod-v2) vs DGX-local (prod-v3).

The question this exists to answer is **quality parity**: can the local stack match the cloud one,
stage by stage? So it reports per-stage metrics on the episodes the two corpora share, joined by
episode GUID, and says nothing about stages that use the same model in both (there is nothing to
compare there).

Everything here is **deterministic and free** — no LLM judge is called. The metrics are the ones a
machine can check without an opinion:

  transcript   timeline error (does the transcript end where the audio does? — #1173), length
  diarization  voices found, how many resolved to real names vs left as SPEAKER_NN (#1167)
  insights     count, attribution, and timestamp range-validity
  kg           nodes/edges, and unresolved person placeholders leaking in as entities
  coverage     episodes that actually produced each artifact

``timeline_error_pct`` is the metric that catches #1173-style drift; ``quote_ts_valid_pct`` is
NOT. The latter only asks whether a timestamp falls inside ``[0, duration]``, and a quote drifted
by a minute still does — it scored 99.6% on the very corpus whose timestamps were broken. Read it
as a range sanity check, never as evidence that the timestamps point at the right moment.

Summary/insight *text* quality is deliberately NOT scored here: that needs a judge panel from a
vendor disjoint from both candidates (see autoresearch/JUDGING.md), which costs money and is a
separate, explicit step. This gives the free half of the picture first.

Usage
-----

    python scripts/eval/compare_corpora_v1.py \
        --a .test_outputs/manual/prod-v2/corpus --a-label cloud \
        --b .test_outputs/manual/prod-v3/corpus --b-label dgx \
        --out docs/wip/CORPUS_COMPARE_V2_V3.md

Exit codes: 0 ok · 2 input error
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# A quote/insight timestamp must land inside the episode. Anything outside is broken by definition.
_TIMESTAMP_SLACK_S = 5.0


def _audio_duration(path: Path) -> Optional[float]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _latest_runs(corpus: Path) -> List[Path]:
    """Newest run per feed — what the corpus actually serves."""
    runs = []
    for feed in sorted(corpus.glob("feeds/*")):
        feed_runs = sorted(feed.glob("run_*"))
        if feed_runs:
            runs.append(feed_runs[-1])
    return runs


def _episodes(corpus: Path) -> Dict[str, Dict[str, Path]]:
    """guid -> {metadata, transcript, segments, gi, kg, media} for the newest run of each feed."""
    out: Dict[str, Dict[str, Path]] = {}
    for run in _latest_runs(corpus):
        for meta_path in sorted((run / "metadata").glob("*.metadata.json")):
            meta = _load_json(meta_path)
            if not isinstance(meta, dict):
                continue
            guid = (meta.get("episode") or {}).get("guid")
            if not guid:
                continue
            stem = meta_path.name[: -len(".metadata.json")]
            entry = {"metadata": meta_path}
            for key, rel in (
                ("transcript", f"transcripts/{stem}.txt"),
                ("segments", f"transcripts/{stem}.segments.json"),
                ("gi", f"metadata/{stem}.gi.json"),
                ("kg", f"metadata/{stem}.kg.json"),
                ("media", f"media/{stem}.mp3"),
            ):
                p = run / rel
                if p.is_file():
                    entry[key] = p
            out[guid] = entry
    return out


def _is_placeholder(name: str) -> bool:
    n = str(name or "").strip().lower()
    return n.startswith("speaker_") or n.startswith("speaker-") or n.startswith("spk_")


def _score_episode(entry: Dict[str, Path], duration: Optional[float]) -> Dict[str, Any]:
    """Deterministic quality signals for one episode."""
    rec: Dict[str, Any] = {}

    segments = _load_json(entry["segments"]) if "segments" in entry else None
    if isinstance(segments, list) and segments:
        try:
            span = float(segments[-1]["end"])
        except (KeyError, TypeError, ValueError):
            span = 0.0
        rec["n_segments"] = len(segments)
        rec["transcript_span_s"] = round(span, 1)
        if duration:
            # The transcript should end where the audio does. A transcript that stops well short
            # is the #1173 signature: it was cut from silence-stripped (shorter) audio.
            rec["timeline_error_pct"] = round(100.0 * (duration - span) / duration, 2)

        voices = {str(s.get("speaker_label") or s.get("speaker") or "") for s in segments}
        voices.discard("")
        if voices:
            rec["voices"] = len(voices)
            rec["voices_named"] = sum(1 for v in voices if not _is_placeholder(v))

    gi = _load_json(entry["gi"]) if "gi" in entry else None
    if isinstance(gi, dict):
        nodes = gi.get("nodes") or []
        quotes = [n for n in nodes if n.get("type") == "Quote"]
        insights = [n for n in nodes if n.get("type") == "Insight"]
        rec["insights"] = len(insights)
        rec["quotes"] = len(quotes)
        if quotes:
            valid = attributed = 0
            for q in quotes:
                props = q.get("properties") or {}
                ts = props.get("timestamp_start_ms")
                if isinstance(ts, (int, float)) and ts > 0:
                    # A timestamp is only meaningful if it lands inside the episode.
                    if duration is None or ts / 1000.0 <= duration + _TIMESTAMP_SLACK_S:
                        valid += 1
                if props.get("speaker_id"):
                    attributed += 1
            rec["quote_ts_valid_pct"] = round(100.0 * valid / len(quotes), 1)
            rec["quote_attributed_pct"] = round(100.0 * attributed / len(quotes), 1)

    kg = _load_json(entry["kg"]) if "kg" in entry else None
    if isinstance(kg, dict):
        nodes = kg.get("nodes") or []
        persons = [n for n in nodes if n.get("type") == "Person"]
        rec["kg_nodes"] = len(nodes)
        rec["kg_edges"] = len(kg.get("edges") or [])
        rec["kg_persons"] = len(persons)
        rec["kg_person_placeholders"] = sum(
            1 for p in persons if _is_placeholder((p.get("properties") or {}).get("name", ""))
        )

    meta = _load_json(entry["metadata"])
    if isinstance(meta, dict):
        content = meta.get("content") or {}
        summary = content.get("summary") or meta.get("summary")
        rec["has_summary"] = bool(summary)
        rec["summary_chars"] = len(str(summary)) if summary else 0

    return rec


def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of each metric over the episodes that reported it (never invent a zero)."""
    keys: set = set()
    for s in scores:
        keys |= set(s)
    agg: Dict[str, Any] = {"episodes": len(scores)}
    for k in sorted(keys):
        vals = [s[k] for s in scores if isinstance(s.get(k), (int, float, bool))]
        if not vals:
            continue
        agg[k] = round(sum(float(v) for v in vals) / len(vals), 2)
        agg[f"{k}__n"] = len(vals)
    return agg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", type=Path, required=True, help="Corpus A root (baseline, e.g. cloud)")
    ap.add_argument("--b", type=Path, required=True, help="Corpus B root (challenger, e.g. DGX)")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--out", type=Path, help="Write a markdown scorecard here")
    args = ap.parse_args(argv)

    for root in (args.a, args.b):
        if not root.is_dir():
            print(f"error: not a corpus root: {root}", file=sys.stderr)
            return 2

    eps_a, eps_b = _episodes(args.a), _episodes(args.b)
    shared = sorted(set(eps_a) & set(eps_b))
    if not shared:
        print("error: the two corpora share no episode GUIDs", file=sys.stderr)
        return 2

    print(
        f"{args.a_label}: {len(eps_a)} eps | {args.b_label}: {len(eps_b)} | shared: {len(shared)}"
    )

    scores_a: List[Dict[str, Any]] = []
    scores_b: List[Dict[str, Any]] = []
    for guid in shared:
        # Same audio in both corpora — measure duration once.
        media = eps_a[guid].get("media") or eps_b[guid].get("media")
        duration = _audio_duration(media) if media else None
        scores_a.append(_score_episode(eps_a[guid], duration))
        scores_b.append(_score_episode(eps_b[guid], duration))

    agg_a, agg_b = _aggregate(scores_a), _aggregate(scores_b)

    rows = []
    for key in sorted(set(agg_a) | set(agg_b)):
        if key.endswith("__n") or key == "episodes":
            continue
        va, vb = agg_a.get(key), agg_b.get(key)
        na, nb = agg_a.get(f"{key}__n", 0), agg_b.get(f"{key}__n", 0)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = f"{vb - va:+.2f}"
        rows.append((key, va, na, vb, nb, delta))

    lines = [
        f"# Corpus comparison — {args.a_label} vs {args.b_label}",
        "",
        f"Shared episodes (joined by GUID): **{len(shared)}**",
        "",
        "Deterministic metrics only — no LLM judge. Summary/insight *text* quality needs a",
        "cross-vendor judge panel (autoresearch/JUDGING.md) and is scored separately.",
        "",
        f"| metric | {args.a_label} | n | {args.b_label} | n | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, va, na, vb, nb, delta in rows:
        lines.append(
            f"| `{key}` | {va if va is not None else '—'} | {na} | "
            f"{vb if vb is not None else '—'} | {nb} | {delta} |"
        )
    lines += [
        "",
        "## How to read the load-bearing rows",
        "",
        "- **`timeline_error_pct`** — how far short of the audio the transcript ends. This is the",
        "  #1173 signature: a corpus transcribed from silence-stripped audio ends early, and every",
        "  timestamp derived from it is wrong. Near 0 is correct.",
        "- **`quote_ts_valid_pct`** — a *range* check only: does the timestamp fall inside",
        "  `[0, duration]`? A quote drifted by a minute still does, so this cannot see drift. It",
        "  scored 99.6% on the corpus whose timestamps were broken. Never read it as accuracy.",
        "- **`voices_named`** vs **`voices`** — voices resolved to a real person rather than left",
        "  as `SPEAKER_NN`. Unresolved placeholders must never reach an entity surface (#1167).",
        "- **`kg_person_placeholders`** — should be 0. Anything above it is a leak.",
        "",
    ]
    report = "\n".join(lines)

    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
