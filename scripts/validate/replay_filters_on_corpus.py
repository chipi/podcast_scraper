#!/usr/bin/env python3
"""#652 Layer 1 validation — replay post-extraction filters on an existing corpus.

Walks a corpus (multi-feed or single-feed layout) and runs the four
deterministic filters from ``gi.filters`` + ``kg.filters`` against each
episode's stored ``gi.json`` + ``kg.json`` + transcript WITHOUT rewriting
anything. Reports how many insights / topics / entities would be dropped
or normalised, so we know whether the filters have real-world impact
before committing to a live pipeline re-run.

Zero cloud cost. Instant.

Usage
-----

    python scripts/validate/replay_filters_on_corpus.py <corpus_root>

    # JSON output for piping:
    python scripts/validate/replay_filters_on_corpus.py <corpus_root> --json

Exit codes
----------
- 0 — scan complete (includes "no changes" result).
- 1 — no artifacts found.
- 2 — script/input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from podcast_scraper.gi.filters import apply_insight_filters  # noqa: E402
from podcast_scraper.kg.filters import (  # noqa: E402
    normalize_topic_labels,
    repair_entity_kind,
)


def _iter_gi_paths(corpus_root: Path) -> List[Path]:
    feeds_root = corpus_root / "feeds"
    if feeds_root.is_dir():
        return sorted(feeds_root.glob("*/run_*/metadata/*.gi.json"))
    return sorted(corpus_root.glob("run_*/metadata/*.gi.json"))


def _kg_path_for_gi(gi_path: Path) -> Optional[Path]:
    stem = gi_path.name[: -len(".gi.json")]
    candidate = gi_path.parent / f"{stem}.kg.json"
    return candidate if candidate.is_file() else None


def _transcript_for_gi(gi_path: Path) -> Optional[str]:
    """Transcript sits in the run's ``transcripts/`` sibling dir."""
    run_dir = gi_path.parent.parent
    transcripts_dir = run_dir / "transcripts"
    if not transcripts_dir.is_dir():
        return None
    # Match by basename stem (strip .gi.json, try .txt).
    stem = gi_path.name[: -len(".gi.json")]
    candidate = transcripts_dir / f"{stem}.txt"
    if candidate.is_file():
        try:
            return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    # Fall back to any .txt in the dir whose name starts with the episode ordinal.
    parts = stem.split(" - ", 1)
    if parts and parts[0].isdigit():
        for tx in transcripts_dir.glob(f"{parts[0]}*.txt"):
            try:
                return tx.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _insights_from_gi(gi: Dict[str, Any]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    for node in gi.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "Insight":
            continue
        props = node.get("properties") or {}
        insights.append(
            {
                "text": str(props.get("text") or "").strip(),
                "quote_text": props.get("quote_text") or props.get("quote"),
                "id": node.get("id"),
            }
        )
    return insights


def _topics_from_kg(kg: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for node in kg.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "Topic":
            continue
        props = node.get("properties") or {}
        label = str(props.get("label") or "").strip()
        if label:
            out.append(label)
    return out


def _entities_from_kg(kg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in kg.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "Entity":
            continue
        props = node.get("properties") or {}
        name = str(props.get("name") or "").strip()
        kind = str(props.get("kind") or "").strip().lower()
        if name:
            out.append({"name": name, "kind": kind})
    return out


def _apply_one(
    gi_path: Path,
) -> Optional[Dict[str, int]]:
    gi = _load_json(gi_path)
    if not gi:
        return None
    kg_path = _kg_path_for_gi(gi_path)
    kg = _load_json(kg_path) if kg_path else {}

    # Insights + ad/dialogue filters.
    insights = _insights_from_gi(gi)
    # NB: production wiring (gi/pipeline.py) calls apply_insight_filters
    # WITHOUT transcript_window_by_index, so the ad filter scans only the
    # insight text itself. We mirror that here. An earlier version of this
    # harness passed the FULL transcript per insight, but that produced
    # episode-level over-filtering: any episode containing ad content
    # anywhere in its transcript would lose ALL its insights, which is
    # incorrect. The right narrow-window design needs quote char offsets,
    # which we don't have at the gi.json layer.
    _kept, ads_dropped, dialogue_dropped = apply_insight_filters(insights)

    # Topics + normalizer.
    topics = _topics_from_kg(kg or {})
    _norm, topics_changed = normalize_topic_labels(topics)

    # Entities + kind repair.
    entities = _entities_from_kg(kg or {})
    _rep, entities_repaired = repair_entity_kind(entities)

    return {
        "insights": len(insights),
        "topics": len(topics),
        "entities": len(entities),
        "ads_dropped": ads_dropped,
        "dialogue_dropped": dialogue_dropped,
        "topics_changed": topics_changed,
        "entities_repaired": entities_repaired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="replay_filters_on_corpus",
        description="Dry-run #652 filters on an existing corpus (no writes).",
    )
    parser.add_argument("corpus_root", help="Corpus root (parent of feeds/ or run_*/).")
    parser.add_argument("--json", action="store_true", help="Emit result as JSON on stdout.")
    args = parser.parse_args()

    root = Path(args.corpus_root).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: corpus path does not exist: {root}", file=sys.stderr)
        return 2

    gi_paths = _iter_gi_paths(root)
    if not gi_paths:
        print(f"No gi.json artifacts found under {root}.", file=sys.stderr)
        return 1

    totals = {
        "insights": 0,
        "topics": 0,
        "entities": 0,
        "ads_dropped": 0,
        "dialogue_dropped": 0,
        "topics_changed": 0,
        "entities_repaired": 0,
        "episodes_scanned": 0,
        "episodes_skipped": 0,
    }
    per_episode: List[Tuple[Path, Dict[str, int]]] = []

    for gi_path in gi_paths:
        result = _apply_one(gi_path)
        if result is None:
            totals["episodes_skipped"] += 1
            continue
        totals["episodes_scanned"] += 1
        for k, v in result.items():
            totals[k] += v
        per_episode.append((gi_path, result))

    if args.json:
        print(json.dumps({"totals": totals}, indent=2, sort_keys=True))
        return 0

    # Human-readable report.
    print(f"#652 filter replay — corpus: {root}")
    print(
        f"  Episodes scanned: {totals['episodes_scanned']}  "
        f"skipped: {totals['episodes_skipped']}"
    )
    print()
    print("Totals across corpus:")
    print(f"  Insights:               {totals['insights']:>6d}")
    print(
        f"    → ads dropped:        {totals['ads_dropped']:>6d}  "
        f"({_pct(totals['ads_dropped'], totals['insights'])})"
    )
    print(
        f"    → dialogue dropped:   {totals['dialogue_dropped']:>6d}  "
        f"({_pct(totals['dialogue_dropped'], totals['insights'])})"
    )
    print(f"  Topics:                 {totals['topics']:>6d}")
    print(
        f"    → normalized:         {totals['topics_changed']:>6d}  "
        f"({_pct(totals['topics_changed'], totals['topics'])})"
    )
    print(f"  Entities:               {totals['entities']:>6d}")
    print(
        f"    → kind repaired:      {totals['entities_repaired']:>6d}  "
        f"({_pct(totals['entities_repaired'], totals['entities'])})"
    )

    # Top-10 episodes by total filter activity.
    if per_episode:
        per_episode.sort(
            key=lambda t: (
                t[1]["ads_dropped"]
                + t[1]["dialogue_dropped"]
                + t[1]["topics_changed"]
                + t[1]["entities_repaired"]
            ),
            reverse=True,
        )
        print()
        print("Top 10 episodes by filter activity:")
        for path, r in per_episode[:10]:
            rel = path.relative_to(root)
            print(
                f"  {rel}: ads={r['ads_dropped']}, "
                f"dialogue={r['dialogue_dropped']}, "
                f"topics={r['topics_changed']}, "
                f"entities={r['entities_repaired']}"
            )

    return 0


def _pct(n: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * n / total:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
