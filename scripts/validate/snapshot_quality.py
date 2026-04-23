#!/usr/bin/env python3
"""#657 Part B — end-of-phase quality snapshot.

Reads a corpus and emits a dated JSON summarising the load-bearing quality
signals so future tickets can diff against a baseline.

The snapshot composes existing reports — it does not recompute anything from
scratch:

- Top-20 canonical topics from ``search/topic_clusters.json`` (#655).
- Top-10 insight clusters from ``search/insight_clusters.json`` (#600).
- Bridge ``{both, gi_only, kg_only}`` distribution on every
  ``*.gi.json`` / ``*.kg.json`` pair in the corpus (#654).
- #652 filter impact: sponsor-ad share, dialogue rate, topic-normalization
  rate, entity-kind-repair rate — replays the four filters via the existing
  ``gi.filters`` / ``kg.filters`` modules.
- Cost summary from ``corpus_manifest.json::cost_rollup`` (#650).
- GIL + KG quality bundles from ``make gil-quality-metrics`` /
  ``make kg-quality-metrics`` (PRD-017 / PRD-019).
- Git SHA, snapshot date, corpus root.

Missing inputs degrade gracefully — the relevant field gets
``"status": "not-built"`` rather than crashing the whole snapshot.

Usage::

    python scripts/validate/snapshot_quality.py <corpus_root>
    # writes {corpus_root}/_quality_snapshot_<YYYY-MM-DD>.json

    # or point the output elsewhere for committed baselines:
    python scripts/validate/snapshot_quality.py <corpus_root> \\
        --output data/quality_snapshots/my-manual-run4_2026-04-23.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

SCHEMA_VERSION = "1.0.0"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _load_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_search_dir(corpus_root: Path) -> Optional[Path]:
    """Corpus-level ``search/`` dir holds topic/insight clusters."""
    direct = corpus_root / "search"
    if direct.is_dir():
        return direct
    # Corpus may also live under feeds/<slug>/run_*/search but we expect
    # corpus-level post-backfill clusters when topic-clusters CLI was run.
    return None


def _top20_topic_clusters(search_dir: Optional[Path]) -> Dict[str, Any]:
    if search_dir is None:
        return {"status": "no-search-dir"}
    data = _load_json_or_none(search_dir / "topic_clusters.json")
    if data is None:
        return {"status": "not-built", "expected": str(search_dir / "topic_clusters.json")}
    clusters = sorted(data.get("clusters") or [], key=lambda c: -c.get("member_count", 0))
    top20 = []
    for c in clusters[:20]:
        members = c.get("members") or []
        top20.append(
            {
                "canonical_label": c.get("canonical_label"),
                "member_count": c.get("member_count", len(members)),
                "aliases": sorted({m.get("label") for m in members if m.get("label")}),
            }
        )
    return {
        "status": "ok",
        "threshold": data.get("threshold"),
        "cluster_count": data.get("cluster_count"),
        "topic_count": data.get("topic_count"),
        "singletons": data.get("singletons"),
        "top20": top20,
    }


def _top10_insight_clusters(search_dir: Optional[Path]) -> Dict[str, Any]:
    if search_dir is None:
        return {"status": "no-search-dir"}
    data = _load_json_or_none(search_dir / "insight_clusters.json")
    if data is None:
        return {"status": "not-built", "expected": str(search_dir / "insight_clusters.json")}
    clusters = data.get("clusters") or []
    # insight_clusters schema uses "items" for members.
    sorted_clusters = sorted(clusters, key=lambda c: -len(c.get("items") or []))
    top10 = []
    for c in sorted_clusters[:10]:
        items = c.get("items") or []
        top10.append(
            {
                "cluster_id": c.get("cluster_id"),
                "label": c.get("label"),
                "member_count": len(items),
                "sample_insights": [i.get("text")[:140] for i in items[:3] if i.get("text")],
            }
        )
    return {
        "status": "ok",
        "cluster_count": len(clusters),
        "top10": top10,
    }


def _bridge_distribution(corpus_root: Path) -> Dict[str, Any]:
    try:
        from podcast_scraper.builders.bridge_builder import build_bridge
    except Exception as exc:
        return {"status": "import-error", "detail": str(exc)}

    gi_paths = list(corpus_root.rglob("*.gi.json"))
    if not gi_paths:
        return {"status": "no-gi-artifacts"}

    both = gi_only = kg_only = 0
    episodes_scanned = 0
    for gi_path in gi_paths:
        kg_path = gi_path.with_name(gi_path.name.replace(".gi.json", ".kg.json"))
        if not kg_path.exists():
            continue
        try:
            gi = json.loads(gi_path.read_text(encoding="utf-8"))
            kg = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = build_bridge("episode:snapshot", gi, kg, fuzzy_reconcile=False)
        for ident in result.get("identities") or []:
            src = ident.get("sources") or {}
            if src.get("gi") and src.get("kg"):
                both += 1
            elif src.get("gi"):
                gi_only += 1
            elif src.get("kg"):
                kg_only += 1
        episodes_scanned += 1
    total = both + gi_only + kg_only
    return {
        "status": "ok",
        "episodes_scanned": episodes_scanned,
        "both": both,
        "gi_only": gi_only,
        "kg_only": kg_only,
        "total_identities": total,
        "pct_both": round(100.0 * both / total, 2) if total else 0.0,
        "pct_gi_only": round(100.0 * gi_only / total, 2) if total else 0.0,
        "pct_kg_only": round(100.0 * kg_only / total, 2) if total else 0.0,
    }


def _filter_impact(corpus_root: Path) -> Dict[str, Any]:
    """Replay the 4 #652 filters on the corpus without touching the files."""
    try:
        from podcast_scraper.gi.filters import apply_insight_filters
        from podcast_scraper.kg.filters import normalize_topic_labels, repair_entity_kind
    except Exception as exc:
        return {"status": "import-error", "detail": str(exc)}

    insights_total = topics_total = entities_total = 0
    ads_dropped = dialogue_dropped = topics_changed = entities_repaired = 0
    kg_topic_length_histogram: Dict[int, int] = {}
    entity_kind_histogram: Dict[str, int] = {}

    for gi_path in corpus_root.rglob("*.gi.json"):
        try:
            gi = json.loads(gi_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        insights = []
        for n in gi.get("nodes") or []:
            if n.get("type") != "Insight":
                continue
            props = n.get("properties") or {}
            insights.append(
                {
                    "text": (props.get("text") or "").strip(),
                    "quote_text": props.get("quote_text") or props.get("quote"),
                }
            )
        if not insights:
            continue
        insights_total += len(insights)
        _, ads, dialogue = apply_insight_filters(insights)
        ads_dropped += ads
        dialogue_dropped += dialogue

    for kg_path in corpus_root.rglob("*.kg.json"):
        try:
            kg = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        topics: List[str] = []
        entities: List[Dict[str, Any]] = []
        for n in kg.get("nodes") or []:
            props = n.get("properties") or {}
            if n.get("type") == "Topic":
                label = (props.get("label") or "").strip()
                if label:
                    topics.append(label)
                    tokens = len(label.split())
                    kg_topic_length_histogram[tokens] = kg_topic_length_histogram.get(tokens, 0) + 1
            elif n.get("type") == "Entity":
                entities.append(
                    {
                        "name": (props.get("name") or "").strip(),
                        "kind": (props.get("kind") or "").strip().lower(),
                    }
                )
                kind = (props.get("kind") or "unknown").strip().lower()
                entity_kind_histogram[kind] = entity_kind_histogram.get(kind, 0) + 1
        topics_total += len(topics)
        entities_total += len(entities)
        _, tchanges = normalize_topic_labels(topics)
        _, erepairs = repair_entity_kind(entities)
        topics_changed += tchanges
        entities_repaired += erepairs

    def _pct(n: int, total: int) -> float:
        return round(100.0 * n / total, 2) if total else 0.0

    return {
        "status": "ok",
        "insights_total": insights_total,
        "ads_dropped": ads_dropped,
        "ads_dropped_pct": _pct(ads_dropped, insights_total),
        "dialogue_dropped": dialogue_dropped,
        "dialogue_dropped_pct": _pct(dialogue_dropped, insights_total),
        "topics_total": topics_total,
        "topics_normalized": topics_changed,
        "topics_normalized_pct": _pct(topics_changed, topics_total),
        "entities_total": entities_total,
        "entities_kind_repaired": entities_repaired,
        "entities_kind_repaired_pct": _pct(entities_repaired, entities_total),
        "kg_topic_length_histogram": dict(sorted(kg_topic_length_histogram.items())),
        "entity_kind_histogram": dict(sorted(entity_kind_histogram.items(), key=lambda kv: -kv[1])),
    }


def _cost_rollup(corpus_root: Path) -> Dict[str, Any]:
    manifest_path = corpus_root / "corpus_manifest.json"
    manifest = _load_json_or_none(manifest_path)
    if manifest is None:
        return {"status": "not-built", "expected": str(manifest_path)}
    rollup = manifest.get("cost_rollup")
    if not isinstance(rollup, dict):
        return {"status": "missing-cost-rollup"}
    return {"status": "ok", **rollup}


def _run_quality_metric_target(target: str, corpus_root: Path) -> Dict[str, Any]:
    """Invoke ``make <target> DIR=<corpus_root> ARGS='--json'`` and parse JSON."""
    try:
        out = subprocess.run(
            ["make", target, f"DIR={corpus_root}", "ARGS=--json"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "make-not-available"}
    if out.returncode != 0:
        return {
            "status": "make-failed",
            "returncode": out.returncode,
            "stderr_tail": (out.stderr or "")[-400:],
        }
    # Extract the JSON blob from stdout (there's a pre-amble comment line).
    stdout = out.stdout or ""
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"status": "no-json-in-output", "stdout_tail": stdout[-400:]}
    try:
        return {"status": "ok", **json.loads(stdout[start : end + 1])}
    except json.JSONDecodeError as exc:
        return {"status": "json-parse-error", "detail": str(exc)}


def build_snapshot(corpus_root: Path) -> Dict[str, Any]:
    search_dir = _find_search_dir(corpus_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": date.today().isoformat(),
        "git_sha": _git_sha(),
        "corpus_root": str(corpus_root.resolve()),
        "topic_clusters": _top20_topic_clusters(search_dir),
        "insight_clusters": _top10_insight_clusters(search_dir),
        "bridge_distribution": _bridge_distribution(corpus_root),
        "filter_impact": _filter_impact(corpus_root),
        "cost_rollup": _cost_rollup(corpus_root),
        "gil_quality_metrics": _run_quality_metric_target("gil-quality-metrics", corpus_root),
        "kg_quality_metrics": _run_quality_metric_target("kg-quality-metrics", corpus_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="snapshot_quality")
    parser.add_argument("corpus_root", help="Corpus root (contains feeds/ or run_*/)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: {corpus_root}/_quality_snapshot_<date>.json)",
    )
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).expanduser().resolve()
    if not corpus_root.is_dir():
        print(f"Error: corpus path does not exist: {corpus_root}", file=sys.stderr)
        return 2

    snapshot = build_snapshot(corpus_root)

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = corpus_root / f"_quality_snapshot_{snapshot['snapshot_date']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Tiny human-readable summary to stdout.
    print(f"Wrote {output}")
    fi = snapshot["filter_impact"]
    bd = snapshot["bridge_distribution"]
    if fi.get("status") == "ok":
        print(
            f"  filters: insights={fi['insights_total']}  ads={fi['ads_dropped']}"
            f" ({fi['ads_dropped_pct']}%)  dialogue={fi['dialogue_dropped']}"
            f" ({fi['dialogue_dropped_pct']}%)  topics_norm={fi['topics_normalized']}"
            f" ({fi['topics_normalized_pct']}%)"
        )
    if bd.get("status") == "ok":
        print(
            f"  bridge:  both={bd['both']} ({bd['pct_both']}%)  "
            f"gi_only={bd['gi_only']} ({bd['pct_gi_only']}%)  "
            f"kg_only={bd['kg_only']} ({bd['pct_kg_only']}%)"
        )
    tc = snapshot["topic_clusters"]
    if tc.get("status") == "ok":
        print(
            f"  topics:  clusters={tc['cluster_count']}  topics={tc['topic_count']}"
            f"  singletons={tc['singletons']}  threshold={tc['threshold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
