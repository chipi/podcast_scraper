"""CIL cross-episode bridge baseline (issue #903 AC).

Aggregates per-episode `build_bridge` outputs across the v2 KG/GIL runs and
measures cross-episode CIL spans:

- person:* bridges spanning >1 episode (issue #903 person-bridging target)
- topic:* bridges spanning >=2 feeds   (issue #903 cross-feed topic target)
- org:*   bridges spanning >1 episode

Inputs:
- GI predictions.jsonl from a grounded_insights run on the v2 KG dataset
- KG predictions.jsonl from a knowledge_graph run on the v2 KG dataset
- The episode->feed map from the v2 dataset JSON

Usage:
    python scripts/eval/score/cil_baseline_v2.py \\
        --gi-run gil_gemini_curated_5feeds_kg_v2_provider \\
        --kg-run kg_gemini_curated_5feeds_kg_v2_provider \\
        --dataset curated_5feeds_kg_v2 \\
        --output data/eval/runs/baseline_cil_curated_5feeds_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.builders.bridge_builder import build_bridge


def _load_predictions(path: Path, output_key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            ep_id = d["episode_id"]
            out[ep_id] = d["output"][output_key]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{path.name}:line {line_no}: malformed prediction "
                f"(missing episode_id or output.{output_key}): {exc}"
            ) from exc
    return out


def _episode_to_feed(dataset_path: Path) -> dict[str, str]:
    """Map episode_id -> feed slug parsed from the transcript_path.

    The v2 sources layout puts each episode under ``feed-<podcast_id>/``, so the
    parse looks for that prefix in the path. If a future dataset restructure
    drops the convention we fail loud rather than silently flatten cross-feed
    counts: a single "unknown" feed would suppress every topic-cross-feed bridge
    and the AC test would pass for the wrong reason.
    """
    data = json.loads(dataset_path.read_text())
    feed_map: dict[str, str] = {}
    unresolved: list[str] = []
    for ep in data["episodes"]:
        eid = ep["episode_id"]
        path = ep.get("transcript_path", "")
        parts = Path(path).parts
        feed = next((p for p in parts if p.startswith("feed-")), None)
        if feed is None:
            unresolved.append(f"{eid} (transcript_path={path!r})")
            feed = "unknown"
        feed_map[eid] = feed
    if unresolved:
        total = len(feed_map)
        share = 100.0 * len(unresolved) / total if total else 0
        print(
            f"WARNING [{dataset_path.name}]: {len(unresolved)}/{total} ({share:.0f}%) "
            f"episodes have no `feed-*` prefix in transcript_path; treating as "
            f'feed="unknown" — cross-feed metrics will be flattened. '
            f"Examples: {unresolved[:3]}",
            file=sys.stderr,
        )
        if share > 25.0:
            raise ValueError(
                f"{dataset_path.name}: {share:.0f}% of episodes are missing a "
                f"`feed-*` directory prefix in transcript_path. Cross-feed metrics "
                f"would be meaningless. Restore the convention or update the script."
            )
    return feed_map


def aggregate_cil(
    gi_by_ep: dict[str, dict],
    kg_by_ep: dict[str, dict],
    ep_to_feed: dict[str, str],
) -> dict[str, Any]:
    """Per-identity cross-episode aggregation."""
    # identity_id -> {"episodes": set, "feeds": set, "display_names": set, "type": str}
    per_id: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "episodes": set(),
            "feeds": set(),
            "display_names": set(),
            "type": None,
        }
    )

    for ep_id in sorted(set(gi_by_ep) | set(kg_by_ep)):
        gi = gi_by_ep.get(ep_id)
        kg = kg_by_ep.get(ep_id)
        feed = ep_to_feed.get(ep_id, "unknown")
        bridge = build_bridge(ep_id, gi, kg)
        for identity in bridge.get("identities", []):
            cid = identity.get("id")
            if not cid:
                continue
            entry = per_id[cid]
            entry["type"] = identity.get("type")
            entry["episodes"].add(ep_id)
            entry["feeds"].add(feed)
            for nm in identity.get("display_names", []) or []:
                entry["display_names"].add(nm)
            primary = identity.get("display_name")
            if primary:
                entry["display_names"].add(primary)

    # Serialize + compute summary
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cid, entry in per_id.items():
        by_type[entry["type"] or "unknown"].append(
            {
                "id": cid,
                "type": entry["type"],
                "episode_count": len(entry["episodes"]),
                "episodes": sorted(entry["episodes"]),
                "feed_count": len(entry["feeds"]),
                "feeds": sorted(entry["feeds"]),
                "display_names": sorted(entry["display_names"]),
            }
        )

    persons = by_type.get("person", [])
    topics = by_type.get("topic", [])
    orgs = by_type.get("org", [])

    person_bridges_multi_ep = [p for p in persons if p["episode_count"] > 1]
    topic_bridges_cross_feed = [t for t in topics if t["feed_count"] >= 2]
    org_bridges_multi_ep = [o for o in orgs if o["episode_count"] > 1]

    return {
        "summary": {
            "person_count_total": len(persons),
            "person_bridges_multi_ep": len(person_bridges_multi_ep),
            "topic_count_total": len(topics),
            "topic_bridges_cross_feed": len(topic_bridges_cross_feed),
            "org_count_total": len(orgs),
            "org_bridges_multi_ep": len(org_bridges_multi_ep),
            "episode_count": len(set(gi_by_ep) | set(kg_by_ep)),
            "feed_count": len(set(ep_to_feed.values())),
        },
        "ac_targets": {
            "person_bridges_multi_ep_target_gt": 0,
            "topic_bridges_cross_feed_target_gt": 0,
        },
        "ac_pass": {
            "person_bridges_multi_ep": len(person_bridges_multi_ep) > 0,
            "topic_bridges_cross_feed": len(topic_bridges_cross_feed) > 0,
        },
        "person_bridges_multi_ep": sorted(
            person_bridges_multi_ep, key=lambda p: -p["episode_count"]
        ),
        "topic_bridges_cross_feed": sorted(
            topic_bridges_cross_feed, key=lambda t: -t["feed_count"]
        ),
        "org_bridges_multi_ep": sorted(org_bridges_multi_ep, key=lambda o: -o["episode_count"]),
        "persons": sorted(persons, key=lambda p: -p["episode_count"]),
        "topics": sorted(topics, key=lambda t: -t["feed_count"]),
        "orgs": sorted(orgs, key=lambda o: -o["episode_count"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gi-run", required=True, help="GI run_id under data/eval/runs/")
    p.add_argument("--kg-run", required=True, help="KG run_id under data/eval/runs/")
    p.add_argument("--dataset", required=True, help="Dataset id (under data/eval/datasets/)")
    p.add_argument("--output", type=Path, required=True, help="Output directory for baseline")
    p.add_argument("--baseline-id", default="baseline_cil_curated_5feeds_v2")
    args = p.parse_args()

    gi_path = PROJECT_ROOT / "data" / "eval" / "runs" / args.gi_run / "predictions.jsonl"
    kg_path = PROJECT_ROOT / "data" / "eval" / "runs" / args.kg_run / "predictions.jsonl"
    dataset_path = PROJECT_ROOT / "data" / "eval" / "datasets" / f"{args.dataset}.json"
    for p_ in (gi_path, kg_path, dataset_path):
        if not p_.exists():
            print(f"Missing input: {p_}", file=sys.stderr)
            return 1

    gi_by_ep = _load_predictions(gi_path, "gil")
    kg_by_ep = _load_predictions(kg_path, "kg")
    ep_to_feed = _episode_to_feed(dataset_path)

    cil = aggregate_cil(gi_by_ep, kg_by_ep, ep_to_feed)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "baseline_id": args.baseline_id,
        "task": "cil_cross_episode_bridge",
        "gi_run_id": args.gi_run,
        "kg_run_id": args.kg_run,
        "dataset_id": args.dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **cil,
        "schema": "metrics_cil_baseline_v1",
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    s = cil["summary"]
    report = [
        f"# CIL Baseline — {args.baseline_id}",
        "",
        f"**GI run:** `{args.gi_run}`  ",
        f"**KG run:** `{args.kg_run}`  ",
        f"**Dataset:** `{args.dataset}`  ",
        f"**Episodes:** {s['episode_count']}  ",
        f"**Feeds:** {s['feed_count']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value | AC target |",
        "| --- | ---: | --- |",
    ]
    p_v = "PASS" if cil["ac_pass"]["person_bridges_multi_ep"] else "FAIL"
    t_v = "PASS" if cil["ac_pass"]["topic_bridges_cross_feed"] else "FAIL"
    report += [
        f"| person:* identities | {s['person_count_total']} | — |",
        f"| person:* bridges spanning >1 ep | {s['person_bridges_multi_ep']} | >0 ({p_v}) |",
        f"| topic:* identities | {s['topic_count_total']} | — |",
        f"| topic:* bridges spanning >=2 feeds | {s['topic_bridges_cross_feed']} | >0 ({t_v}) |",
        f"| org:* identities | {s['org_count_total']} | — |",
        f"| org:* bridges spanning >1 ep | {s['org_bridges_multi_ep']} | — |",
        "",
        "## Top person bridges (multi-episode)",
        "",
        "| Identity | Episodes | Display names |",
        "| --- | ---: | --- |",
    ]
    for pb in cil["person_bridges_multi_ep"][:10]:
        names = ", ".join(pb["display_names"][:3])
        report.append(f"| `{pb['id']}` | {pb['episode_count']} | {names} |")
    report += [
        "",
        "## Top topic bridges (cross-feed)",
        "",
        "| Identity | Feeds | Display names |",
        "| --- | ---: | --- |",
    ]
    for tb in cil["topic_bridges_cross_feed"][:10]:
        names = ", ".join(tb["display_names"][:3])
        report.append(f"| `{tb['id']}` | {tb['feed_count']} | {names} |")
    (args.output / "metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(
        f"  person_multi_ep={s['person_bridges_multi_ep']} "
        f"topic_cross_feed={s['topic_bridges_cross_feed']} "
        f"org_multi_ep={s['org_bridges_multi_ep']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
