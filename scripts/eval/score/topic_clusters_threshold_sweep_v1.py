"""Topic-clusters threshold sweep (#904 Phase C) + synthetic frame-negative test.

Sweeps `cluster_indices_by_threshold` similarity threshold over the v2 KG
output. Default is 0.75. Looks for:
- How many tc:* parents emerge at each threshold
- How many span >=2 feeds (cross-feed match against v2 spec targets)
- Whether the frame-negative test is exercised (synthetic non-p04 frame
  topic injected to validate the cluster predicate discriminates correctly)

Usage:
    python scripts/eval/score/topic_clusters_threshold_sweep_v1.py \\
        --kg-run kg_gemini_curated_5feeds_kg_v2_provider \\
        --dataset curated_5feeds_kg_v2 \\
        --output  data/eval/runs/baseline_topic_clusters_threshold_sweep_v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Re-use the baseline's KG-topic collection + cluster-build path
_PATH = PROJECT_ROOT / "scripts" / "eval" / "score" / "topic_clusters_baseline_v2.py"
_spec = importlib.util.spec_from_file_location("topic_clusters_baseline_v2_for_sweep", _PATH)
assert _spec and _spec.loader
sweep = importlib.util.module_from_spec(_spec)
sys.modules["topic_clusters_baseline_v2_for_sweep"] = sweep
_spec.loader.exec_module(sweep)

THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    kg_path = PROJECT_ROOT / "data" / "eval" / "runs" / args.kg_run / "predictions.jsonl"
    dataset_path = PROJECT_ROOT / "data" / "eval" / "datasets" / f"{args.dataset}.json"
    ep_to_feed = sweep._episode_to_feed(dataset_path)
    rows = sweep._collect_kg_topics(kg_path, ep_to_feed)

    # Synthetic frame-negative test injection: add a non-p04 "frame" topic
    # to exercise the negative-test assertion (currently vacuous on real v2
    # because frame only exists in p04 — see #903 audit). If clustering is
    # working correctly, this synthetic p02 "frame" should NOT cluster with
    # p04's "frame composition" / "frame and light relationship".
    # Inject a NON-PHOTOGRAPHIC bare "frame" label into a non-p04 feed.
    # If the cluster predicate is doing its job, this should NOT cluster
    # with p04's "frame composition" / "frame and light relationship"
    # (different domain, embeddings should diverge despite shared token).
    # If they DO cluster, the negative test fires — telling us the
    # embedding-similarity-only approach can be fooled by word overlap.
    synthetic_row = {
        "episode_id": "p02_synthetic",
        "feed": "feed-p02",
        "topic_id": "topic:legal-frame",
        "label": "legal frame for engineering trade-off decisions",
    }
    rows_with_injection = rows + [synthetic_row]

    results = []
    for threshold in THRESHOLDS:
        for label, rows_for in (("no_injection", rows), ("with_injection", rows_with_injection)):
            payload = sweep.build_clusters(rows_for, threshold=threshold, model=sweep.DEFAULT_MODEL)
            agg = sweep.aggregate(payload)
            results.append(
                {
                    "threshold": threshold,
                    "injection": label,
                    "tc_parent_count": agg["tc_parent_count"],
                    "tc_cross_feed_count": agg["tc_cross_feed_count"],
                    "frame_negative_test_pass": agg["ac_pass"]["frame_negative_test"],
                    "frame_negative_test_exercised": agg["frame_negative_test"]["exercised"],
                    "frame_negative_violations": len(agg["frame_negative_test"]["violations"]),
                    "parents_per_podcast": agg["tc_parents_per_podcast"],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(
            {"schema": "metrics_topic_clusters_threshold_sweep_v1", "results": results},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{'thresh':>6} {'inj':<14} {'tc#':>3} {'xfeed':>5} "
        f"{'frame_pass':>10} {'exercised':>9} {'viols':>5}"
    )
    for r in results:
        print(
            f"{r['threshold']:>6.2f} {r['injection']:<14} "
            f"{r['tc_parent_count']:>3} {r['tc_cross_feed_count']:>5} "
            f"{str(r['frame_negative_test_pass']):>10} "
            f"{str(r['frame_negative_test_exercised']):>9} "
            f"{r['frame_negative_violations']:>5}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
