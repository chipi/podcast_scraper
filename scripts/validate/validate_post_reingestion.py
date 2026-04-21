"""Post-reingestion validation harness (docs/wip/POST_REINGESTION_PLAN.md).

Runs the 5 explore-expansion CLI commands + collects metrics so a re-
ingested production corpus can be validated against the expected
grounded-insights / multi-quote / KG v3 improvements.

This is a **scaffold** — specific assertion thresholds are TODO until real
re-ingested data is available to calibrate them. For now the script runs
every command, captures stdout/stderr/exit code, computes the headline
numbers from gi/kg artifacts, and writes a report that the user can eyeball
against the "Expected improvements" table in the plan.

Usage::

    .venv/bin/python scripts/validate/validate_post_reingestion.py \\
        --corpus /path/to/reingested/corpus \\
        [--report .test_outputs/_post_reingest_report.json]

Prerequisites on the corpus path:
  - gi_insight_source=provider was used (not stub)
  - Multi-quote extraction enabled
  - KG v3 prompt (noun-phrase topics)
  - Bridge fuzzy reconciliation enabled

Exit code 0 if every command ran without error; 1 otherwise.
Absence of a pass/fail judgement on *quality* is intentional — the plan
calls for an eyeball review of clusters, bridge merges, and eval scores
that a harness cannot automate without ground truth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY = str(_REPO_ROOT / ".venv" / "bin" / "python")


@dataclass
class CommandResult:
    label: str
    cmd: List[str]
    exit_code: int
    elapsed: float
    stdout_tail: str
    stderr_tail: str
    extras: Dict[str, Any] = field(default_factory=dict)


def _run(label: str, cmd: List[str], timeout: float = 300.0) -> CommandResult:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc = p.returncode
        out, err = p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -1, "", f"TIMEOUT after {timeout}s"
    return CommandResult(
        label=label,
        cmd=cmd,
        exit_code=rc,
        elapsed=time.time() - t0,
        stdout_tail=(out or "")[-600:],
        stderr_tail=(err or "")[-600:],
    )


def _summarise_gi_kg(corpus: Path) -> Dict[str, Any]:
    """Walk every gi.json / kg.json and aggregate the headline counts the
    POST_REINGESTION_PLAN expected-improvements table needs."""
    gi_paths = sorted(corpus.rglob("*gi.json"))
    kg_paths = sorted(corpus.rglob("*kg.json"))

    total_insights = 0
    grounded_insights = 0
    total_quotes = 0
    insight_with_quotes = 0
    for p in gi_paths:
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        nodes = payload.get("nodes") or []
        edges = payload.get("edges") or []
        insight_ids = {n["id"] for n in nodes if n.get("type") == "Insight"}
        quote_ids = {n["id"] for n in nodes if n.get("type") == "Quote"}
        total_insights += len(insight_ids)
        total_quotes += len(quote_ids)
        # Grounded = insight has ≥1 SUPPORTED_BY edge.
        grounded_ids = {
            e["from"]
            for e in edges
            if e.get("type") == "SUPPORTED_BY" and e.get("from") in insight_ids
        }
        grounded_insights += len(grounded_ids)
        if grounded_ids:
            insight_with_quotes += len(grounded_ids)

    total_topics = 0
    total_entities = 0
    for p in kg_paths:
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for n in payload.get("nodes") or []:
            if n.get("type") == "Topic":
                total_topics += 1
            elif n.get("type") == "Entity":
                total_entities += 1

    return {
        "gi_artifact_count": len(gi_paths),
        "kg_artifact_count": len(kg_paths),
        "total_insights": total_insights,
        "grounded_insights": grounded_insights,
        "grounded_pct": (
            round(100.0 * grounded_insights / total_insights, 1) if total_insights else 0.0
        ),
        "total_quotes": total_quotes,
        # Aggregate — includes ungrounded insights in the denominator.
        "quotes_per_insight": (round(total_quotes / total_insights, 2) if total_insights else 0.0),
        # Only counts grounded insights — cleaner signal for multi-quote
        # validation (POST_REINGESTION_PLAN target: 3-5 per grounded insight).
        "quotes_per_grounded_insight": (
            round(total_quotes / grounded_insights, 2) if grounded_insights else 0.0
        ),
        "total_kg_topics": total_topics,
        "total_kg_entities": total_entities,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="Path to reingested production corpus.")
    ap.add_argument(
        "--report",
        default=str(_REPO_ROOT / ".test_outputs" / "_post_reingest_report.json"),
        help="Output JSON report path.",
    )
    ap.add_argument(
        "--topic-sample",
        default="investing",
        help="Topic string used for explore / topic-insights / sort-density probes.",
    )
    ap.add_argument(
        "--query-sample",
        default="index funds",
        help="Query string used for explore-quotes probe.",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    if not corpus.is_dir():
        print(f"FATAL: corpus path does not exist or is not a directory: {corpus}", file=sys.stderr)
        return 2

    report: Dict[str, Any] = {
        "corpus": str(corpus),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commands": [],
    }
    all_ok = True

    # Step 1: explore expansion — 5 CLI commands from POST_REINGESTION_PLAN.md.
    commands: List[tuple[str, List[str]]] = [
        (
            "insight-clusters-build",
            [_PY, "-m", "podcast_scraper.cli", "insight-clusters", "--output-dir", str(corpus)],
        ),
        (
            "gi-clusters",
            [
                _PY,
                "-m",
                "podcast_scraper.cli",
                "gi",
                "clusters",
                "--output-dir",
                str(corpus),
                "--top",
                "10",
            ],
        ),
        (
            "gi-explore-with-expand",
            [
                _PY,
                "-m",
                "podcast_scraper.cli",
                "gi",
                "explore",
                "--topic",
                args.topic_sample,
                "--output-dir",
                str(corpus),
                "--expand-clusters",
            ],
        ),
        (
            "gi-explore-quotes",
            [
                _PY,
                "-m",
                "podcast_scraper.cli",
                "gi",
                "explore-quotes",
                "--query",
                args.query_sample,
                "--output-dir",
                str(corpus),
                "--top-k",
                "10",
            ],
        ),
        (
            "gi-topic-insights",
            [
                _PY,
                "-m",
                "podcast_scraper.cli",
                "gi",
                "topic-insights",
                "--topic",
                args.topic_sample,
                "--output-dir",
                str(corpus),
            ],
        ),
        (
            "gi-explore-evidence-density",
            [
                _PY,
                "-m",
                "podcast_scraper.cli",
                "gi",
                "explore",
                "--topic",
                args.topic_sample,
                "--output-dir",
                str(corpus),
                "--sort",
                "evidence-density",
            ],
        ),
    ]

    for label, cmd in commands:
        print(f"\n[run] {label} …", flush=True)
        r = _run(label, cmd)
        print(f"  exit={r.exit_code} elapsed={r.elapsed:.1f}s")
        if r.exit_code != 0:
            all_ok = False
            print(f"  stderr tail: {r.stderr_tail.strip()[-200:]}")
        report["commands"].append(asdict(r))

    # Step 2–3 (headline numbers only — eyeball review of cluster/bridge quality
    # is intentionally not automated per the plan).
    print("\n[agg] summarising gi.json / kg.json artifacts …", flush=True)
    agg = _summarise_gi_kg(corpus)
    report["aggregate"] = agg
    print(
        "  gi={gi_artifact_count} kg={kg_artifact_count} "
        "insights={total_insights} grounded={grounded_pct}% "
        "quotes/insight={quotes_per_insight} "
        "(grounded={quotes_per_grounded_insight}) "
        "topics={total_kg_topics} entities={total_kg_entities}".format(**agg)
    )

    # Plan "Expected improvements" table — soft thresholds that will be
    # tightened once re-ingested data calibrates the pass line.
    soft_gates = {
        "grounded_pct_at_least_50": agg["grounded_pct"] >= 50.0,
        "quotes_per_insight_at_least_2": agg["quotes_per_insight"] >= 2.0,
        "gi_artifacts_present": agg["gi_artifact_count"] > 0,
        "kg_artifacts_present": agg["kg_artifact_count"] > 0,
    }
    report["soft_gates"] = soft_gates
    print()
    print("SOFT GATES (calibrated against plan's expected improvements):")
    for k, v in soft_gates.items():
        print(f"  [{'✓' if v else '✗'}] {k}: {v}")
    soft_pass = all(soft_gates.values())

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.report}")

    print()
    if all_ok and soft_pass:
        print("POST-REINGESTION VALIDATION: PASS (soft gates)")
        print(
            "Next: eyeball review of cluster / bridge / NER quality per "
            "POST_REINGESTION_PLAN.md steps 2, 3, 6."
        )
    else:
        print("POST-REINGESTION VALIDATION: FAIL — see report.json")
        for c in report["commands"]:
            if c["exit_code"] != 0:
                print(f"  cmd failed: {c['label']} exit={c['exit_code']}")
        for k, v in soft_gates.items():
            if not v:
                print(f"  soft gate: {k}")
    return 0 if (all_ok and soft_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
