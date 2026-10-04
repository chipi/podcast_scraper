#!/usr/bin/env python3
"""Per-rung precision of the speaker-naming ladder, from decision traces joined to gold labels.

#2276 phase 2. Input: the JSONL `roster_replay.py --trace-out` writes (one episode per line:
`meta_path`, `roster`, `decision_trace`) and the labelled gold cases. For every scored gold voice
it takes the gate's own outcome (`naming_gate.outcome`, so the numbers agree with the scoreboard)
and attributes it to a rung:

- a published name (correct, wrong, spurious, non-participant): the EARLIEST trace step that set a
  name for that same person on that voice — the rung that brought the name in;
- a missing name: where the gold name appears in the trace instead — published on ANOTHER voice,
  refused at a rung (publish gate, intro reader, guest pool, …), present only in the stated inputs,
  or nowhere at all;
- a host/guest role error: the host-seat step that seated the voice (or that it was never seated).

Read-only. The labels, cases and traces come from real episodes and are NEVER committed.

    python scripts/measure/trace_rung_precision.py --traces traces.jsonl \\
        --cases GOLD/val_v1/cases --labels GOLD/val_v1/labels_v1 [--cases ... --labels ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import naming_gate as G  # noqa: E402

#: Decisions that put a name on the voice (see docs/wip/NAMING_DECISION_TRACE.md, step semantics).
SETS_A_NAME = frozenset(
    {
        "named",
        "renamed",
        "set",
        "changed",
        "accepted",
        "forced_pool_name",
        "forced_name",
        "prefix_stripped",
        "restored",
    }
)
#: Rungs whose role-diff `set`/`changed` steps only restate an earlier name: never the origin.
NOT_AN_ORIGIN = frozenset({"voice_types", "ad_voice_placeholder", "host_naming", "guest_naming"})


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b) and G._same(str(a), str(b))


def origin_rung(steps: List[Dict[str, Any]], name: str) -> str:
    """The rung of the earliest step that set ``name`` (the same person) on this voice."""
    for s in steps:
        if s.get("decision") == "forced_pool_name" and _same(s.get("name"), name):
            return "host_naming:forced_pool_name"
        if s.get("decision") == "forced_name" and _same(s.get("name"), name):
            return f"guest_naming:forced_name:{s.get('forced_by')}"
        if s.get("rung") in NOT_AN_ORIGIN:
            continue
        if s.get("decision") in SETS_A_NAME and _same(s.get("name"), name):
            return f"{s['rung']}:{s['decision']}"
    return "?no_setting_step"


def where_missing(trace: Dict[str, Any], roster: Dict[str, Any], voice: str, gold: str) -> str:
    """Where a gold name that did not land on its voice shows up in the trace."""
    for v, r in roster.items():
        if v != voice and r.get("named") and _same(r.get("name"), gold):
            return "published_on_another_voice"
    for n, steps in (trace.get("names") or {}).items():
        if _same(n, gold):
            return "name_step:" + ",".join(sorted({f"{s['rung']}:{s['decision']}" for s in steps}))
    for v, steps in (trace.get("voices") or {}).items():
        for s in steps:
            for k in ("proposed", "heard", "name"):
                if _same(s.get(k), gold) and s.get("decision") in (
                    "skipped",
                    "refused",
                    "refused_spelling",
                    "removed",
                    "heard",
                ):
                    where = "this_voice" if v == voice else "other_voice"
                    return f"seen_{where}:{s['rung']}:{s['decision']}"
    inputs = trace.get("inputs") or {}
    for key in ("known_hosts", "detected_guests", "metadata_named"):
        if any(_same(x, gold) for x in inputs.get(key) or []):
            return f"stated_only:{key}"
    if any(_same(x, gold) for x in (inputs.get("llm_voice_names") or {}).values()):
        return "stated_only:llm_voice_names"
    return "never_in_trace"


def seat_step(steps: List[Dict[str, Any]]) -> str:
    for s in steps:
        if s.get("rung") == "host_seat_step":
            return str(s.get("step"))
    return "not_seated"


def _load_gold(pairs: Iterable[Tuple[Path, Path]]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    gold: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for cases, labels in pairs:
        for case, lab in G.load_labelled(cases, labels):
            gold[case["meta_relpath"]] = (
                case["case_id"],
                {v["voice"]: v for v in lab.get("voices", [])},
            )
    return gold


def analyse(traces: Path, gold: Dict[str, Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    by_origin: Dict[str, Counter] = defaultdict(Counter)
    missing: Counter = Counter()
    roles: Dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    matched = 0
    for line in traces.open(encoding="utf-8"):
        rec = json.loads(line)
        rel = str(rec.get("meta_path") or "")
        rel = rel.split("/app/output/", 1)[-1]
        if rel not in gold:
            continue
        matched += 1
        case_id, labels = gold[rel]
        trace, roster = rec["decision_trace"], rec["roster"]
        for voice, label in labels.items():
            if not G.is_scored(label):
                continue
            r = roster.get(voice) or {}
            name = r.get("name") if r.get("named") else None
            out = G.outcome(label, name, r.get("role"))
            totals[out] += 1
            steps = (trace.get("voices") or {}).get(voice, [])
            if name:
                origin = origin_rung(steps, name)
                by_origin[origin][out] += 1
                if out != "correct_name" and len(examples[origin]) < 6:
                    examples[origin].append(
                        f"{case_id} {voice}: {name!r} (gold {label.get('name')!r})"
                    )
            elif out == "missing_name":
                where = where_missing(trace, roster, voice, label["name"])
                missing[where] += 1
                if len(examples["missing:" + where]) < 6:
                    examples["missing:" + where].append(
                        f"{case_id} {voice}: gold {label['name']!r}"
                    )
            if label.get("role") in G.PARTICIPANT:
                ok = r.get("role") == label["role"]
                roles[f"{seat_step(steps)} -> published {r.get('role')}"][
                    f"gold_{label['role']}{'' if ok else '_WRONG'}"
                ] += 1
    return {
        "episodes_matched": matched,
        "totals": dict(totals),
        "by_origin": {k: dict(v) for k, v in by_origin.items()},
        "missing": dict(missing),
        "roles": {k: dict(v) for k, v in roles.items()},
        "examples": dict(examples),
    }


def _print(rep: Dict[str, Any]) -> None:
    print(f"episodes matched: {rep['episodes_matched']}   totals: {rep['totals']}\n")
    print("PUBLISHED NAMES by the rung that brought the name in")
    cols = ["correct_name", "role_error", "wrong_name", "spurious_name", "non_participant"]
    print(f"{'origin rung':58s} " + " ".join(f"{c[:12]:>12s}" for c in cols) + "   precision")
    rows = sorted(rep["by_origin"].items(), key=lambda kv: -sum(kv[1].values()))
    for origin, c in rows:
        n = sum(c.values())
        prec = c.get("correct_name", 0) / n if n else 0.0
        print(f"{origin:58s} " + " ".join(f"{c.get(k, 0):12d}" for k in cols) + f"   {prec:6.1%}")
    print("\nMISSING NAMES — where the gold name is in the trace instead")
    for where, n in sorted(rep["missing"].items(), key=lambda kv: -kv[1]):
        print(f"{n:5d}  {where}")
    print("\nHOST/GUEST ROLE by host-seat step (participants only)")
    for k, c in sorted(rep["roles"].items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{k:60s} {dict(c)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--cases", type=Path, action="append", required=True)
    ap.add_argument("--labels", type=Path, action="append", required=True)
    ap.add_argument("--json", type=Path, help="also write the full report (with examples) here")
    args = ap.parse_args(argv)
    if len(args.cases) != len(args.labels):
        ap.error("--cases and --labels pair up: give them the same number of times")
    rep = analyse(args.traces, _load_gold(zip(args.cases, args.labels)))
    _print(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
