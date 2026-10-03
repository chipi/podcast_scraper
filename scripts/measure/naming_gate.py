#!/usr/bin/env python3
"""The naming gate: score speaker naming against human-quality labels, old code vs new. Read-only.

Every naming change is a trade (it fixes some voices and may break others), and reading a replay
diff by hand cannot say whether the trade is good. This script answers it against GROUND TRUTH:
a labelled set of real episodes where each diarized voice carries its true role (host / guest /
ad / promo / clip / unknown) and the name the episode's own text supports (or null).

For each labelled episode it replays the speaker roster twice — the OLD variant and the NEW one,
exactly as ``roster_replay.py`` does (stored voice texts, stored LLM answers, no LLM, nothing
written) — and scores every voice:

  correct_name        labelled name, published the same person
  wrong_name          labelled name, published a different person
  missing_name        labelled name, published nobody
  spurious_name       labelled null, published a name (a name the text does not support)
  non_participant     labelled ad / promo / clip, but published as a speaker
  correct_unnamed     labelled null (participant), published nobody
  role_error          participant named correctly but host/guest swapped

Labels with role ``unknown`` or confidence ``low`` are not scored (counted separately).

Output: one table per variant, then every REGRESSION (a voice OLD scored correct that NEW does
not) and every FIX (the reverse). The gate for a change is: no regression on a labelled host,
and the regression list read in full.

The labels and cases are derived from real episodes and are NEVER committed; keep them on the box
or in a scratch directory (never-commit-real-episodes). This file is code only.

    python scripts/measure/naming_gate.py --corpus /app/output \\
        --cases /tmp/rp_labels/cases --labels /tmp/rp_labels/out \\
        --new roster=/tmp/rp/roster_v4.py --signatures
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roster_replay as R  # noqa: E402

PARTICIPANT = frozenset({"host", "guest"})
NON_PARTICIPANT = frozenset({"ad", "promo", "clip"})
CORRECT = frozenset({"correct_name", "correct_unnamed"})
#: How bad an outcome is. A wrong name is worse than no name; a missing name worse than correct.
#: A change is BETTER or WORSE by this rank: "wrong name became unnamed" is an improvement.
SEVERITY = {
    "correct_name": 0,
    "correct_unnamed": 0,
    "missing_name": 1,
    "role_error": 2,
    "spurious_name": 3,
    "wrong_name": 3,
    "non_participant": 3,
}


def _same(a: str, b: str) -> bool:
    from podcast_scraper.kg.speaker_coherence import same_person

    return a.strip().lower() == b.strip().lower() or same_person(a, b)


def is_scored(label: Dict[str, Any]) -> bool:
    """Unknown roles and low-confidence labels are not scored."""
    role = str(label.get("role") or "unknown")
    return role != "unknown" and str(label.get("confidence") or "") != "low"


def score_voice(label: Dict[str, Any], name: Optional[str], role: Optional[str]) -> Optional[str]:
    """The outcome for one voice, or None when the label is not scored."""
    return outcome(label, name, role) if is_scored(label) else None


def outcome(label: Dict[str, Any], name: Optional[str], role: Optional[str]) -> str:
    """The outcome for one voice whose label is scored."""
    lrole = str(label.get("role") or "unknown")
    lname = label.get("name") or None
    if lrole in NON_PARTICIPANT:
        return "non_participant" if name else "correct_unnamed"
    if lname and name:
        if not _same(lname, name):
            return "wrong_name"
        if role in PARTICIPANT and role != lrole:
            return "role_error"
        return "correct_name"
    if lname:
        return "missing_name"
    return "spurious_name" if name else "correct_unnamed"


def _published(result: Any, voice: str) -> Tuple[Optional[str], Optional[str]]:
    v = result.by_voice.get(voice)
    if v is None or not getattr(v, "named", False):
        return None, getattr(v, "role", None) if v is not None else None
    return v.name, getattr(v, "role", None)


def load_labelled(cases_dir: Path, labels_dir: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    out = []
    for lab_path in sorted(labels_dir.glob("*.json")):
        case_path = cases_dir / lab_path.name
        if not case_path.is_file():
            continue
        out.append(
            (
                json.loads(case_path.read_text(encoding="utf-8")),
                json.loads(lab_path.read_text(encoding="utf-8")),
            )
        )
    return out


def run(
    corpus: Path,
    labelled: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    old: Dict[str, Any],
    new: Dict[str, Any],
    sig_old: Any = None,
    sig_new: Any = None,
) -> Dict[str, Any]:
    tallies: Dict[str, Counter] = {"old": Counter(), "new": Counter()}
    host_tallies: Dict[str, Counter] = {"old": Counter(), "new": Counter()}
    regressions: List[Dict[str, Any]] = []
    fixes: List[Dict[str, Any]] = []
    better: List[Dict[str, Any]] = []
    worse: List[Dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for case, labels in labelled:
        try:
            ep = R.load_episode(corpus / case["meta_relpath"])
            a = R.replay(old["roster"], ep, sig_old)
            b = R.replay(new["roster"], ep, sig_new)
        except Exception as exc:  # noqa: BLE001 — one bad episode must not stop the gate
            skipped[f"replay_error:{type(exc).__name__}"] += 1
            continue
        for lab in labels.get("voices") or []:
            voice = str(lab.get("voice"))
            on, orole = _published(a, voice)
            nn, nrole = _published(b, voice)
            if not is_scored(lab):
                skipped["unscored_label"] += 1
                continue
            so, sn = outcome(lab, on, orole), outcome(lab, nn, nrole)
            tallies["old"][so] += 1
            tallies["new"][sn] += 1
            if lab.get("role") == "host":
                host_tallies["old"][so] += 1
                host_tallies["new"][sn] += 1
            row = {
                "case": case["case_id"],
                "feed": (case.get("feed") or {}).get("title"),
                "episode": (case.get("episode") or {}).get("title"),
                "voice": voice,
                "label": {k: lab.get(k) for k in ("role", "name", "confidence")},
                "old": {"name": on, "role": orole, "score": so},
                "new": {"name": nn, "role": nrole, "score": sn},
            }
            if so in CORRECT and sn not in CORRECT:
                regressions.append(row)
            elif sn in CORRECT and so not in CORRECT:
                fixes.append(row)
            if SEVERITY[sn] < SEVERITY[so]:
                better.append(row)
            elif SEVERITY[sn] > SEVERITY[so]:
                worse.append(row)
    return {
        "episodes": len(labelled),
        "old": dict(tallies["old"]),
        "new": dict(tallies["new"]),
        "hosts_old": dict(host_tallies["old"]),
        "hosts_new": dict(host_tallies["new"]),
        "regressions": regressions,
        "fixes": fixes,
        "better": better,
        "worse": worse,
        "skipped": dict(skipped),
    }


def ladder(
    corpus: Path,
    labelled: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    steps: List[Tuple[str, Dict[str, Any], Any]],
) -> Dict[str, Any]:
    """Score each cumulative step (baseline, +slice 1, +slices 1-2, ...) and what each step changed.

    ``steps`` is ``[(name, variant, signatures), ...]`` in order; the first is the baseline. Every
    step is a complete code variant, so a column shows the whole stack up to that slice.
    """
    names = [s[0] for s in steps]
    tallies: Dict[str, Counter[str]] = {n: Counter() for n in names}
    hosts: Dict[str, Counter[str]] = {n: Counter() for n in names}
    changes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        n: {"better": [], "worse": []} for n in names[1:]
    }
    skipped: Counter[str] = Counter()
    for case, labels in labelled:
        try:
            ep = R.load_episode(corpus / case["meta_relpath"])
            results = [R.replay(v["roster"], ep, sig) for _n, v, sig in steps]
        except Exception as exc:  # noqa: BLE001 — one bad episode must not stop the gate
            skipped[f"replay_error:{type(exc).__name__}"] += 1
            continue
        for lab in labels.get("voices") or []:
            voice = str(lab.get("voice"))
            published = [_published(r, voice) for r in results]
            if not is_scored(lab):
                skipped["unscored_label"] += 1
                continue
            scores = [outcome(lab, n, ro) for n, ro in published]
            for name, score in zip(names, scores):
                tallies[name][score] += 1
                if lab.get("role") == "host":
                    hosts[name][score] += 1
            for i in range(1, len(steps)):
                prev, cur = scores[i - 1], scores[i]
                if SEVERITY[cur] == SEVERITY[prev]:
                    continue
                row = {
                    "case": case["case_id"],
                    "feed": (case.get("feed") or {}).get("title"),
                    "voice": voice,
                    "label": {k: lab.get(k) for k in ("role", "name", "confidence")},
                    "before": {"name": published[i - 1][0], "score": prev},
                    "after": {"name": published[i][0], "score": cur},
                }
                kind = "better" if SEVERITY[cur] < SEVERITY[prev] else "worse"
                changes[names[i]][kind].append(row)
    return {
        "episodes": len(labelled),
        "steps": names,
        "tallies": {n: dict(t) for n, t in tallies.items()},
        "hosts": {n: dict(t) for n, t in hosts.items()},
        "changes": changes,
        "skipped": dict(skipped),
    }


_OUTCOMES = [
    "correct_name",
    "correct_unnamed",
    "wrong_name",
    "missing_name",
    "spurious_name",
    "non_participant",
    "role_error",
]


def ladder_table(report: Dict[str, Any]) -> str:
    names = report["steps"]
    width = max(12, *(len(n) for n in names))
    lines = [f"{'outcome':18s} " + " ".join(f"{n:>{width}s}" for n in names)]
    for k in _OUTCOMES:
        lines.append(
            f"{k:18s} " + " ".join(f"{report['tallies'][n].get(k, 0):>{width}d}" for n in names)
        )
    lines.append(
        f"{'hosts correct':18s} "
        + " ".join(f"{report['hosts'][n].get('correct_name', 0):>{width}d}" for n in names)
    )
    lines.append(
        f"{'better vs prev':18s} {'':>{width}s} "
        + " ".join(f"{len(report['changes'][n]['better']):>{width}d}" for n in names[1:])
    )
    lines.append(
        f"{'worse vs prev':18s} {'':>{width}s} "
        + " ".join(f"{len(report['changes'][n]['worse']):>{width}d}" for n in names[1:])
    )
    return "\n".join(lines)


def _table(report: Dict[str, Any]) -> str:
    keys = [
        "correct_name",
        "correct_unnamed",
        "wrong_name",
        "missing_name",
        "spurious_name",
        "non_participant",
        "role_error",
    ]
    lines = [f"{'outcome':18s} {'old':>6s} {'new':>6s}   hosts: {'old':>5s} {'new':>5s}"]
    for k in keys:
        lines.append(
            f"{k:18s} {report['old'].get(k, 0):6d} {report['new'].get(k, 0):6d}   "
            f"       {report['hosts_old'].get(k, 0):5d} {report['hosts_new'].get(k, 0):5d}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--old", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--new", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--signatures", action="store_true")
    ap.add_argument("--json", type=Path, help="also write the full report here")
    ap.add_argument(
        "--step",
        nargs="+",
        action="append",
        metavar="NAME [MODULE=PATH ...]",
        help="ladder mode: one cumulative step per flag, in order; the first is the baseline "
        "(e.g. --step today --step +seat_v4 roster=/tmp/v4.py). Replaces --old/--new.",
    )
    ap.add_argument("--repool", action="store_true", help="see roster_replay.py --repool")
    args = ap.parse_args(argv)
    R.REPOOL = bool(args.repool)
    labelled = load_labelled(args.cases, args.labels)
    R.index_siblings(args.corpus)
    if args.step:
        steps = []
        for name, *mods in args.step:
            variant = R.load_variant(R._variant_arg(mods))
            sig = (
                R._signatures(variant.get("ad_signatures"), args.corpus)
                if args.signatures
                else None
            )
            steps.append((name, variant, sig))
        lad = ladder(args.corpus, labelled, steps)
        print(f"labelled episodes: {lad['episodes']}  skipped: {lad['skipped']}")
        print(ladder_table(lad))
        for name in lad["steps"][1:]:
            for kind in ("worse", "better"):
                for r in lad["changes"][name][kind]:
                    print(f"   {name} {kind}", json.dumps(r, ensure_ascii=False))
        if args.json:
            args.json.write_text(json.dumps(lad, ensure_ascii=False, indent=1))
        return 0
    old = R.load_variant(R._variant_arg(args.old))
    new = R.load_variant(R._variant_arg(args.new))
    sig_old = sig_new = None
    if args.signatures:
        sig_old = R._signatures(old.get("ad_signatures"), args.corpus)
        sig_new = R._signatures(new.get("ad_signatures"), args.corpus)
    report = run(args.corpus, labelled, old, new, sig_old, sig_new)
    print(f"labelled episodes: {report['episodes']}  skipped: {report['skipped']}")
    print(_table(report))
    print(f"\nREGRESSIONS (old correct, new not): {len(report['regressions'])}")
    for r in report["regressions"]:
        print("  ", json.dumps(r, ensure_ascii=False))
    print(f"\nFIXES (new correct, old not): {len(report['fixes'])}")
    for r in report["fixes"]:
        print("  ", json.dumps(r, ensure_ascii=False))
    print(
        f"\nBY SEVERITY: better {len(report['better'])}, worse {len(report['worse'])} "
        "(wrong > role swap > missing > correct)"
    )
    for tag, rows in (("better", report["better"]), ("worse", report["worse"])):
        for r in rows:
            print(f"   {tag}", json.dumps(r, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
