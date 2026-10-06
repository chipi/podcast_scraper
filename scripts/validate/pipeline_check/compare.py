"""Apply a check from ``expectations.yaml`` to the workers' JSON. Pure: no pipeline imports.

Three kinds of finding, each with where it happened:

* **hole** — a decision that did not resolve to the expected key (a lookup asked for another key
  or missed the expected row; a language argument that does not normalise to it);
* **variant difference** — two variants that must behave identically produced different stage
  records or different decisions;
* **base difference** — a stage that must match the base ref did not, or a pinned value was wrong.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

Norm = Dict[str, Optional[str]]

#: Stage -> report area (the report's row label).
AREAS = {
    "language": "Language resolution",
    "gate": "Refusal gate",
    "hosts": "Host detection",
    "naming": "Speaker naming",
    "ad_regions": "Ad removal",
    "transcript": "Transcript selection",
    "translation": "Translation / analysis gate",
    "search_routing": "Search routing",
    "": "Outside a stage",
}


@dataclass
class Finding:
    kind: str  # hole | variant | base | value
    area: str
    where: str
    detail: str


@dataclass
class CheckResult:
    name: str
    findings: List[Finding] = field(default_factory=list)
    decisions_per_area: Dict[str, int] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    episodes: int = 0
    variants: List[str] = field(default_factory=list)
    report_only: Dict[str, Any] = field(default_factory=dict)
    #: area -> one-line outcome of the base comparison / pinned values for that stage
    stage_status: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.findings


def _get(record: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    stage, _, fld = path.partition(".")
    rec = record.get(stage)
    if not isinstance(rec, dict) or fld not in rec:
        return False, None
    return True, rec[fld]


def _strip(record: Dict[str, Any], ignore: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = json.loads(json.dumps(record))
    for path in ignore:
        stage, _, fld = path.partition(".")
        if isinstance(out.get(stage), dict):
            out[stage].pop(fld, None)
    return out


def _normalised(value: Any, norm: Dict[str, Optional[str]]) -> Any:
    if isinstance(value, str):
        return norm.get(value, value)
    return value


def event_resolves(
    event: Dict[str, Any], target: str, norm: Dict[str, Optional[str]]
) -> Optional[str]:
    """``None`` when the event resolved to *target*; otherwise why not."""
    if event["kind"] == "map":
        key = event["key"]
        if key is None:
            return None
        if key != target:
            # A code-keyed map answers only the code itself: "en_us" misses the "en" row even
            # though it names the same language.
            return f"looked up {key!r}"
        if event.get("hit") is False:
            return f"no {target!r} row"
        return None
    for name, value in (event.get("key") or {}).items():
        if value in (None, ""):
            continue
        if _normalised(value, norm) != target:
            return f"{name}={value!r}"
    return None


def _holes(
    res: CheckResult, events: List[Dict[str, Any]], target: Optional[str], norm: Norm
) -> None:
    """Count decisions per area; report each distinct decision that did not resolve to *target*."""
    seen = set()
    for ev in events:
        area = AREAS.get(ev.get("stage", ""), ev.get("stage", ""))
        res.decisions_per_area[area] = res.decisions_per_area.get(area, 0) + 1
        why = event_resolves(ev, target, norm) if target else None
        if why and (ev["site"], ev["caller"], why) not in seen:
            seen.add((ev["site"], ev["caller"], why))
            where = f"{ev['site']} (called from {ev['caller']})"
            res.findings.append(Finding("hole", area, where, f"{why} under {ev['variant']!r}"))


def _trace_key(ev: Dict[str, Any], norm: Norm) -> Tuple[Any, ...]:
    """A decision as compared across variants: where, by whom, and the key it resolved to."""
    key = ev["key"]
    if ev["kind"] == "call":
        key = json.dumps({k: _normalised(v, norm) for k, v in (key or {}).items()}, sort_keys=True)
    else:
        key = _normalised(key, norm)
    return (ev["stage"], ev["site"], ev["caller"], key, ev.get("hit"))


def _variants(
    res: CheckResult, check: Dict[str, Any], candidate: Dict[str, Any], norm: Norm
) -> None:
    """Variants that must be identical: same stage records, and the same decisions."""
    results = candidate["results"]
    if not check.get("variants_identical") or len(results) < 2:
        return
    ignore = check.get("variants_identical_ignore", [])
    first, *rest = list(results)
    for ep in candidate["episodes"]:
        ref = _strip(results[first][ep], ignore)
        for other in rest:
            got = _strip(results[other][ep], ignore)
            for stage in sorted(set(ref) | set(got)):
                if ref.get(stage) != got.get(stage):
                    detail = f"{stage}: {first!r} vs {other!r} differ"
                    res.findings.append(Finding("variant", AREAS.get(stage, stage), ep, detail))
    traces: Dict[str, List[Tuple[Any, ...]]] = defaultdict(list)
    for ev in candidate["events"]:
        traces[ev["variant"]].append(_trace_key(ev, norm))
    for other in rest:
        a, b = traces.get(first, []), traces.get(other, [])
        if a == b:
            continue
        idx = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        left = a[idx] if idx < len(a) else "end"
        right = b[idx] if idx < len(b) else "end"
        detail = f"{first!r}: {left} vs {other!r}: {right}"
        res.findings.append(Finding("variant", "Decisions", f"decision #{idx}", detail))


def _pinned(
    res: CheckResult, check: Dict[str, Any], results: Dict[str, Any]
) -> Dict[str, List[int]]:
    """Pinned ``<stage>.<field>`` values; returns area -> [checked, failed]."""
    tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for path, want in (check.get("expect_values") or {}).items():
        area = AREAS.get(path.split(".")[0], path)
        for variant, eps in results.items():
            for ep, record in eps.items():
                present, got = _get(record, path)
                tally[area][0] += 1
                if not present or got != want:
                    tally[area][1] += 1
                    detail = f"{path} = {got!r}, expected {want!r}"
                    res.findings.append(Finding("value", area, f"{ep} under {variant!r}", detail))
    return tally


def _against_base(
    res: CheckResult, check: Dict[str, Any], results: Dict[str, Any], base: Optional[Dict[str, Any]]
) -> Dict[str, List[int]]:
    """Stages that must equal the base ref; returns area -> [compared, differ]."""
    tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    if base is None:
        return tally
    for stage in check.get("base_identical_stages", []):
        area = AREAS.get(stage, stage)
        for variant, eps in results.items():
            base_eps = base["results"].get(variant, {})
            for ep, record in eps.items():
                tally[area][0] += 1
                if record.get(stage) != (base_eps.get(ep) or {}).get(stage):
                    tally[area][1] += 1
                    detail = f"{stage} differs from base"
                    res.findings.append(Finding("base", area, f"{ep} under {variant!r}", detail))
    return tally


def _stage_status(
    res: CheckResult, based: Dict[str, List[int]], pinned: Dict[str, List[int]]
) -> None:
    for area, (n, bad) in based.items():
        res.stage_status[area] = (
            f"identical to base ({n})" if not bad else f"{bad} of {n} differ from base"
        )
    for area, (n, bad) in pinned.items():
        note = f"pinned values hold ({n})" if not bad else f"{bad} of {n} pinned values fail"
        res.stage_status[area] = (
            f"{res.stage_status[area]}; {note}" if area in res.stage_status else note
        )


def coverage(worker: Dict[str, Any]) -> Dict[str, Any]:
    """Decision points discovered vs exercised, and what could not be traced."""
    points = set(worker["decision_points"]["maps"]) | set(worker["decision_points"]["functions"])
    hit = {ev["site"] for ev in worker["events"]}
    return {
        "discovered": len(points),
        "exercised": len(points & hit),
        "not_exercised": sorted(points - hit),
        "import_time_copies": worker.get("import_time_copies", []),
        "skipped_modules": worker.get("skipped_modules", {}),
    }


def evaluate(
    name: str,
    check: Dict[str, Any],
    candidate: Dict[str, Any],
    base: Optional[Dict[str, Any]],
) -> CheckResult:
    """Apply one check to the candidate worker's output (and the base's, when given)."""
    res = CheckResult(
        name=name, episodes=len(candidate["episodes"]), variants=list(candidate["variants"])
    )
    norm: Norm = candidate.get("normalised", {})
    results = candidate["results"]
    _holes(res, candidate["events"], check.get("decisions_resolve_to"), norm)
    _variants(res, check, candidate, norm)
    pinned = _pinned(res, check, results)
    based = _against_base(res, check, results, base)
    _stage_status(res, based, pinned)
    res.coverage = coverage(candidate)
    if check.get("report_only"):
        # Nothing is expected, so nothing can fail: the differences ARE the answer.
        res.report_only = _variant_differences(results)
        res.findings = []
    return res


def _variant_differences(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """For a report-only check: which stages differ between which variants, per episode."""
    out: Dict[str, Any] = {}
    variants = list(results)
    for i, a in enumerate(variants):
        for b in variants[i + 1 :]:
            diffs = []
            for ep in results[a]:
                for stage in sorted(set(results[a][ep]) | set(results[b][ep])):
                    if stage == "language":
                        continue
                    if results[a][ep].get(stage) != results[b][ep].get(stage):
                        diffs.append(f"{ep}:{stage}")
            out[f"{a} vs {b}"] = diffs
    return out
