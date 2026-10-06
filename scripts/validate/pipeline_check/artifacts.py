"""Real-episode mode: collect a run's artifacts, and compare runs in two different ways.

* **Deterministic artifacts** (file set, ``metadata.json`` minus the LLM's own fields, the
  processing manifest) are compared exactly — volatile keys (timestamps, durations, paths, costs)
  and the expected differences from ``expectations.yaml`` excepted.
* **LLM output** (insights, quotes, grounding, summary, KG) is compared as a handful of metrics
  against a noise band measured from two runs of the base: the LLM does not repeat itself, so
  "the same" can only mean "within the spread the base shows against itself".
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: The run timestamp the pipeline puts in every file name, before `_<hash>` or the extension.
_TS = re.compile(r"_\d{8}-\d{6}(?=[_.])")
_RUN = re.compile(r"run_[^/]+")
VOLATILE = re.compile(
    r"(time|_at$|date|duration|elapsed|run_id|^run$|trace|path|dir$|release|sha|started|finished|"
    r"wall|seconds|_s$|_ms$|host|pid|uuid|hash|fingerprint|created|updated|cost|tokens)",
    re.I,
)


def _norm(rel: str) -> str:
    return _TS.sub("_TS_", _RUN.sub("run_X", rel))


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def collect(output_dir: Path) -> Dict[str, Any]:
    """Every episode the run produced: its files, metadata, manifest, and LLM metrics."""
    episodes: Dict[str, Any] = {}
    files = sorted(
        _norm(str(p.relative_to(output_dir))) for p in output_dir.rglob("*") if p.is_file()
    )
    for meta_path in sorted(output_dir.rglob("metadata/*.metadata.json")):
        meta = _load(meta_path) or {}
        base = meta_path.name[: -len(".metadata.json")]
        gi = _load(meta_path.with_name(base + ".gi.json")) or {}
        kg = _load(meta_path.with_name(base + ".kg.json")) or {}
        run_root = meta_path.parent.parent
        manifests = sorted(run_root.glob(f"transcripts/{base}*.manifest.json"))
        nodes = gi.get("nodes") or []
        insights = [n for n in nodes if n.get("type") == "Insight"]
        quotes = [n for n in nodes if n.get("type") == "Quote"]
        grounded = [n for n in insights if (n.get("properties") or {}).get("grounded")]
        kg_nodes = kg.get("nodes") or []
        summary = meta.get("summary") or {}
        key = str((meta.get("episode") or {}).get("episode_id") or base)
        episodes[key] = {
            "metadata": meta,
            "manifest": _load(manifests[0]) if manifests else None,
            "llm": {
                "insights": len(insights),
                "grounded_share": round(len(grounded) / len(insights), 3) if insights else None,
                "quotes": len(quotes),
                "summary_bullets": len(summary.get("bullets") or []),
                "summary_text": " ".join(summary.get("bullets") or [])
                or summary.get("raw_text")
                or "",
                "kg_topics": sorted(
                    str((n.get("properties") or {}).get("label") or n.get("id"))
                    for n in kg_nodes
                    if n.get("type") == "Topic"
                ),
                "kg_entities": sorted(
                    str((n.get("properties") or {}).get("name") or n.get("id"))
                    for n in kg_nodes
                    if n.get("type") in ("Person", "Organization", "Entity")
                ),
            },
        }
    return {"files": files, "episodes": episodes}


def _diff(a: Any, b: Any, path: str, skip: List[str], out: List[str]) -> None:
    if any(path == s or path.startswith(s + ".") for s in skip):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}" if path else str(k)
            if VOLATILE.search(str(k)):
                continue
            if k not in a or k not in b:
                if not any(sub == s or sub.startswith(s + ".") for s in skip):
                    out.append(f"{'+' if k not in a else '-'} {sub}")
                continue
            _diff(a[k], b[k], sub, skip, out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", skip, out)
        return
    if a != b:
        out.append(f"~ {path}: {json.dumps(a)[:80]} -> {json.dumps(b)[:80]}")


def diff_key(line: str) -> str:
    """The field a difference line is about, without its values: ``<episode>: ~ <path>``.

    Two runs that differ on the same field produce the same key, which is what lets a field the
    base disagrees with itself on be recognised as noise.
    """
    parts = line.split(": ", 2)
    return f"{parts[0]}: {parts[1]}" if len(parts) == 3 else line


def compare_deterministic(
    base: Dict[str, Any], cand: Dict[str, Any], allowed: Dict[str, Any], llm_fields: List[str]
) -> List[str]:
    """Differences in what should not move, apart from the allowed list."""
    out: List[str] = []
    new_globs = allowed.get("new_files", [])
    # Normalised again here, not only at collection: results saved by an older run (--reuse) carry
    # names normalised by whatever rule that run had.
    cand_files = {_norm(f) for f in cand["files"]}
    base_files = {_norm(f) for f in base["files"]}
    for f in sorted(cand_files - base_files):
        if not any(fnmatch.fnmatch(Path(f).name, g) for g in new_globs):
            out.append(f"new file: {f}")
    for f in sorted(base_files - cand_files):
        out.append(f"missing file: {f}")
    for key in sorted(set(base["episodes"]) | set(cand["episodes"])):
        b, c = base["episodes"].get(key), cand["episodes"].get(key)
        if b is None or c is None:
            out.append(f"episode {key}: only on {'candidate' if b is None else 'base'}")
            continue
        found: List[str] = []
        _diff(
            b["metadata"],
            c["metadata"],
            "",
            list(allowed.get("metadata.json", [])) + llm_fields,
            found,
        )
        _diff(b["manifest"], c["manifest"], "", list(allowed.get("manifest.json", [])), found)
        out.extend(f"{key}: {d}" for d in found)
    return out


def _band(b1: float, b2: float, floor: float) -> Tuple[float, float]:
    mean = (b1 + b2) / 2
    return mean, max(abs(b1 - b2), 0.1 * abs(mean), floor)


def _overlap(a: Any, b: Any) -> Optional[float]:
    sa = _words(a) if isinstance(a, str) else set(a)
    sb = _words(b) if isinstance(b, str) else set(b)
    if not sa and not sb:
        return None
    return round(len(sa & sb) / len(sa | sb), 3)


def llm_rows(bases: List[Dict[str, Any]], cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The informative table: each metric for base 1, base 2 and the candidate, and whether the
    candidate sits inside the band the two base runs define."""
    rows: List[Dict[str, Any]] = []
    for key in sorted(cand["episodes"]):
        b_eps = [b["episodes"].get(key) for b in bases]
        if any(e is None for e in b_eps):
            continue
        bl = [e["llm"] for e in b_eps]
        cl = cand["episodes"][key]["llm"]
        for metric, floor in (
            ("insights", 1),
            ("grounded_share", 0.1),
            ("quotes", 1),
            ("summary_bullets", 1),
        ):
            vals = [x[metric] for x in bl]
            if cl[metric] is None or any(v is None for v in vals):
                rows.append(
                    {"metric": f"{key} {metric}", "base": vals, "candidate": cl[metric], "ok": True}
                )
                continue
            b2 = vals[1] if len(vals) > 1 else vals[0]
            mean, band = _band(float(vals[0]), float(b2), floor)
            rows.append(
                {
                    "metric": f"{key} {metric}",
                    "base": vals,
                    "candidate": cl[metric],
                    "ok": abs(cl[metric] - mean) <= band,
                }
            )
        for metric in ("summary_text", "kg_topics", "kg_entities"):
            within = _overlap(bl[0][metric], bl[1][metric]) if len(bl) > 1 else None
            across = _overlap(bl[0][metric], cl[metric])
            ok = within is None or across is None or across >= within - 0.1
            rows.append(
                {
                    "metric": f"{key} {metric} overlap",
                    "base": [1.0, within if within is not None else "—"],
                    "candidate": across,
                    "ok": ok,
                }
            )
    return rows
