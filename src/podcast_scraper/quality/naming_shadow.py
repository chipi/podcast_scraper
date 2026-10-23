"""The naming shadow report: what the side namers and the new ladder rules did on served episodes.

Read only, from the speaker-diagnostics sidecars the pipeline writes (coding plan C9,
podcast-scraper-eval-data ``docs/wip/NAMING-GATES-2026-10-22.md``). Three tables:

- **Side namers.** Per namer that wrote a sidecar key (``transcript_naming``, the LLM pass, staged
  or bundled; ``voice_identity``, the fingerprint rung): episodes, calls that answered, and per
  voice agree / adds / removes / changes against the ladder's published name. Every non-agree
  voice is listed, so a reader can judge it; the counts alone say nothing about who is right.
- **Talk floor.** Every name the floor took off (the decision trace's ``talk_floor`` /
  ``removed`` step), by the role the LLM pass gave that voice. The role is a PROXY for whether
  the name was the voice's own (an ``ad`` / ``promo`` voice losing a name is what the floor is
  for; a ``host`` losing one is a loss to read), not a label.
- **Rescues.** Every voice published by E1d (``e1d_rescue``) or S30s (``s30_split_merge``), with
  the LLM pass's name for the same voice beside it.

Episodes written before these keys existed are counted as ``without``, never as agreement.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..upgrade.corpus_selection import select_served_artifacts
from ..upgrade.migrations.m0025_one_person_one_entry import _load, _run_relative

NAMERS = ("transcript_naming", "voice_identity")
OUTCOMES = ("agree", "adds", "removes", "changes")
RESCUE_SOURCES = ("e1d_rescue", "s30_split_merge")
#: The result key each namer's per-voice entry carries its own name under.
_RESULT_KEY = {"transcript_naming": "model_result", "voice_identity": "fingerprint_result"}


def served_diagnostics(root: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """``(episode id, metadata path, diagnostics)`` for every served episode with a sidecar."""
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        meta = _load(meta_path)
        if not isinstance(meta, dict):
            continue
        eid = str((meta.get("episode") or {}).get("episode_id") or "")
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), meta_path.parent.parent
        )
        if not eid or not rel.endswith(".txt"):
            continue
        diag = _load(meta_path.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
        if isinstance(diag, dict):
            yield eid, meta_path, diag


def _model_view(diag: Mapping[str, Any], voice: str) -> Dict[str, Any]:
    """The LLM pass's raw answer for one voice (``{}`` when it did not run or name it)."""
    entry = ((diag.get("transcript_naming") or {}).get("voices") or {}).get(voice) or {}
    model = entry.get("model")
    return model if isinstance(model, dict) else {}


def namer_rows(eid: str, diag: Mapping[str, Any], namer: str) -> List[Dict[str, Any]]:
    """One row per voice a namer compared with the ladder (empty when it wrote nothing)."""
    block = diag.get(namer)
    if not isinstance(block, dict):
        return []
    out: List[Dict[str, Any]] = []
    for voice, entry in (block.get("voices") or {}).items():
        if not isinstance(entry, dict) or entry.get("outcome") not in OUTCOMES:
            continue
        result = entry.get(_RESULT_KEY[namer]) or {}
        out.append(
            {
                "episode": eid,
                "voice": voice,
                "outcome": entry["outcome"],
                "ladder": (entry.get("ladder") or {}).get("name"),
                "namer": result.get("name"),
                "namer_role": result.get("role"),
            }
        )
    return out


def floor_rows(eid: str, diag: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every name the talk floor took off, with the LLM pass's role for that voice."""
    trace = diag.get("decision_trace") or {}
    out: List[Dict[str, Any]] = []
    for voice, steps in (trace.get("voices") or {}).items():
        for step in steps or []:
            if step.get("rung") == "talk_floor" and step.get("decision") == "removed":
                model = _model_view(diag, voice)
                out.append(
                    {
                        "episode": eid,
                        "voice": voice,
                        "name": step.get("name"),
                        "talk_share": step.get("talk_share"),
                        "llm_role": model.get("role") if model else None,
                        "llm_name": model.get("name") if model else None,
                    }
                )
    return out


def rescue_rows(eid: str, diag: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every voice a ladder rescue published, with the LLM pass's name for it."""
    out: List[Dict[str, Any]] = []
    for v in diag.get("voices") or []:
        if v.get("source") in RESCUE_SOURCES and v.get("named"):
            model = _model_view(diag, str(v.get("voice")))
            out.append(
                {
                    "episode": eid,
                    "voice": v.get("voice"),
                    "rule": v.get("source"),
                    "name": v.get("resolved_name"),
                    "role": v.get("role"),
                    "llm_name": model.get("name") if model else None,
                }
            )
    return out


def report(root: Path) -> Dict[str, Any]:
    """The three tables over every served episode of ``root``."""
    episodes = 0
    namers: Dict[str, Dict[str, Any]] = {
        n: {"episodes": 0, "answered": 0, "outcomes": Counter(), "differences": []} for n in NAMERS
    }
    floor: List[Dict[str, Any]] = []
    rescues: List[Dict[str, Any]] = []
    for eid, _meta, diag in served_diagnostics(root):
        episodes += 1
        for n in NAMERS:
            if not isinstance(diag.get(n), dict):
                continue
            t = namers[n]
            t["episodes"] += 1
            rows = namer_rows(eid, diag, n)
            if diag[n].get("ok", True) and rows:
                t["answered"] += 1
            for r in rows:
                t["outcomes"][r["outcome"]] += 1
                if r["outcome"] != "agree":
                    t["differences"].append(r)
        floor.extend(floor_rows(eid, diag))
        rescues.extend(rescue_rows(eid, diag))
    for t in namers.values():
        t["outcomes"] = {o: t["outcomes"].get(o, 0) for o in OUTCOMES}
        t["without"] = episodes - t["episodes"]
    return {
        "episodes": episodes,
        "namers": namers,
        "floor": {
            "removed": len(floor),
            "by_llm_role": dict(Counter(str(r["llm_role"]) for r in floor).most_common()),
            "rows": floor,
        },
        "rescues": {
            "fires": len(rescues),
            "by_rule": dict(Counter(str(r["rule"]) for r in rescues).most_common()),
            "llm_agrees": sum(
                1 for r in rescues if (r["llm_name"] or "").lower() == (r["name"] or "").lower()
            ),
            "rows": rescues,
        },
    }


def render(rep: Mapping[str, Any], limit: Optional[int] = None) -> str:
    """The report as plain text tables (``limit`` caps each listed table's rows)."""
    lines = [f"served episodes with a speaker sidecar: {rep['episodes']}", ""]
    for n, t in rep["namers"].items():
        o = t["outcomes"]
        lines.append(
            f"{n}: {t['episodes']} episodes ({t['without']} without), {t['answered']} answered; "
            + ", ".join(f"{k} {o[k]}" for k in OUTCOMES)
        )
        for r in t["differences"][:limit]:
            lines.append(
                f"  {r['outcome']:<8} {r['episode']} {r['voice']}: "
                f"ladder {r['ladder']!r} -> {n} {r['namer']!r} ({r['namer_role']})"
            )
        lines.append("")
    f = rep["floor"]
    lines.append(
        f"talk floor: {f['removed']} names removed; by the LLM pass's role {f['by_llm_role']}"
    )
    for r in f["rows"][:limit]:
        lines.append(
            f"  {r['episode']} {r['voice']}: {r['name']!r} at {r['talk_share']} "
            f"(LLM: {r['llm_role']}, {r['llm_name']!r})"
        )
    lines.append("")
    s = rep["rescues"]
    lines.append(
        f"rescues: {s['fires']} fires {s['by_rule']}; the LLM pass names the same person "
        f"{s['llm_agrees']} times"
    )
    for r in s["rows"][:limit]:
        lines.append(
            f"  {r['rule']:<16} {r['episode']} {r['voice']}: {r['name']!r} ({r['role']}); "
            f"LLM {r['llm_name']!r}"
        )
    return "\n".join(lines) + "\n"
