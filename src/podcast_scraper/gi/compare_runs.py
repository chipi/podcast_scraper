"""Compare GIL outcomes between two pipeline run directories (reference vs candidate).

Loads ``*.gi.json`` under ``<run_root>/metadata`` (or ``run_root`` if there is no
``metadata`` child). Used by ``scripts/tools/compare_gil_runs.py`` and unit tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GilArtifactStats:
    """Counts derived from one ``gi.json`` payload."""

    episode_id: str
    path: str
    n_insights: int
    n_grounded_insights: int
    n_quotes: int


def _stats_from_payload(data: dict, path: Path) -> GilArtifactStats:
    eid = (data.get("episode_id") or "").strip()
    if not eid:
        eid = path.stem
    nodes = data.get("nodes") or []
    n_ins = sum(1 for n in nodes if n.get("type") == "Insight")
    n_gr = sum(
        1
        for n in nodes
        if n.get("type") == "Insight" and (n.get("properties") or {}).get("grounded") is True
    )
    n_q = sum(1 for n in nodes if n.get("type") == "Quote")
    return GilArtifactStats(
        episode_id=eid,
        path=str(path),
        n_insights=n_ins,
        n_grounded_insights=n_gr,
        n_quotes=n_q,
    )


def collect_gil_stats_from_run_root(run_root: Path) -> Dict[str, GilArtifactStats]:
    """Map ``episode_id`` → stats for each ``*.gi.json`` under the run."""
    meta = run_root / "metadata"
    base = meta if meta.is_dir() else run_root
    if not base.is_dir():
        return {}
    out: Dict[str, GilArtifactStats] = {}
    for path in sorted(base.glob("*.gi.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            st = _stats_from_payload(data, path)
            out[st.episode_id] = st
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return out


def paired_episode_rows(
    reference: Dict[str, GilArtifactStats],
    candidate: Dict[str, GilArtifactStats],
) -> List[Tuple[str, Optional[GilArtifactStats], Optional[GilArtifactStats]]]:
    """Stable rows: ``(episode_id, ref_stats_or_none, cand_stats_or_none)``."""
    ids = sorted(set(reference) | set(candidate))
    return [(eid, reference.get(eid), candidate.get(eid)) for eid in ids]


def summarize_agreement(
    rows: List[Tuple[str, Optional[GilArtifactStats], Optional[GilArtifactStats]]],
) -> Dict[str, int]:
    """Episode-level agreement counts (reference has quotes vs candidate has quotes)."""
    both = 0
    ref_only = 0
    cand_only = 0
    neither = 0
    missing_ref = 0
    missing_cand = 0
    for _eid, ref, cand in rows:
        if ref is None:
            missing_ref += 1
        if cand is None:
            missing_cand += 1
        rq = ref.n_quotes if ref else 0
        cq = cand.n_quotes if cand else 0
        r_ok = rq > 0
        c_ok = cq > 0
        if r_ok and c_ok:
            both += 1
        elif r_ok and not c_ok:
            ref_only += 1
        elif c_ok and not r_ok:
            cand_only += 1
        else:
            neither += 1
    return {
        "episodes_compared": len(rows),
        "both_have_quotes": both,
        "reference_only_quotes": ref_only,
        "candidate_only_quotes": cand_only,
        "neither_has_quotes": neither,
        "missing_in_reference": missing_ref,
        "missing_in_candidate": missing_cand,
    }


def format_text_report(
    reference_root: Path,
    candidate_root: Path,
    rows: List[Tuple[str, Optional[GilArtifactStats], Optional[GilArtifactStats]]],
    summary: Dict[str, int],
) -> str:
    """Human-readable table + summary (for stdout or tests)."""
    lines = [
        "GIL run comparison",
        f"  reference:  {reference_root.resolve()}",
        f"  candidate: {candidate_root.resolve()}",
        "",
        f"{'episode_id':<40} {'ref_q':>6} {'ref_g':>6} {'cand_q':>7} {'cand_g':>7}",
        "-" * 72,
    ]
    for eid, ref, cand in rows:
        rq = ref.n_quotes if ref else "-"
        rg = ref.n_grounded_insights if ref else "-"
        cq = cand.n_quotes if cand else "-"
        cg = cand.n_grounded_insights if cand else "-"
        lines.append(f"{eid[:40]:<40} {str(rq):>6} {str(rg):>6} {str(cq):>7} {str(cg):>7}")
    lines.extend(
        [
            "",
            "Summary (quote = at least one Quote node):",
            f"  episodes_compared:        {summary['episodes_compared']}",
            f"  both_have_quotes:       {summary['both_have_quotes']}",
            f"  reference_only_quotes:  {summary['reference_only_quotes']}",
            f"  candidate_only_quotes:  {summary['candidate_only_quotes']}",
            f"  neither_has_quotes:     {summary['neither_has_quotes']}",
            f"  missing_in_reference:   {summary['missing_in_reference']}",
            f"  missing_in_candidate:   {summary['missing_in_candidate']}",
        ]
    )
    return "\n".join(lines)
