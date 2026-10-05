"""The one-page verdict. Holes and differences are listed; what passed is counted, not listed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .compare import AREAS, CheckResult

_KIND_TITLE = {
    "hole": "Decisions that did not resolve as expected",
    "variant": "Variants that should behave identically but do not",
    "value": "Pinned values that do not hold",
    "base": "Stages that differ from the base ref",
}


def render(
    results: List[CheckResult],
    *,
    base_label: Optional[str],
    candidate_label: str,
    corpus: str = "",
    llm: Optional[Dict[str, Any]] = None,
) -> str:
    verdict = all(r.passed for r in results)
    lines: List[str] = [
        "# pipeline-check report",
        "",
        f"**Verdict: {'PASS' if verdict else 'FAIL'}** — candidate `{candidate_label}`"
        + (f" vs base `{base_label}`" if base_label else " (no base ref)"),
        "",
        f"Input corpus (same for both sides): `{corpus}`" if corpus else "",
        "",
    ]
    for r in results:
        status = (
            "report only"
            if r.report_only
            else ("PASS" if r.passed else f"FAIL ({len(r.findings)})")
        )
        lines += [
            f"## Check `{r.name}` — {status}",
            "",
            f"{r.episodes} episode(s) × {len(r.variants)} variant(s): "
            + ", ".join(repr(v) for v in r.variants),
            "",
        ]
        if r.report_only:
            lines += ["| Variants | Episodes/stages that differ |", "| --- | --- |"]
            for pair, diffs in r.report_only.items():
                lines.append(
                    f"| {pair} | {len(diffs)}{': ' + ', '.join(diffs[:8]) if diffs else ''} |"
                )
            lines.append("")
            continue

        lines += ["| Area | Decisions | Holes | Outputs |", "| --- | --- | --- | --- |"]
        holes_by_area: Dict[str, int] = {}
        for f in r.findings:
            if f.kind == "hole":
                holes_by_area[f.area] = holes_by_area.get(f.area, 0) + 1
        shown = [
            a
            for a in AREAS.values()
            if a in r.decisions_per_area or a in holes_by_area or a in r.stage_status
        ]
        for area in shown:
            lines.append(
                f"| {area} | {r.decisions_per_area.get(area, 0)} | {holes_by_area.get(area, 0)} "
                f"| {r.stage_status.get(area, '—')} |"
            )
        lines.append("")

        for kind, title in _KIND_TITLE.items():
            found = [f for f in r.findings if f.kind == kind]
            if not found:
                continue
            lines += [f"**{title}** ({len(found)}):", ""]
            for f in found[:25]:
                lines.append(f"- {f.area} — `{f.where}`: {f.detail}")
            if len(found) > 25:
                lines.append(f"- … {len(found) - 25} more in the JSON")
            lines.append("")

        cov = r.coverage
        lines += [
            f"**Coverage:** {cov.get('exercised', 0)} of {cov.get('discovered', 0)} "
            "decision points exercised. "
            f"{len(cov.get('import_time_copies', []))} values were chosen at import time "
            "(checked statically, not traced). "
            + (
                f"{len(cov.get('skipped_modules', {}))} module(s) could not be imported here "
                "and were not traced."
                if cov.get("skipped_modules")
                else ""
            ),
            "",
        ]

    lines += ["## LLM variation (informative)", ""]
    if not llm:
        lines += ["Not run (REAL=0).", ""]
    else:
        lines += [
            "| Metric | Base 1 | Base 2 | Candidate | Within band? |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in llm["rows"]:
            lines.append(
                f"| {row['metric']} | {row['base'][0]} "
                f"| {row['base'][1] if len(row['base']) > 1 else '—'} "
                f"| {row['candidate']} | {'OK' if row['ok'] else 'LOOK AT THIS'} |"
            )
        lines.append("")

    lines += [
        "## Not covered",
        "",
        "- LLM output quality (only variation is measured, and only with REAL=1).",
        "- Search quality on an existing corpus.",
        "- Decision points not exercised by the driven stages (listed in the JSON).",
        "",
    ]
    return "\n".join(lines)
