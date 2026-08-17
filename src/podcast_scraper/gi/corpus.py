"""GIL corpus I/O: load artifacts for export (NDJSON / merged bundle), mirroring ``kg.corpus``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

from .io import read_artifact

logger = logging.getLogger(__name__)


def load_gi_artifacts(
    paths: List[Path],
    *,
    validate: bool = False,
    strict: bool = False,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load ``.gi.json`` from paths; skip invalid with warning when not strict."""
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        try:
            data = read_artifact(path, validate=validate, strict=strict)
            out.append((path, data))
        except Exception as e:
            if strict:
                raise
            logger.warning(
                "Skip invalid GIL artifact %s: %s",
                path,
                format_exception_for_log(e),
            )
    return out


#: FORENSIC CONSTANT — legacy cleanup only. Nothing in the pipeline produces this any more.
#:
#: The GI pipeline used to emit one fabricated insight with this exact text whenever extraction
#: produced nothing, and that placeholder reached production episodes. #1657 deleted the
#: placeholder outright: an episode with no insights now gets an artifact with no Insight nodes.
#: This string survives ONLY so the artifacts written before that change can still be found and
#: re-derived.
#:
#: Matched on the text because it is the one field stable across every version of the old
#: placeholder — its properties changed twice (grounded/CORE/surface, then ungrounded/FILLER/
#: drop), so a detector keyed on those would miss exactly the oldest artifacts that most need
#: finding.
#:
#: DELETE THIS, and everything below it, once a corpus scan reports ``legacy_placeholders: 0``.
#: It is the last thing in the repo that knows the placeholder ever existed.
LEGACY_PLACEHOLDER_INSIGHT_TEXT = "Summary insight (stub)."

#: Deprecated alias kept for one release so external callers do not break on the rename.
STUB_INSIGHT_TEXT = LEGACY_PLACEHOLDER_INSIGHT_TEXT


def is_legacy_placeholder_artifact(artifact: Dict[str, Any]) -> bool:
    """True when this ``.gi.json`` holds a pre-#1657 placeholder instead of real insights.

    The shape is exact: one Insight node whose text is the legacy placeholder string. An episode
    that genuinely produced a single real insight is NOT a placeholder and must not be swept up
    — the point of re-deriving these is to replace failures, not to redo work that succeeded.
    """
    insights = [
        n
        for n in (artifact.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "Insight"
    ]
    if len(insights) != 1:
        return False
    return (
        str((insights[0].get("properties") or {}).get("text", "")).strip()
        == LEGACY_PLACEHOLDER_INSIGHT_TEXT
    )


def find_legacy_placeholder_artifacts(
    corpus_root: Path,
    *,
    validate: bool = False,
    include_superseded: bool = False,
) -> List[Tuple[Path, str]]:
    """Every legacy-placeholder ``.gi.json`` under *corpus_root*, as ``(path, episode_id)``.

    The work-list for corpus repair. ~112 of 678 production episodes (16.5 %) got a placeholder
    while insight generation failed silently; they are indistinguishable from real episodes
    to anything that only asks "does this episode have GI?", which is what
    ``episode_complete_for_append_resume`` asks. An append-mode re-run will therefore SKIP them
    forever. Re-deriving them needs an explicit list, and this produces it.

    Ordered by path so two runs of the same corpus yield the same work-list.

    SCOPED TO CORPUS MEMBERS by default (newest run per episode). An unscoped rglob also returns
    placeholders sitting in SUPERSEDED run dirs — artifacts the corpus does not serve. Repairing
    those spends an LLM call and ~22s each to rewrite history nobody reads, and it inflates the
    count the operator's go/no-go decision is made on. Pass ``include_superseded=True`` for a
    forensic sweep.
    """
    all_paths = sorted(corpus_root.rglob("*.gi.json"))
    paths = _member_gi_paths(corpus_root) if not include_superseded else None
    if not paths and all_paths:
        # Members resolved to nothing while artifacts DO exist — a corpus whose metadata is
        # absent or shaped differently (the synthetic p01..p09 fixture is exactly this). Scoping
        # must never turn "I cannot tell" into "there is nothing to repair": that is the silent
        # empty result #34 was about, in a work-list instead of a provider.
        if paths is not None:
            logger.warning(
                "gi work-list: corpus membership resolved 0 of %d artifacts under %s; "
                "falling back to an unscoped scan (superseded run dirs may be included)",
                len(all_paths),
                corpus_root,
            )
        paths = all_paths
    out: List[Tuple[Path, str]] = []
    for path, doc in load_gi_artifacts(paths, validate=validate, strict=False):
        if is_legacy_placeholder_artifact(doc):
            out.append((path, str(doc.get("episode_id") or "")))
    return out


def _member_gi_paths(corpus_root: Path) -> Optional[List[Path]]:
    """gi.json paths belonging to CURRENT episodes, or ``None`` if the rule is unavailable.

    Resolves each corpus-member metadata file to its declared artifact, falling back to the
    sibling-name convention search itself uses (``<name>.gi.json`` next to
    ``<name>.metadata.json``) so an artifact that exists but was never declared is still seen.
    """
    try:
        from ..search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode
    except Exception:  # noqa: BLE001 - the work-list must still build without the search extra
        return None

    all_meta = sorted(corpus_root.rglob("*.metadata.json"))
    try:
        members = dedupe_metadata_paths_newest_run_per_episode(corpus_root, all_meta)
    except Exception:  # noqa: BLE001
        return None

    out: List[Path] = []
    for meta_path in members:
        meta_path = Path(meta_path)
        declared = None
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
            block = doc.get("grounded_insights") if isinstance(doc, dict) else None
            if isinstance(block, dict) and block.get("artifact_path"):
                declared = (meta_path.parent.parent / str(block["artifact_path"])).resolve()
        except (OSError, ValueError):
            declared = None
        sibling = meta_path.with_name(meta_path.name[: -len(".metadata.json")] + ".gi.json")
        for candidate in (declared, sibling):
            if candidate is not None and Path(candidate).is_file():
                out.append(Path(candidate))
                break
    return sorted(set(out))


def summarize_legacy_placeholder_artifacts(corpus_root: Path) -> Dict[str, Any]:
    """Counts for an operator deciding whether a repair run is worth starting."""
    total = len(sorted(corpus_root.rglob("*.gi.json")))
    found = find_legacy_placeholder_artifacts(corpus_root)
    return {
        "artifacts_total": total,
        "legacy_placeholders": len(found),
        "legacy_placeholder_share": round(len(found) / total, 4) if total else 0.0,
        "episode_ids": [eid for _, eid in found if eid],
        "paths": [str(p) for p, _ in found],
    }


def check_corpus_for_placeholders(corpus_root: Path) -> Tuple[bool, str]:
    """``(ok, formatted_report)`` — ``ok`` is False when ANY legacy placeholder survives.

    The operator-facing entrypoint for the repair of #1655. Without it the detection above was
    reachable only from its own unit test, so "are there still placeholders in the corpus?" —
    the exit criterion for the repair — could not be answered with shipped code. Detection that
    only its own test calls is not tooling.

    Shaped after ``corpus_completeness.check_corpus``: same ``(ok, report)`` contract, same
    VERDICT line, so the Make target stays a one-liner and the two gates read alike.

    Non-zero exit on any find is deliberate. This is meant to run twice — before the repair to
    size the work, and after it to prove the work landed — and the second run is the one whose
    failure must be impossible to miss.
    """
    summary = summarize_legacy_placeholder_artifacts(corpus_root)
    count = int(summary["legacy_placeholders"])
    total = int(summary["artifacts_total"])

    lines = [
        f"Corpus: {corpus_root}",
        f"  gi.json artifacts scanned : {total}",
        f"  legacy placeholders found : {count}",
        f"  share                     : {summary['legacy_placeholder_share']:.2%}",
    ]

    if count:
        lines.append("")
        lines.append("  Episodes needing re-derivation:")
        # Full list, not a head -20. This IS the work-list an operator acts on; a truncated
        # one silently under-reports the repair and there is no second place to get it.
        for path, episode_id in find_legacy_placeholder_artifacts(corpus_root):
            lines.append(f"    {episode_id or '(no episode_id)'}  {path}")

    lines.append("")
    lines.append(f"VERDICT: {'PASS' if count == 0 else 'FAIL'}")
    if count == 0 and total:
        lines.append(
            "  No pre-#1657 placeholders remain. Per the note on "
            "LEGACY_PLACEHOLDER_INSIGHT_TEXT, the detection code in this module can now be "
            "deleted."
        )
    elif count == 0:
        lines.append("  NOTE: zero artifacts scanned — check CORPUS_DIR points at a real corpus.")

    return count == 0, "\n".join(lines)


def export_ndjson(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    output_dir: Optional[Path],
    stream_write: Callable[[str], None],
) -> None:
    """Write one JSON object per line; each includes ``_artifact_path``."""
    for path, art in loaded:
        row = dict(art)
        rel = str(path)
        if output_dir:
            try:
                rel = str(path.relative_to(output_dir))
            except ValueError:
                pass
        row["_artifact_path"] = rel
        stream_write(json.dumps(row, ensure_ascii=False) + "\n")


def export_merged_json(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    output_dir: Optional[Path],
) -> Dict[str, Any]:
    """Single JSON document with all GIL artifacts (corpus bundle)."""
    artifacts: List[Dict[str, Any]] = []
    total_insights = 0
    total_quotes = 0
    for path, art in loaded:
        copy = dict(art)
        rel = str(path)
        if output_dir:
            try:
                rel = str(path.relative_to(output_dir))
            except ValueError:
                pass
        copy["_artifact_path"] = rel
        artifacts.append(copy)
        nodes = art.get("nodes") or []
        for n in nodes:
            t = n.get("type")
            if t == "Insight":
                total_insights += 1
            elif t == "Quote":
                total_quotes += 1
    return {
        "export_kind": "gi_corpus_bundle",
        "schema_version": "1.0",
        "artifact_count": len(artifacts),
        "insight_count_total": total_insights,
        "quote_count_total": total_quotes,
        "artifacts": artifacts,
    }
