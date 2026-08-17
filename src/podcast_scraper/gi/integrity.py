"""Corpus GI integrity: does every episode that claims insights actually have them?

WHY THIS EXISTS SEPARATELY FROM ``check_corpus_for_placeholders``
The placeholder gate asks "does the string 'Summary insight (stub).' appear anywhere?". That is a
NEGATIVE assertion over the files that happen to exist, and it was proven useless as a repair
exit criterion on 2026-08-16: after deleting a placeholder artifact and failing to regenerate it,
the gate reported ``VERDICT: PASS`` for a corpus where that episode had NO GI at all. An operator
whose "repair" is ``rm`` gets a green light and a corpus with holes — a worse outcome than the
defect the gate was built to catch.

So this module asserts the POSITIVE, keyed on episodes rather than on artifacts found by glob:

  1. every episode whose metadata declares a ``grounded_insights`` block must have an artifact
     that resolves, loads, and is not a placeholder;
  2. no episode_id may resolve to more than one ``gi.json`` — that catches the supersede failure
     mode, where a pipeline re-run writes a second artifact into a fresh ``run_<ts>/`` dir and
     leaves the old one on disk. It matters beyond tidiness: ``_scan_corpus_metadata_index`` is
     first-writer-wins (keeps the OLDER entry) while search's ``merged_episode_gi_paths`` takes
     the NEWEST, so duplicates mean two subsystems disagree about which artifact is canonical;
  3. zero-insight artifacts are LEGAL after #1657 — "nothing extracted means nothing returned" —
     so they must not fail, but they are counted and listed. 112 placeholders quietly becoming
     112 empty artifacts would satisfy a naive gate while having re-derived nothing, and that
     number being visible is what prevents declaring victory over an empty corpus;
  4. artifacts are resolved the way the SERVING path resolves them — declared ``artifact_path``
     first, then the sibling-name convention. Search's ``_episode_to_gi_path_from_discovered``
     falls back to ``<name>.gi.json`` beside ``<name>.metadata.json`` and serves whatever it
     finds, so an episode with no ``grounded_insights`` block but an artifact on disk IS being
     served. Judging only the declaration meant a placeholder in that position was filed as
     "GI never ran for this episode" and the gate returned PASS — the same false green as the
     ``rm``-and-pass failure above, reached from the opposite direction. An undeclared artifact
     fails the gate when its CONTENT is bad, and is reported (not failed) when it is fine.

Deliberately no schema validation here beyond "it parses and has the shape we read": this gate
runs before and after a repair on a production corpus and must not fail for reasons unrelated to
the repair.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .corpus import is_legacy_placeholder_artifact, resolve_episode_gi_path

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _episode_id_of(meta: Dict[str, Any]) -> str:
    episode = meta.get("episode")
    if isinstance(episode, dict):
        eid = episode.get("episode_id")
        if isinstance(eid, str) and eid.strip():
            return eid.strip()
    eid = meta.get("episode_id")
    return eid.strip() if isinstance(eid, str) and eid.strip() else ""


def _corpus_member_metadata(corpus_root: Path) -> Tuple[List[Path], bool]:
    """The metadata files that ARE the corpus, per the project's own membership rule.

    ``dedupe_metadata_paths_newest_run_per_episode`` is the single documented rule for
    "which copy of a reprocessed episode counts" — newest ``run_*`` wins per
    ``(feed_id, episode_id)``, with disjoint episodes across runs all surviving. Indexing,
    digest, topic-clusters, enrichment, catalog and staleness all share it specifically so they
    cannot diverge (the 94-vs-106 split-brain).

    A gate that invented its own answer would be the next thing to diverge. This one asks.

    Returns ``(paths, used_canonical_rule)``; the flag lets the report say so, and lets a corpus
    the rule cannot be applied to fall back to a plain scan rather than failing.
    """
    all_paths = sorted(corpus_root.rglob("*.metadata.json"))
    try:
        from ..search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

        return list(dedupe_metadata_paths_newest_run_per_episode(corpus_root, all_paths)), True
    except Exception as exc:  # noqa: BLE001 - a gate must still run without the search extra
        logger.debug("corpus membership rule unavailable (%s); scanning all metadata", exc)
        return all_paths, False


def assess_gi_integrity(corpus_root: Path) -> Dict[str, Any]:
    """Check that the GI each corpus-member episode declares actually exists.

    Metadata-first, not artifact-first: an episode that lost its artifact is invisible to any
    scan that starts from ``rglob("*.gi.json")``, and that invisibility is exactly the hole this
    module closes.

    Scoped to corpus MEMBERS. An episode reprocessed into a newer run dir legitimately leaves an
    older copy on disk — that is the supported incremental-add / reprocess shape, not a defect —
    so superseded copies are excluded here rather than reported as duplicates. An earlier version
    of this gate failed on them and would have failed on any corpus where anything was ever
    reprocessed.
    """
    metadata_paths, used_rule = _corpus_member_metadata(corpus_root)

    unreadable_metadata: List[str] = []
    no_gi_block: List[str] = []
    missing_artifact: List[Dict[str, str]] = []
    unreadable_artifact: List[Dict[str, str]] = []
    placeholders: List[Dict[str, str]] = []
    empty_artifacts: List[Dict[str, str]] = []
    undeclared_artifact: List[Dict[str, str]] = []
    healthy: List[Dict[str, Any]] = []
    by_episode: Dict[str, List[str]] = defaultdict(list)

    for meta_path in metadata_paths:
        meta = _load_json(meta_path)
        if meta is None:
            unreadable_metadata.append(str(meta_path))
            continue

        episode_id = _episode_id_of(meta)
        gi_block = meta.get("grounded_insights")
        declared_rel = (
            str(gi_block["artifact_path"])
            if isinstance(gi_block, dict) and gi_block.get("artifact_path")
            else ""
        )

        # Resolved the way the SERVING path resolves it: declared first, else the sibling-name
        # convention. A gate that only honoured the declaration would clear a corpus that is not
        # the corpus users query — an undeclared artifact is still indexed and still returned.
        artifact, how = resolve_episode_gi_path(meta_path)

        if artifact is None:
            if declared_rel:
                # Metadata claims GI that is not on disk under EITHER resolution.
                claimed = (meta_path.parent.parent / declared_rel).resolve()
                missing_artifact.append(
                    {
                        "episode_id": episode_id,
                        "path": str(claimed),
                        "metadata": str(meta_path),
                    }
                )
            else:
                # GI was never enabled for this episode, or the run predates the block. Not a
                # failure: the corpus legitimately contains episodes without insights.
                no_gi_block.append(episode_id or str(meta_path))
            continue

        entry = {"episode_id": episode_id, "path": str(artifact), "metadata": str(meta_path)}

        if how == "sibling":
            # Served, but undeclared. Recorded whatever its contents, because this is the state
            # in which a placeholder hides from a declaration-keyed gate. Severity comes from the
            # CONTENT below: bad => hard failure (it is being served), good => provenance gap.
            undeclared_artifact.append(entry)

        by_episode[episode_id or str(artifact)].append(str(artifact))

        doc = _load_json(artifact)
        if doc is None:
            unreadable_artifact.append(entry)
            continue

        if is_legacy_placeholder_artifact(doc):
            placeholders.append(entry)
            continue

        insights = [
            n
            for n in (doc.get("nodes") or [])
            if isinstance(n, dict) and n.get("type") == "Insight"
        ]
        if not insights:
            empty_artifacts.append(entry)
        else:
            healthy.append({**entry, "insight_count": len(insights)})

    # Only meaningful when the canonical membership rule could NOT be applied. With the rule on,
    # a reprocessed episode's older copy is already excluded, so a survivor here means two
    # artifacts inside the SAME run dir — which the layout does not produce and which no
    # resolver can arbitrate.
    duplicates = {eid: paths for eid, paths in by_episode.items() if len(paths) > 1}

    return {
        "metadata_scanned": len(metadata_paths),
        "membership_rule_applied": used_rule,
        "unreadable_metadata": unreadable_metadata,
        "episodes_without_gi_block": no_gi_block,
        "missing_artifact": missing_artifact,
        "unreadable_artifact": unreadable_artifact,
        "legacy_placeholders": placeholders,
        "empty_artifacts": empty_artifacts,
        "undeclared_artifact": undeclared_artifact,
        "healthy": healthy,
        "duplicate_episode_ids": duplicates,
    }


def check_corpus_gi_integrity(corpus_root: Path) -> Tuple[bool, str]:
    """``(ok, formatted_report)`` — the repair's real exit criterion.

    Scoped to corpus MEMBERS (newest run per episode). FAILS on: a surviving legacy placeholder,
    a declared-but-missing artifact, an unreadable artifact, or two artifacts for one episode
    that the membership rule could not arbitrate.

    PASSES (but reports) on: episodes with no GI block at all, artifacts with zero insights, and
    healthy artifacts that exist without being declared — all legal states. They appear in the
    NOT-COVERED section so a "clean" verdict cannot hide them.

    An UNDECLARED artifact whose content is bad still fails, because search serves it regardless
    of what the metadata says.
    """
    r = assess_gi_integrity(corpus_root)

    hard_failures = (
        len(r["legacy_placeholders"])
        + len(r["missing_artifact"])
        + len(r["unreadable_artifact"])
        + len(r["duplicate_episode_ids"])
    )
    ok = hard_failures == 0

    scope = (
        "corpus members (newest run per episode)"
        if r["membership_rule_applied"]
        else "ALL metadata — membership rule unavailable, superseded copies included"
    )
    lines = [
        f"Corpus: {corpus_root}",
        f"  scope                       : {scope}",
        f"  metadata files scanned      : {r['metadata_scanned']}",
        f"  episodes with healthy GI    : {len(r['healthy'])}",
        "",
        "  HARD FAILURES",
        f"    legacy placeholders       : {len(r['legacy_placeholders'])}",
        f"    declared but MISSING      : {len(r['missing_artifact'])}",
        f"    unreadable artifact       : {len(r['unreadable_artifact'])}",
        f"    unarbitrated duplicates   : {len(r['duplicate_episode_ids'])}",
    ]

    def _list(title: str, entries: List[Dict[str, str]]) -> None:
        if not entries:
            return
        lines.append("")
        lines.append(f"  {title}")
        for e in entries:
            lines.append(f"    {e.get('episode_id') or '(no episode_id)'}  {e['path']}")

    _list("Placeholders — re-derive these:", r["legacy_placeholders"])
    _list("Missing artifacts — metadata claims GI that is not on disk:", r["missing_artifact"])
    _list("Unreadable artifacts:", r["unreadable_artifact"])

    if r["duplicate_episode_ids"]:
        lines.append("")
        lines.append("  One episode -> several artifacts the membership rule could not arbitrate")
        lines.append("  (same run dir, or unreadable run segment) — no resolver can pick a winner:")
        for eid, paths in r["duplicate_episode_ids"].items():
            lines.append(f"    {eid}")
            for p in paths:
                lines.append(f"      {p}")

    # Equal-weight NOT-COVERED section: silence on these would read as "no gaps".
    lines.append("")
    lines.append("  NOT COVERED BY THIS VERDICT (legal states, reported so they cannot hide)")
    lines.append(
        f"    zero-insight artifacts    : {len(r['empty_artifacts'])}  "
        "(legal post-#1657 — but a repair that produces these instead of real insights has "
        "re-derived NOTHING)"
    )
    for e in r["empty_artifacts"]:
        lines.append(f"      {e.get('episode_id') or '(no episode_id)'}  {e['path']}")
    lines.append(
        f"    episodes with no GI block : {len(r['episodes_without_gi_block'])} "
        "(GI never ran for them)"
    )
    lines.append(
        f"    served but NOT declared   : {len(r['undeclared_artifact'])}  "
        "(no grounded_insights block, but a sibling <name>.gi.json exists — search resolves and "
        "SERVES it, so its contents are judged above; the missing declaration is a provenance gap)"
    )
    for e in r["undeclared_artifact"]:
        lines.append(f"      {e.get('episode_id') or '(no episode_id)'}  {e['path']}")
    lines.append(f"    unreadable metadata       : {len(r['unreadable_metadata'])}")
    for p in r["unreadable_metadata"]:
        lines.append(f"      {p}")

    lines.append("")
    lines.append(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    if ok and r["metadata_scanned"] == 0:
        lines.append("  NOTE: zero metadata files scanned — check CORPUS_DIR points at a corpus.")
    elif ok and not r["healthy"]:
        lines.append(
            "  NOTE: no episode has healthy GI. Passing on an empty result is not a clean bill."
        )

    return ok, "\n".join(lines)
