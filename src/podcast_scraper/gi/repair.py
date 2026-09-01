"""Re-derive GI for episodes carrying a pre-#1657 placeholder — IN PLACE, corpus-driven.

WHY THIS IS NOT A PIPELINE FLAG
Four flag combinations were tried against a copy of a real corpus on 2026-08-16, and none
reprocessed a placeholder episode:

    --skip-existing                          -> "no transcript for ..." (single-feed layout)
    --skip-existing --single-feed-uses-corpus-layout -> transcript found, still skipped, GI:0
    --reprocess-existing-only                -> "no transcript for ..."
    --reprocess-source whisper_transcription -> #925 force path never fires (see #33), skipped

The skip predicates key on the PRESENCE of a transcript/metadata file and never look at GI, so a
placeholder artifact reads as "this episode is done" — and deleting the artifact does not help
either, because the transcript and metadata still satisfy them.

Worse, the pipeline route is structurally wrong for a repair even once forcing works: every run
writes into a FRESH ``run_<timestamp>/`` directory, so a "repaired" episode gains a SECOND
metadata+gi.json while the placeholder survives in the old run dir. That duplicate is not
cosmetic — ``_scan_corpus_metadata_index`` is first-writer-wins (keeps the OLDER) while search's
``merged_episode_gi_paths`` takes the NEWEST, so the two disagree about which artifact is
canonical. One such duplicate already exists in the acceptance corpus, unrelated to any repair.

So this is a standalone pass in the shape of ``enrich-edges``: read the corpus, rewrite the SAME
gi.json path, touch nothing else. No RSS fetch, no run dir, no skip logic, no new artifacts.

FAILURE POLICY — loud, never silently partial
A per-episode failure writes NOTHING for that episode, leaves the placeholder exactly as it was,
records the reason, and continues. The process exits non-zero if any episode failed. An
unrepaired episode therefore stays on the integrity gate's red list; it can never look repaired.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .corpus import find_legacy_placeholder_artifacts, is_legacy_placeholder_artifact

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRepair:
    """One episode's outcome. ``ok=False`` means the placeholder is still on disk."""

    episode_id: str
    gi_path: str
    ok: bool
    insights_before: int = 0
    insights_after: int = 0
    topics_aligned: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None


@dataclass
class RepairReport:
    """Outcome of one ``gi-repair`` run, split so a partial repair cannot read as a success.

    ``failed`` is kept separate from ``repaired`` rather than folded into a single count because
    the exit code is derived from it: any failure leaves that episode's placeholder byte-identical
    on disk, so it must still appear on the integrity gate's red list afterwards.
    """

    repaired: List[EpisodeRepair] = field(default_factory=list)
    failed: List[EpisodeRepair] = field(default_factory=list)
    skipped_dry_run: List[str] = field(default_factory=list)
    #: episode ids explicitly requested that no artifact matched. Only set for the
    #: selection-by-identity path; a placeholder sweep legitimately matches nothing.
    requested_not_found: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # A requested id that matched nothing is a FAILURE, not a quiet note. "Re-derive these
        # N episodes" returning zero and exiting 0 is the silent-success shape this tool exists
        # to catch: an operator comparing before/after would diff against a run that did
        # nothing. Distinct from the sweep, where "no placeholders found" is a real PASS.
        return not self.failed and not self.requested_not_found

    def format(self) -> str:
        """Operator-facing summary: counts first, then every failure named individually."""
        lines = [
            "GI REPAIR",
            f"  repaired : {len(self.repaired)}",
            f"  FAILED   : {len(self.failed)}",
        ]
        if self.skipped_dry_run:
            lines.append(f"  dry-run  : {len(self.skipped_dry_run)} would be repaired")
            for p in self.skipped_dry_run:
                lines.append(f"    {p}")
        for r in self.repaired:
            lines.append(
                f"    OK   {r.episode_id}  {r.insights_before} -> {r.insights_after} insights"
                f"  topics={r.topics_aligned}  {r.duration_s:.1f}s"
            )
        if self.requested_not_found:
            lines.append("")
            lines.append(
                f"  NOT FOUND — {len(self.requested_not_found)} requested episode id(s) matched "
                "no artifact in this corpus:"
            )
            for eid in self.requested_not_found:
                lines.append(f"    {eid}")
        if self.failed:
            lines.append("")
            lines.append("  FAILURES — placeholder left intact, episode still on the gate's list:")
            for r in self.failed:
                lines.append(f"    {r.episode_id}  {r.gi_path}")
                lines.append(f"      {r.error}")
        lines.append("")
        lines.append(f"VERDICT: {'PASS' if self.ok else 'FAIL'}")
        if self.ok and not self.repaired and not self.skipped_dry_run:
            lines.append("  Nothing to repair — no legacy placeholders found.")
        return "\n".join(lines)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _metadata_path_for(gi_path: Path) -> Path:
    """``<run>/metadata/<name>.gi.json`` -> ``<run>/metadata/<name>.metadata.json``."""
    return gi_path.with_name(gi_path.name[: -len(".gi.json")] + ".metadata.json")


def _segments_for(run_dir: Path, transcript_rel: str) -> Optional[List[Dict[str, Any]]]:
    """The sidecar carrying word/segment offsets, preferring the ad-free variant.

    Quote grounding resolves character spans against the transcript it is given, so the segments
    and the transcript text MUST come from the same variant or every span is off.
    """
    base = (run_dir / transcript_rel).with_suffix("")
    for candidate in (
        base.with_name(base.name + ".adfree.segments.json"),
        base.with_name(base.name + ".segments.json"),
    ):
        doc = _read_json(candidate)
        if isinstance(doc, dict):
            segs = doc.get("segments")
            if isinstance(segs, list):
                return segs
        elif candidate.is_file():
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    return raw
            except (OSError, ValueError):
                pass
    return None


def _transcript_text_for(run_dir: Path, transcript_rel: str) -> Tuple[str, str]:
    """``(text, ref)`` — prefers the ad-free transcript, matching the GI build path.

    Returns the SAME variant the segments come from; see ``_segments_for``.
    """
    base = (run_dir / transcript_rel).with_suffix("")
    adfree = base.with_name(base.name + ".adfree.txt")
    if adfree.is_file():
        return adfree.read_text(encoding="utf-8"), adfree.name
    plain = run_dir / transcript_rel
    return plain.read_text(encoding="utf-8"), Path(transcript_rel).name


def _summary_bullets(meta: Dict[str, Any]) -> List[str]:
    """Insight seeds from the episode's stored summary, as the staged GI path uses."""
    summary = meta.get("summary")
    if not isinstance(summary, dict):
        return []
    for key in ("bullets", "key_points", "highlights"):
        vals = summary.get(key)
        if isinstance(vals, list):
            out = [str(v).strip() for v in vals if str(v or "").strip()]
            if out:
                return out
    return []


def repair_episode(
    gi_path: Path,
    cfg: Any,
    *,
    build_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    write_fn: Optional[Callable[..., Any]] = None,
    force_healthy: bool = False,
) -> EpisodeRepair:
    """Re-derive one placeholder artifact, rewriting the SAME path. Never partial.

    ``build_fn`` / ``write_fn`` exist so tests can drive this without a live provider.

    ``force_healthy`` re-derives an artifact that is NOT a legacy placeholder. Default False,
    because for the repair use case that refusal IS the safety property — a corpus-wide sweep
    must never rewrite work that succeeded. It exists for the opposite use case: re-deriving a
    known-good episode deliberately, to measure what a prompt / model / rater change does.
    Such a comparison is only possible on a healthy artifact, so "never touch healthy" and
    "measure a change on a healthy episode" genuinely conflict; this flag separates the two
    rather than weakening the sweep. Opt-in per call, never inferred from other settings.
    """
    started = time.time()
    meta_path = _metadata_path_for(gi_path)
    run_dir = gi_path.parent.parent

    doc = _read_json(gi_path)
    episode_id = str((doc or {}).get("episode_id") or "")
    before = len([n for n in ((doc or {}).get("nodes") or []) if n.get("type") == "Insight"])

    def _fail(msg: str) -> EpisodeRepair:
        return EpisodeRepair(
            episode_id=episode_id,
            gi_path=str(gi_path),
            ok=False,
            insights_before=before,
            duration_s=time.time() - started,
            error=msg,
        )

    if doc is None:
        return _fail("gi.json unreadable")
    if not is_legacy_placeholder_artifact(doc) and not force_healthy:
        return _fail("not a legacy placeholder — refusing to rewrite a healthy artifact")
    if not is_legacy_placeholder_artifact(doc):
        # Say it out loud. This overwrites a good artifact in place, so the audit trail must
        # show the choice was made rather than leaving a silent rewrite to be discovered later.
        logger.warning(
            "gi-repair: re-deriving HEALTHY artifact %s (%d insights) — --force-healthy is set. "
            "The existing artifact is overwritten in place.",
            gi_path,
            before,
        )

    meta = _read_json(meta_path)
    if meta is None:
        return _fail(f"metadata unreadable or absent: {meta_path}")

    content = meta.get("content") or {}
    transcript_rel = content.get("transcript_file_path")
    if not isinstance(transcript_rel, str) or not transcript_rel:
        return _fail("metadata declares no content.transcript_file_path")

    try:
        transcript_text, transcript_ref = _transcript_text_for(run_dir, transcript_rel)
    except OSError as exc:
        return _fail(f"transcript unreadable: {exc}")
    if not transcript_text.strip():
        return _fail("transcript is empty")

    episode_block = meta.get("episode") or {}
    feed_block = meta.get("feed") or {}

    if build_fn is None or write_fn is None:
        from . import build_artifact as _build, write_artifact as _write

        build_fn = build_fn or _build
        write_fn = write_fn or _write

    try:
        payload = build_fn(
            episode_id,
            transcript_text,
            model_version=_model_version(cfg),
            prompt_version="v1",
            podcast_id=feed_block.get("feed_id") or feed_block.get("id"),
            episode_title=episode_block.get("title"),
            publish_date=episode_block.get("published") or episode_block.get("publish_date"),
            transcript_ref=transcript_ref,
            transcript_segments=_segments_for(run_dir, transcript_rel),
            cfg=cfg,
            insight_texts=_summary_bullets(meta) or None,
            feed_id=feed_block.get("feed_id") or feed_block.get("id"),
        )
    except Exception as exc:  # noqa: BLE001 — any build failure must leave the placeholder alone
        return _fail(f"build_artifact failed: {exc.__class__.__name__}: {exc}")

    after = len([n for n in (payload.get("nodes") or []) if n.get("type") == "Insight"])

    # #585/#653: without this the repaired artifact carries bullet-derived topic slugs while
    # every other episode carries KG noun-phrase labels, and the CIL bridge cannot merge them.
    topics = 0
    kg_doc = _read_json(gi_path.with_name(gi_path.name[: -len(".gi.json")] + ".kg.json"))
    if kg_doc is not None:
        from .topic_alignment import align_gi_topics_with_kg

        topics = align_gi_topics_with_kg(payload, kg_doc)

    try:
        write_fn(gi_path, payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"write_artifact failed (artifact left as-is): {exc}")

    return EpisodeRepair(
        episode_id=episode_id,
        gi_path=str(gi_path),
        ok=True,
        insights_before=before,
        insights_after=after,
        topics_aligned=topics,
        duration_s=time.time() - started,
    )


def _model_version(cfg: Any) -> str:
    """Lineage stamped onto a re-derived artifact.

    The import used to be ``..providers.gil_lineage``, which does not exist — the resolver
    lives in ``gi.provenance``. A bare ``except Exception`` swallowed the ModuleNotFoundError,
    so this ALWAYS fell back to ``cfg.summary_model`` and never once resolved real lineage.
    Two ways that bites: without ``--config`` every repaired artifact was stamped "unknown",
    and the fallback names the SUMMARY model while the resolver names the INSIGHT model —
    different whenever the GI provider is not the summariser.

    That matters most for the ``--episode-ids --force-healthy`` path, whose entire purpose is
    to make a prompt/model change measurable: ``model_version`` is the field that tells two
    derivations apart, and it was fabricated. #1657 was itself about a fake lineage stamped
    onto real artifacts; the tool built to repair that was doing the same thing by a different
    route. No try/except now — a broken import must be loud, not silently downgrade provenance.
    """
    from .provenance import resolve_gil_artifact_model_version

    return str(resolve_gil_artifact_model_version(cfg, None) or "unknown")


def repair_placeholder_artifacts(
    corpus_root: Path,
    cfg: Any,
    *,
    dry_run: bool = False,
    audit_path: Optional[Path] = None,
    episode_ids: Optional[List[str]] = None,
    force_healthy: bool = False,
) -> RepairReport:
    """Re-derive artifacts under *corpus_root*, in place.

    Default work-list is every legacy placeholder (selection by damage). Passing
    ``episode_ids`` selects those episodes instead (selection by identity) — the mode that
    makes a prompt / model / rater change measurable, since re-deriving the SAME episode under
    two configurations is the only way to diff them. Healthy artifacts additionally need
    ``force_healthy``; see :func:`repair_episode`.
    """
    if episode_ids:
        from .corpus import find_gi_artifacts_for_episode_ids

        # Normalise ONCE, here, and compare like with like. ``find_gi_artifacts_for_episode_ids``
        # strips its wanted-set, so comparing the RAW caller list against stripped matches let a
        # whitespace-padded id be repaired AND reported not-found — a successful repair exiting
        # 1. Duplicates would likewise be reported missing more than once.
        episode_ids = list(dict.fromkeys(e.strip() for e in episode_ids if e and e.strip()))
        work = find_gi_artifacts_for_episode_ids(corpus_root, episode_ids)
        found = {eid for _p, eid in work}
        missing = [e for e in episode_ids if e not in found]
        if missing:
            # Loud: asking for 3 and silently getting 2 is how a comparison ends up drawn from
            # a different set than the one requested.
            logger.warning(
                "gi-repair: %d of %d requested episode id(s) not found in the corpus: %s",
                len(missing),
                len(episode_ids),
                ", ".join(missing[:5]),
            )
    else:
        missing = []
        work = find_legacy_placeholder_artifacts(corpus_root)
    report = RepairReport()
    report.requested_not_found = list(missing)

    if dry_run:
        # Apply the SAME refusal the real run would, or the preview promises work that then
        # fails: --episode-ids --dry-run without --force-healthy used to print "1 would be
        # repaired" while the identical command without --dry-run exited 1 refusing to touch a
        # healthy artifact. A dry-run that disagrees with the run is worse than none.
        previewable = []
        for gi_path, _eid in work:
            if force_healthy:
                previewable.append(str(gi_path))
                continue
            doc = _read_json(Path(gi_path))
            if doc is not None and is_legacy_placeholder_artifact(doc):
                previewable.append(str(gi_path))
            else:
                logger.info(
                    "gi-repair dry-run: %s is healthy and would be REFUSED (needs --force-healthy)",
                    gi_path,
                )
        report.skipped_dry_run = previewable
        return report

    for gi_path, _episode_id in work:
        result = repair_episode(Path(gi_path), cfg, force_healthy=force_healthy)
        (report.repaired if result.ok else report.failed).append(result)
        logger.info(
            "gi-repair %s %s (%d -> %d insights)",
            "OK" if result.ok else "FAILED",
            result.episode_id or gi_path,
            result.insights_before,
            result.insights_after,
        )
        if audit_path is not None:
            _append_audit(audit_path, result)

    return report


def _append_audit(audit_path: Path, result: EpisodeRepair) -> None:
    """One JSONL row per episode — what a corpus repair must leave behind to be auditable."""
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "episode_id": result.episode_id,
                        "gi_path": result.gi_path,
                        "ok": result.ok,
                        "insights_before": result.insights_before,
                        "insights_after": result.insights_after,
                        "topics_aligned": result.topics_aligned,
                        "duration_s": round(result.duration_s, 3),
                        "error": result.error,
                    }
                )
                + "\n"
            )
    except OSError as exc:  # pragma: no cover - auditing must not fail the repair
        logger.warning("could not append gi-repair audit row: %s", exc)
