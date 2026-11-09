"""The roster pass over speakers a transcript already names: a publisher's ``<v Name>`` cues, or a
provider that diarized natively (moved from ``workflow.episode_processor``, E04 S2).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from podcast_scraper import config
from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.workflow.episode_processor import (
    _capture_stage_exception,
    _detection_report_of,
    _operator_renames,
    _segments_carry_native_speakers,
)

logger = logging.getLogger(__name__)


def _apply_native_speaker_roster(result: dict, cfg: config.Config, job: Any) -> dict:
    """``_roster_native_segments`` for a :class:`TranscriptionJob`. See that function."""
    return _roster_native_segments(
        result,
        cfg,
        idx=getattr(job, "idx", None),
        detected_speaker_names=job.detected_speaker_names,
        metadata_named=job.metadata_named,
        feed_hosts=job.feed_hosts,
        unbindable_hosts=getattr(getattr(job, "episode", None), "unbindable_hosts", None),
        detection_ran=getattr(job, "speaker_detection_ran", None),
        detection_report=_detection_report_of(job),
        speaker_renames=_operator_renames(job),
    )


def _roster_native_segments(
    result: dict,
    cfg: config.Config,
    *,
    idx: Any = None,
    detected_speaker_names: Optional[List[str]] = None,
    metadata_named: Optional[List[str]] = None,
    feed_hosts: Optional[List[str]] = None,
    unbindable_hosts: Optional[List[str]] = None,
    detection_ran: Optional[bool] = None,
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    detection_report: Optional[Dict[str, Any]] = None,
    speaker_renames: Optional[Dict[str, str]] = None,
) -> dict:
    """Route a natively-diarized transcript through the SINGLE role authority (the roster).

    A provider that diarizes server-side (deepgram/moss) tags each segment with a ``speaker`` id
    but assigns names positionally and never computes host/guest roles — the same guess-from-
    ordering class of bug the roster exists to replace. Feed the provider's OWN clustering to the
    roster (``precomputed_diarization`` — no re-diarization, no extra API call) so names AND roles
    come from evidence and land as ``speaker_label`` + ``speaker_role`` on the durable segments,
    exactly like the local-diarizer path.

    The inputs are EXPLICIT rather than read off a ``job``. Taking the whole object made this
    silently depend on a shape only ``TranscriptionJob`` has; passing an ``Episode`` — which
    declares none of those fields — raised ``AttributeError`` and killed every transcript download
    (be1ed96d, reverted). A keyword-only signature cannot be satisfied by accident.

    A PERSON-LIKE ``speaker`` VALUE IS A NAME THE SOURCE STATED, and it is carried through to the
    roster as ``stated_voice_names`` so it can be published as-is. Anonymising it first — which is
    what the cluster ids below are for — threw the answer away and let the roster paint a feed host
    onto a voice the file had already named (ac1751b4, reverted). Bare ids (``Speaker 2``, ints
    from deepgram/moss) are clustered only, exactly as before.

    No-op unless the segments carry a native ``speaker`` id (provider-agnostic, gated on data). A
    roster failure is swallowed so the successfully-transcribed text is never lost. ``cost_usd=0.0``
    so the reused diarization does not fabricate a cloud-diarizer charge in the manifest."""
    if not _segments_carry_native_speakers(result):
        return result

    from podcast_scraper.exceptions import ProviderDependencyError
    from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
    from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

    segs = result.get("segments") or []
    cluster_ids: Dict[str, str] = {}

    def _anon(key: str) -> str:
        if key not in cluster_ids:
            cluster_ids[key] = f"SPEAKER_{len(cluster_ids):02d}"
        return cluster_ids[key]

    dsegs = [
        DiarizationSegment(
            start=float(s.get("start") or 0.0),
            end=float(s.get("end") or 0.0),
            speaker=_anon(str(s.get("speaker"))),
        )
        for s in segs
        if isinstance(s, dict) and s.get("speaker") is not None
    ]
    if not dsegs:
        return result

    # WHICH CLUSTERS THE SOURCE ACTUALLY NAMED. `_anon` maps each distinct raw value to a cluster
    # id; a value that is a person's name is also recorded as a stated name for that cluster. The
    # roster filters these again (ads, non-person labels) — this only decides what was STATED.
    from podcast_scraper.providers.ml.diarization.roster import _is_person_voice_label

    stated_voice_names: Dict[str, str] = {}
    for raw, cluster in cluster_ids.items():
        if _is_person_voice_label(raw):
            stated_voice_names[cluster] = raw.strip()
    diar = DiarizationResult(segments=dsegs, num_speakers=len(cluster_ids), cost_usd=0.0)
    # Preserve ASR-provenance top-level fields; only the segments are re-clustered by the roster.
    clean = dict(result)
    clean["segments"] = [
        {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
        for s in segs
        if isinstance(s, dict)
    ]
    try:
        out = apply_diarization_to_result(
            clean,
            "",
            cfg,
            detected_speaker_names,
            metadata_named=metadata_named,
            precomputed_diarization=diar,
            stated_voice_names=stated_voice_names or None,
            feed_hosts=feed_hosts,
            unbindable_hosts=unbindable_hosts,
            detection_ran=detection_ran,
            episode_title=episode_title,
            episode_description=episode_description,
            detection_report=detection_report,
            speaker_renames=speaker_renames,
        )
        # KEEP WHAT THE SOURCE SAID, beside what we resolved. `speaker_label` is OUR answer and a
        # later `relabel_only` is entitled to re-derive it — that is what relabel is for. The
        # publisher's own label is not ours to re-derive: it is a fact about the file, and
        # `relabel_only` anonymises every identity it finds (`_anon(_identity(s))`), so without
        # this a relabel would strip the names the transcript stated and fall back to the feed.
        out_segs = out.get("segments")
        if isinstance(out_segs, list):
            for src_seg, out_seg in zip(segs, out_segs):
                if not (isinstance(src_seg, dict) and isinstance(out_seg, dict)):
                    continue
                stated_raw = src_seg.get("speaker")
                if stated_raw is not None and _is_person_voice_label(str(stated_raw)):
                    out_seg["stated_speaker"] = str(stated_raw).strip()
        return out
    except (ProviderDependencyError, ValueError, OSError, RuntimeError) as exc:
        logger.warning(
            "[%s] Native-speaker roster pass failed; keeping the provider screenplay: %s",
            idx,
            format_exception_for_log(exc),
        )
        _capture_stage_exception(exc, stage="diarization")
        return result
