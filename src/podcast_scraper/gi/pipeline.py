"""GIL extraction pipeline: transcript + metadata -> nodes/edges -> artifact dict.

Builds a valid gi.json with Episode, Insight(s), Quote(s), and SUPPORTED_BY edges.
When cfg is provided and gi_require_grounding is True, uses the evidence stack
via ``quote_extraction_provider`` / ``entailment_provider`` (from callers or
``create_gil_evidence_providers(cfg, summary_provider=...)``). Insight texts come
from insight_texts, insight_provider.generate_insights(), or stub.

When transcript_segments are provided (from transcription with word/segment
timestamps), Quote nodes get precise timestamp_start_ms and timestamp_end_ms
(FR2.2).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from unittest.mock import Mock

from .. import config_constants
from ..exceptions import GILGroundingUnsatisfiedError
from ..graph_id_utils import (
    episode_node_id,
    gil_insight_node_id,
    gil_quote_node_id,
    speaker_node_id,
)
from .grounding import GroundedQuote
from .provenance import resolve_gil_artifact_model_version

if TYPE_CHECKING:
    from podcast_scraper import config

logger = logging.getLogger(__name__)

# Stub insight text used when no real insights (single stub)
_STUB_INSIGHT_TEXT = "Summary insight (stub)."


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    """Read a float from cfg; tolerate MagicMock / bad types in tests."""
    if cfg is None:
        return default
    raw = getattr(cfg, name, default)
    # MagicMock.__float__ returns 1.0; treat mocks as "unset" and use default.
    if isinstance(raw, Mock):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    """Read an int from cfg; tolerate MagicMock / bad types in tests."""
    if cfg is None:
        return default
    raw = getattr(cfg, name, default)
    if isinstance(raw, Mock):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _handle_zero_grounded_quotes(
    *,
    episode_id: str,
    insights: List[str],
    total_grounded: int,
    cfg: Optional["config.Config"],
    pipeline_metrics: Optional[Any],
) -> None:
    """Warn, update metrics, and optionally fail when grounding required but empty."""
    if total_grounded > 0 or not insights:
        return
    if pipeline_metrics is not None and hasattr(
        pipeline_metrics, "gi_episodes_zero_grounded_when_required"
    ):
        pipeline_metrics.gi_episodes_zero_grounded_when_required += 1
    if pipeline_metrics is not None and hasattr(pipeline_metrics, "gi_grounding_degraded"):
        pipeline_metrics.gi_grounding_degraded = True
    logger.warning(
        "GIL: episode %s: gi_require_grounding is True but evidence produced 0 grounded "
        "quotes for %d insight(s)",
        episode_id,
        len(insights),
    )
    if cfg is not None and getattr(cfg, "gi_fail_on_missing_grounding", False):
        raise GILGroundingUnsatisfiedError(
            message=(
                f"No grounded quotes for episode {episode_id} " f"({len(insights)} insight(s))."
            ),
            provider="GIL/grounding",
            suggestion=(
                "Install .[ml] for local NLI, tune thresholds, or set "
                "gi_fail_on_missing_grounding: false."
            ),
        )


def _safe_iso_date(s: Optional[str]) -> str:
    """Return ISO date-time string; use placeholder if missing."""
    if s and s.strip():
        return s.strip()
    return "2020-01-01T00:00:00Z"


def _char_range_to_ms(
    transcript: str,
    char_start: int,
    char_end: int,
    segments: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Map character span to (start_ms, end_ms) using transcription segments.

    Segments are expected to have "start" (seconds), "end" (seconds), "text" (str).
    Builds cumulative character offsets per segment and finds overlapping segment(s)
    for [char_start, char_end]; returns the time range of those segments in ms.

    Args:
        transcript: Full transcript text (used to validate segment text alignment).
        char_start: Quote start character offset (0-based).
        char_end: Quote end character offset (exclusive).
        segments: List of {"start": float, "end": float, "text": str}.

    Returns:
        (timestamp_start_ms, timestamp_end_ms). (0, 0) if no segments or no overlap.
    """
    if not segments or char_start >= char_end:
        return 0, 0
    seg_list = []
    pos = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start_s = float(seg.get("start", 0.0))
        end_s = float(seg.get("end", 0.0))
        text = seg.get("text") or ""
        seg_start = pos
        seg_end = pos + len(text)
        seg_list.append((seg_start, seg_end, start_s, end_s))
        pos = seg_end
    if not seg_list:
        return 0, 0
    # Only map when segment text length matches transcript (no heavy reformatting)
    if abs(len(transcript) - pos) > 50:
        return 0, 0
    # Find segments overlapping [char_start, char_end]; use first overlap for start, last for end
    start_ms = 0
    end_ms = 0
    first_set = False
    for seg_start, seg_end, start_s, end_s in seg_list:
        if seg_end <= char_start or seg_start >= char_end:
            continue
        if not first_set:
            start_ms = int(start_s * 1000)
            first_set = True
        end_ms = int(end_s * 1000)
    return start_ms, end_ms


def _segment_speaker_label(seg: Dict[str, Any]) -> Optional[str]:
    """Return normalized speaker label from a segment dict if present."""
    raw = seg.get("speaker")
    if raw is None:
        raw = seg.get("speaker_id")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _speaker_id_for_char_range(
    transcript: str,
    char_start: int,
    char_end: int,
    segments: List[Dict[str, Any]],
) -> Optional[str]:
    """Map quote character span to a speaker using optional diarization on segments.

    Segments follow the same layout as ``_char_range_to_ms`` (cumulative ``text``).
    When ``speaker`` or ``speaker_id`` is set on overlapping segments, prefers the
    segment containing ``char_start``, else the segment with the largest overlap.
    Returns ``None`` when labels are absent or transcript/segments do not align.
    """
    if not segments or char_start >= char_end:
        return None
    seg_list: List[Tuple[int, int, Dict[str, Any]]] = []
    pos = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text") or ""
        seg_start = pos
        seg_end = pos + len(text)
        seg_list.append((seg_start, seg_end, seg))
        pos = seg_end
    if not seg_list:
        return None
    if abs(len(transcript) - pos) > 50:
        return None
    for seg_start, seg_end, seg in seg_list:
        if seg_start <= char_start < seg_end:
            sp = _segment_speaker_label(seg)
            if sp:
                return sp
            break
    best_overlap = 0
    best: Optional[str] = None
    for seg_start, seg_end, seg in seg_list:
        if seg_end <= char_start or seg_start >= char_end:
            continue
        ov = min(seg_end, char_end) - max(seg_start, char_start)
        sp = _segment_speaker_label(seg)
        if sp and ov > best_overlap:
            best_overlap = ov
            best = sp
    return best


def _attach_speaker_for_quote(
    nodes: list,
    edges: list,
    episode_id: str,
    quote_id: str,
    speaker_label: Optional[str],
    speakers_added: Set[str],
) -> None:
    """Add Speaker node and SPOKEN_BY (Quote -> Speaker) when diarization label exists."""
    if not speaker_label or not str(speaker_label).strip():
        return
    raw = str(speaker_label).strip()
    sid = speaker_node_id(raw)
    if sid not in speakers_added:
        nodes.append(
            {
                "id": sid,
                "type": "Speaker",
                "properties": {"name": raw},
            }
        )
        speakers_added.add(sid)
    edges.append({"type": "SPOKEN_BY", "from": quote_id, "to": sid})


def _build_stub_artifact(
    episode_id: str,
    transcript_text: str,
    *,
    model_version: str,
    prompt_version: str,
    podcast_id: str,
    episode_title: str,
    date_str: str,
    transcript_ref: str,
) -> Dict[str, Any]:
    """Build minimal stub artifact (one Episode, one Insight, one Quote, one SUPPORTED_BY)."""
    ep_node_id = episode_node_id(episode_id)
    stub_insight_text = _STUB_INSIGHT_TEXT
    insight_id = gil_insight_node_id(episode_id, 0, stub_insight_text)
    text_slice = (transcript_text or "No transcript.").strip()[:100]
    char_start = 0
    char_end = max(0, len(text_slice) - 1)
    quote_id = gil_quote_node_id(episode_id, 0, text_slice, char_start, char_end)

    nodes: list = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": {
                "podcast_id": podcast_id,
                "title": episode_title,
                "publish_date": date_str,
            },
        },
        {
            "id": insight_id,
            "type": "Insight",
            "properties": {
                "text": _STUB_INSIGHT_TEXT,
                "episode_id": episode_id,
                "grounded": True,
            },
        },
        {
            "id": quote_id,
            "type": "Quote",
            "properties": {
                "text": text_slice,
                "episode_id": episode_id,
                "speaker_id": None,
                "char_start": char_start,
                "char_end": char_end,
                "timestamp_start_ms": 0,
                "timestamp_end_ms": 0,
                "transcript_ref": transcript_ref,
            },
        },
    ]
    edges: list = [
        {"type": "HAS_INSIGHT", "from": ep_node_id, "to": insight_id},
        {"type": "SUPPORTED_BY", "from": insight_id, "to": quote_id},
    ]
    return {
        "schema_version": "1.0",
        "model_version": model_version,
        "prompt_version": prompt_version,
        "episode_id": episode_id,
        "nodes": nodes,
        "edges": edges,
    }


def _resolve_insight_texts(
    transcript_text: str,
    cfg: Optional["config.Config"],
    insight_texts: Optional[List[str]] = None,
    insight_provider: Optional[Any] = None,
    episode_title: Optional[str] = None,
    pipeline_metrics: Optional[Any] = None,
) -> List[str]:
    """Resolve list of insight strings for GIL from config, provider, or stub.

    Order: use insight_texts if non-empty; else if gi_insight_source=provider
    and provider has generate_insights, call it; else if single stub desired,
    return [_STUB_INSIGHT_TEXT]. Normalizes and caps length per gi_max_insights.
    """
    max_insights = config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
    if cfg is not None:
        max_insights = getattr(
            cfg, "gi_max_insights", config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
        )

    if insight_texts:
        # Normalize: strip, drop empty, cap
        resolved = [s.strip() for s in insight_texts if (s and s.strip())][:max_insights]
        if resolved:
            return resolved

    source = "stub"
    if cfg is not None:
        source = getattr(cfg, "gi_insight_source", "stub")

    if source == "provider" and insight_provider is not None:
        gen = getattr(insight_provider, "generate_insights", None)
        if callable(gen):
            try:
                out = gen(
                    text=transcript_text or "",
                    episode_title=episode_title,
                    max_insights=max_insights,
                    params=None,
                    pipeline_metrics=pipeline_metrics,
                )
                if isinstance(out, list):
                    resolved = [s.strip() for s in out if (s and s.strip())][:max_insights]
                    if resolved:
                        return resolved
            except Exception as e:
                logger.debug(
                    "generate_insights failed, falling back to stub: %s",
                    e,
                    exc_info=True,
                )

    return [_STUB_INSIGHT_TEXT]


def build_artifact(
    episode_id: str,
    transcript_text: str,
    *,
    model_version: str = "stub",
    prompt_version: str = "v1",
    podcast_id: Optional[str] = None,
    episode_title: Optional[str] = None,
    publish_date: Optional[str] = None,
    transcript_ref: str = "transcript.txt",
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional["config.Config"] = None,
    insight_texts: Optional[List[str]] = None,
    insight_provider: Optional[Any] = None,
    quote_extraction_provider: Optional[Any] = None,
    entailment_provider: Optional[Any] = None,
    summary_provider: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    gil_created_evidence_providers: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build a GIL artifact for one episode.

    Insight texts come from insight_texts (e.g. summary bullets), from
    insight_provider.generate_insights() when gi_insight_source=provider, or
    a single stub. For each insight, provider-based QA + NLI finds grounded
    quotes; artifact has one Insight node per insight and their
    Quote nodes + SUPPORTED_BY edges.

    When transcript_segments is provided (from transcription with segments),
    Quote nodes get precise timestamp_start_ms and timestamp_end_ms (FR2.2).

    Args:
        episode_id: Stable episode key (RSS GUID family). Episode node id is
            ``episode:{episode_id}``; do not pass a value that already includes the
            ``episode:`` prefix.
        transcript_text: Full transcript text (used for QA context and quote spans).
        model_version: Fallback model id when ``cfg`` is omitted; when ``cfg`` is set,
            ``gi.json`` ``model_version`` is derived from ``gi_insight_source`` and the
            summarization / insight provider (see ``gi.provenance``).
        prompt_version: Prompt version tag.
        podcast_id: Optional podcast node ID for Episode.podcast_id.
        episode_title: Optional title for Episode node.
        publish_date: Optional ISO date-time for Episode node.
        transcript_ref: Reference string for Quote.transcript_ref.
        transcript_segments: Optional list of {"start", "end", "text"} from transcription.
        cfg: Optional config; when set and gi_require_grounding, use evidence stack.
        insight_texts: Optional precomputed list of insight strings (e.g. summary bullets).
        insight_provider: Optional provider with generate_insights() when source=provider.
        quote_extraction_provider: Optional provider with extract_quotes() for GIL QA.
        entailment_provider: Optional provider with score_entailment() for GIL NLI.
        summary_provider: When quote/entail providers are omitted, reused if cfg matches
            summary_provider (see ``create_gil_evidence_providers``).
        pipeline_metrics: Optional metrics; when set, evidence path counters are updated.
        gil_created_evidence_providers: When set, instantiated evidence providers (not equal
            to ``summary_provider``) are appended for caller cleanup (e.g. ``.cleanup()``).

    Returns:
        Dict with schema_version, model_version, prompt_version, episode_id, nodes, edges.
    """
    pid = podcast_id or "podcast:unknown"
    title = (episode_title or "Episode").strip() or "Episode"
    date_str = _safe_iso_date(publish_date)

    insights = _resolve_insight_texts(
        transcript_text=transcript_text or "",
        cfg=cfg,
        insight_texts=insight_texts,
        insight_provider=insight_provider,
        episode_title=episode_title or title,
        pipeline_metrics=pipeline_metrics,
    )

    artifact_model_version = model_version
    if cfg is not None:
        raw_gi_src = getattr(cfg, "gi_insight_source", "stub")
        if isinstance(raw_gi_src, str):
            gi_src_norm = raw_gi_src.strip().lower() or "stub"
        else:
            gi_src_norm = "stub"
        lineage_provider = summary_provider or insight_provider
        artifact_model_version = resolve_gil_artifact_model_version(
            cfg,
            lineage_provider,
            gi_insight_source=gi_src_norm,
        )

    use_evidence_stack = (
        cfg is not None
        and getattr(cfg, "generate_gi", False)
        and getattr(cfg, "gi_require_grounding", True)
        and (transcript_text or "").strip()
    )

    if use_evidence_stack and insights:
        assert cfg is not None
        try:
            from .deps import create_gil_evidence_providers
            from .grounding import (
                find_grounded_quotes_via_providers,
                NLI_ENTAILMENT_MIN,
                QA_SCORE_MIN,
            )

            q_prov = quote_extraction_provider
            e_prov = entailment_provider
            if q_prov is None or e_prov is None:
                q_new, e_new = create_gil_evidence_providers(cfg, summary_provider=summary_provider)
                if quote_extraction_provider is None:
                    q_prov = q_new
                    if (
                        gil_created_evidence_providers is not None
                        and q_prov is not summary_provider
                        and hasattr(q_prov, "cleanup")
                    ):
                        gil_created_evidence_providers.append(q_prov)
                else:
                    q_prov = quote_extraction_provider
                if entailment_provider is None:
                    e_prov = e_new
                    if (
                        gil_created_evidence_providers is not None
                        and e_prov is not summary_provider
                        and e_prov is not q_prov
                        and hasattr(e_prov, "cleanup")
                    ):
                        gil_created_evidence_providers.append(e_prov)
                else:
                    e_prov = entailment_provider
            else:
                q_prov = quote_extraction_provider
                e_prov = entailment_provider

            insight_quotes: List[List[GroundedQuote]] = []
            _er = max(0, _cfg_int(cfg, "gi_evidence_extract_retries", 0))
            qa_min = _cfg_float(cfg, "gi_qa_score_min", QA_SCORE_MIN)
            nli_min = _cfg_float(cfg, "gi_nli_entailment_min", NLI_ENTAILMENT_MIN)
            for it in insights:
                grounded = find_grounded_quotes_via_providers(
                    transcript=transcript_text.strip(),
                    insight_text=it,
                    quote_extraction_provider=q_prov,
                    entailment_provider=e_prov,
                    qa_score_min=qa_min,
                    nli_entailment_min=nli_min,
                    pipeline_metrics=pipeline_metrics,
                    extract_retries=_er,
                )
                insight_quotes.append(grounded if isinstance(grounded, list) else [])
            total_grounded = sum(len(q) for q in insight_quotes)
            _handle_zero_grounded_quotes(
                episode_id=episode_id,
                insights=insights,
                total_grounded=total_grounded,
                cfg=cfg,
                pipeline_metrics=pipeline_metrics,
            )
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "gi_evidence_stack_completed"
            ):
                pipeline_metrics.gi_evidence_stack_completed += 1
            return _artifact_from_multi_insight(
                episode_id=episode_id,
                insight_texts=insights,
                insight_quotes_list=insight_quotes,
                model_version=artifact_model_version,
                prompt_version=prompt_version,
                podcast_id=pid,
                episode_title=title,
                date_str=date_str,
                transcript_ref=transcript_ref,
                transcript_text=transcript_text or "",
                transcript_segments=transcript_segments,
            )
        except GILGroundingUnsatisfiedError:
            raise
        except Exception as e:
            logger.debug(
                "GIL evidence stack (provider path) failed, using stub/degraded artifact: %s",
                e,
                exc_info=True,
            )

    # Single stub path (no evidence stack or fallback)
    if len(insights) == 1 and insights[0] == _STUB_INSIGHT_TEXT:
        return _build_stub_artifact(
            episode_id=episode_id,
            transcript_text=transcript_text,
            model_version=artifact_model_version,
            prompt_version=prompt_version,
            podcast_id=pid,
            episode_title=title,
            date_str=date_str,
            transcript_ref=transcript_ref,
        )

    # Multiple insights but evidence stack failed: still emit multi-insight artifact
    # with no quotes (grounded=False per insight)
    try:
        return _artifact_from_multi_insight(
            episode_id=episode_id,
            insight_texts=insights,
            insight_quotes_list=[[]] * len(insights),
            model_version=artifact_model_version,
            prompt_version=prompt_version,
            podcast_id=pid,
            episode_title=title,
            date_str=date_str,
            transcript_ref=transcript_ref,
            transcript_text=transcript_text or "",
            transcript_segments=transcript_segments,
        )
    except Exception:
        return _build_stub_artifact(
            episode_id=episode_id,
            transcript_text=transcript_text,
            model_version=artifact_model_version,
            prompt_version=prompt_version,
            podcast_id=pid,
            episode_title=title,
            date_str=date_str,
            transcript_ref=transcript_ref,
        )


def _artifact_from_multi_insight(
    episode_id: str,
    insight_texts: List[str],
    insight_quotes_list: List[List[GroundedQuote]],
    *,
    model_version: str,
    prompt_version: str,
    podcast_id: str,
    episode_title: str,
    date_str: str,
    transcript_ref: str,
    transcript_text: Optional[str] = None,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build artifact from Episode + N Insights + their grounded quote lists.

    When transcript_text and transcript_segments are provided, Quote nodes get
    precise timestamp_start_ms and timestamp_end_ms (FR2.2).
    """
    ep_node_id = episode_node_id(episode_id)
    nodes: list = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": {
                "podcast_id": podcast_id,
                "title": episode_title,
                "publish_date": date_str,
            },
        },
    ]
    edges: list = []
    quote_global_idx = 0
    speakers_added: Set[str] = set()
    use_segments = bool(transcript_text and transcript_segments and len(transcript_segments) > 0)

    # Pad so we have one quote list per insight
    while len(insight_quotes_list) < len(insight_texts):
        insight_quotes_list.append([])

    for idx, (it_text, quotes) in enumerate(zip(insight_texts, insight_quotes_list)):
        insight_id = gil_insight_node_id(episode_id, idx, it_text)
        nodes.append(
            {
                "id": insight_id,
                "type": "Insight",
                "properties": {
                    "text": it_text,
                    "episode_id": episode_id,
                    "grounded": len(quotes) > 0,
                },
            }
        )
        edges.append({"type": "HAS_INSIGHT", "from": ep_node_id, "to": insight_id})
        for gq in quotes:
            if not isinstance(gq, GroundedQuote):
                continue
            quote_id = gil_quote_node_id(
                episode_id,
                quote_global_idx,
                gq.text,
                gq.char_start,
                gq.char_end,
            )
            quote_global_idx += 1
            ts_start, ts_end = 0, 0
            sp_id: Optional[str] = None
            if use_segments and transcript_segments:
                ts_start, ts_end = _char_range_to_ms(
                    transcript_text or "",
                    gq.char_start,
                    gq.char_end,
                    transcript_segments,
                )
                sp_id = _speaker_id_for_char_range(
                    transcript_text or "",
                    gq.char_start,
                    gq.char_end,
                    transcript_segments,
                )
            nodes.append(
                {
                    "id": quote_id,
                    "type": "Quote",
                    "properties": {
                        "text": gq.text,
                        "episode_id": episode_id,
                        "speaker_id": sp_id,
                        "char_start": gq.char_start,
                        "char_end": gq.char_end,
                        "timestamp_start_ms": ts_start,
                        "timestamp_end_ms": ts_end,
                        "transcript_ref": transcript_ref,
                    },
                }
            )
            edges.append({"type": "SUPPORTED_BY", "from": insight_id, "to": quote_id})
            _attach_speaker_for_quote(
                nodes,
                edges,
                episode_id,
                quote_id,
                sp_id,
                speakers_added,
            )

    return {
        "schema_version": "1.0",
        "model_version": model_version,
        "prompt_version": prompt_version,
        "episode_id": episode_id,
        "nodes": nodes,
        "edges": edges,
    }
