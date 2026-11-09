"""The episode's grounded insights (lifted from ``workflow.metadata_generation``, E04 S2)."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from podcast_scraper import config_constants
from podcast_scraper.exceptions import GILGroundingUnsatisfiedError
from podcast_scraper.speaker_stage.record import _self_introduced_names
from podcast_scraper.workflow.metadata_generation import (
    _capture_stage_exception,
    _determine_gi_path,
    _normalize_whitespace,
    GroundedInsightsMetadata,
)

logger = logging.getLogger(__name__)


def build_episode_insights(  # noqa: C901 - the block as generate_episode_metadata held it
    *,
    _blocked: Any,
    bridge_gi_payload: Any,
    cfg: Any,
    episode: Any,
    episode_duration_seconds: Any,
    episode_id: Any,
    episode_published_date: Any,
    feed: Any,
    feed_id: Any,
    feed_url: Any,
    gi_artifact_path: Any,
    gi_cost: Any,
    gi_elapsed: Any,
    gi_meta: Any,
    metadata_path: Any,
    output_dir: Any,
    pipeline_metrics: Any,
    summary_metadata: Any,
    summary_provider: Any,
    transcript_file_path: Any,
) -> Tuple[Any, ...]:
    """Build ``gi.json`` from the transcript and summary, write it, and return what the metadata
    record and the later GI + KG passes read. Lifted verbatim from ``generate_episode_metadata``;
    returns ``(bridge_gi_payload, gi_artifact_path, gi_cost, gi_elapsed, gi_meta, gi_path)``."""
    gi_path: Any = None
    # `not _blocked` is the GI half of RFC-124 §5.3's gate. GI is where the cost of getting this
    # wrong is highest: it mints claims and attaches them to people via SPOKEN_BY, so a claim
    # extracted from untranslated Spanish by English prompts becomes a durable corpus fact
    # nobody can later distinguish from a correct one.
    #
    # KG IS GATED SEPARATELY, at its own block below. This comment used to claim "KG runs inside
    # this block, so gating here gates both" — it does not, and asserting it instead of checking
    # left KG ungated behind a comment that said otherwise.
    if getattr(cfg, "generate_gi", False) and not _blocked:
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        episode_id_for_status, _ = get_episode_id_from_episode(episode, feed_url)
        gi_path = _determine_gi_path(metadata_path)
        transcript_text = ""
        transcript_segments_arg: Optional[List[Dict[str, Any]]] = None
        # #974: prefer the ad-free processing base (.adfree.txt + offset segments).
        # char_start computed below lives in THAT saved space, so GI, enrich-edges,
        # the search index, and the viewer all share one coordinate system. transcript_ref_for_gi
        # is the relpath actually read — quote/viewer references point at it.
        transcript_ref_for_gi = transcript_file_path or "transcript.txt"
        if transcript_file_path and output_dir:
            from podcast_scraper.workflow.transcript_resolution import load_processing_transcript

            loaded = load_processing_transcript(output_dir, transcript_file_path)
            transcript_text = loaded.text
            transcript_segments_arg = loaded.segments
            if loaded.is_adfree:
                transcript_ref_for_gi = loaded.transcript_ref
                # Ads were excised at save time; record run-level counters from the
                # ad-map so #656 Stage D totals stay accurate without re-excising.
                am = loaded.ad_map or {}
                if pipeline_metrics is not None and am.get("chars_removed"):
                    pre = am.get("preroll_cut_end") or 0
                    pr_start = am.get("postroll_cut_start")
                    post = (am.get("source_length", 0) - pr_start) if pr_start is not None else 0
                    pipeline_metrics.record_ad_region_excision(pre, post)
            elif transcript_text:
                # Legacy fallback (no ad-free base on disk): excise in-memory like #663.
                # Staged-mode GI re-reads the raw transcript (not the summarization-cleaned
                # one); without this, sponsor pre-rolls leak into insights and quotes.
                from podcast_scraper import stage_hooks

                transcript_text, transcript_segments_arg, ad_meta = stage_hooks.excise_ads(
                    transcript_text, segments=transcript_segments_arg
                )
                if pipeline_metrics is not None and ad_meta.chars_removed:
                    pre = ad_meta.preroll_cut_end or 0
                    post = (
                        (ad_meta.source_length - ad_meta.postroll_cut_start)
                        if ad_meta.postroll_cut_start is not None
                        else 0
                    )
                    pipeline_metrics.record_ad_region_excision(pre, post)
        publish_date_str = episode_published_date.isoformat() if episode_published_date else None
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                gi_start = time.time()
                from podcast_scraper.gi import build_artifact, write_artifact
                from podcast_scraper.gi.provenance import resolve_gil_artifact_model_version

                # The provider IS the source. ``gi_insight_source`` used to gate this and
                # defaulted to the placeholder, so a config that never reached the field produced
                # placeholder insights for the whole run (#1657). The gate is gone: if there is
                # a provider it is used, and if there is not, the episode honestly has no
                # insights.
                insight_texts_arg: Optional[List[str]] = None
                insight_provider_arg = summary_provider

                gil_evidence_cleanup: list = []
                _gil_lineage_provider = (
                    summary_provider if summary_provider is not None else insight_provider_arg
                )
                max_gi_topics = int(
                    getattr(
                        cfg,
                        "kg_max_topics",
                        config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX,
                    )
                    or config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
                )
                # #653 Part A: prefer KG canonical noun-phrase topics over summary
                # bullets for GI Topic labels. Bundled pipelines (mega/extraction)
                # surface them via summary_metadata.prefilled_extraction["topics"];
                # staged mode still falls back to bullets (see #653 Part D follow-up).
                gi_topic_labels: Optional[List[str]] = None
                if summary_metadata is not None:
                    _pe = getattr(summary_metadata, "prefilled_extraction", None)
                    if isinstance(_pe, dict):
                        _pe_topics = _pe.get("topics")
                        if isinstance(_pe_topics, list):
                            gi_topic_labels = []
                            for _t in _pe_topics[:max_gi_topics]:
                                _ts = str(_t).strip()
                                if _ts:
                                    gi_topic_labels.append(_ts)
                            if not gi_topic_labels:
                                gi_topic_labels = None
                # ADR-156 / #2164: the staged-mode bullets fallback (#653 Part D) is GONE. It
                # shortened each bullet to a 4-token phrase via ``_bullet_to_topic_phrase`` and
                # called the result a topic. The first four words of a summary bullet are not the
                # episode's subject, and such a label cannot recur across episodes, so it can never
                # cluster — structurally incapable of becoming a storyline, which is the only reason
                # Topics exist. ``gi_topic_labels`` now comes solely from the extractor's own topics
                # above, and otherwise stays None; GI receives the canonical set by alignment with
                # the KG (``gi.topic_alignment``).
                gi_episode_duration_ms: Optional[int] = None
                if episode_duration_seconds is not None and int(episode_duration_seconds) > 0:
                    gi_episode_duration_ms = int(episode_duration_seconds) * 1000
                # #643: prefer insights from mega_bundled / extraction_bundled.
                prefilled_insights_arg: Optional[List[Dict[str, Any]]] = None
                if summary_metadata is not None:
                    pe = getattr(summary_metadata, "prefilled_extraction", None)
                    if isinstance(pe, dict):
                        pe_insights = pe.get("insights")
                        if isinstance(pe_insights, list) and pe_insights:
                            prefilled_insights_arg = pe_insights
                # RFC-109: wrap pipeline_metrics so THIS episode's GI cost is captured in isolation
                # (the run-level accumulator is shared across parallel episodes).
                from podcast_scraper.workflow.processing_manifest import EpisodeCostProbe

                _gi_probe = (
                    EpisodeCostProbe(pipeline_metrics) if pipeline_metrics is not None else None
                )
                payload = build_artifact(
                    episode_id,
                    transcript_text,
                    model_version=resolve_gil_artifact_model_version(cfg, _gil_lineage_provider),
                    prompt_version="v1",
                    podcast_id=feed_id,
                    episode_title=episode.title,
                    publish_date=publish_date_str,
                    transcript_ref=transcript_ref_for_gi,
                    transcript_segments=transcript_segments_arg,
                    cfg=cfg,
                    insight_texts=insight_texts_arg,
                    insight_provider=insight_provider_arg,
                    summary_provider=summary_provider,
                    pipeline_metrics=(_gi_probe if _gi_probe is not None else pipeline_metrics),
                    gil_created_evidence_providers=gil_evidence_cleanup,
                    topic_labels=gi_topic_labels,
                    episode_duration_ms=gi_episode_duration_ms,
                    prefilled_insights=prefilled_insights_arg,
                    feed_id=feed_id,
                    # #2065: GI mints its speakers from the segments sidecar, NOT through
                    # `_speaker_lists_for_graph`, so the show-name refusal that protects kg.json
                    # never saw this path. Without the title that guard is inert by design
                    # (`names_the_show` has no opinion on an empty title), which is exactly how
                    # 53 SPOKEN_BY edges came to point at a Person named `Machine Learning
                    # Street`. Pass it or the guard does nothing.
                    feed_title=getattr(feed, "title", None),
                    self_introduced_names=_self_introduced_names(output_dir, transcript_file_path),
                )
                if _gi_probe is not None:
                    gi_cost = _gi_probe.gi_cost_usd
                write_artifact(Path(gi_path), payload, validate=True)
                bridge_gi_payload = payload
                gi_artifact_path = gi_path
                gi_elapsed = time.time() - gi_start
                logger.debug("[%s] Generated GIL artifact: %s", episode.idx, gi_path)
                # Index for metadata model: provenance only (full graph stays in gi.json)
                nodes = payload.get("nodes", [])
                edges = payload.get("edges", [])
                insight_count = sum(1 for n in nodes if n.get("type") == "Insight")
                quote_count = sum(1 for n in nodes if n.get("type") == "Quote")
                grounded_insight_ids = {
                    e["from"] for e in edges if e.get("type") == "SUPPORTED_BY" and e.get("from")
                }
                grounded_count = len(grounded_insight_ids)
                has_insights_and_quotes = insight_count >= 1 and quote_count >= 1
                quotes_verbatim = 0
                quotes_checked = 0
                if transcript_text and quote_count > 0:
                    for n in nodes:
                        if n.get("type") != "Quote":
                            continue
                        props = n.get("properties") or {}
                        try:
                            start = int(props.get("char_start", 0))
                            end = int(props.get("char_end", 0))
                            text = (props.get("text") or "").strip()
                        except (TypeError, ValueError):
                            continue
                        if end <= start or start < 0 or end > len(transcript_text):
                            continue
                        quotes_checked += 1
                        try:
                            slice_text = transcript_text[start:end].strip()
                            if _normalize_whitespace(text) == _normalize_whitespace(slice_text):
                                quotes_verbatim += 1
                        except Exception:
                            pass
                gi_meta = GroundedInsightsMetadata(
                    artifact_path=os.path.relpath(gi_path, output_dir),
                    insight_count=insight_count,
                    generated_at=datetime.now(),
                    schema_version=payload.get("schema_version", "1.0"),
                )
                if pipeline_metrics is not None:
                    pipeline_metrics.gi_artifacts_generated += 1
                    pipeline_metrics.record_gi_time(gi_elapsed)
                    pipeline_metrics.record_gi_success_counts(
                        insights=insight_count,
                        quotes=quote_count,
                        grounded_insights=grounded_count,
                        has_insights_and_quotes=has_insights_and_quotes,
                        quotes_verbatim=quotes_verbatim,
                        quotes_checked=quotes_checked,
                    )
                    pipeline_metrics.update_episode_status(
                        episode_id=episode_id_for_status, stage="gi_written"
                    )
                    pipeline_metrics.update_episode_metrics(
                        episode_id=episode_id_for_status, gi_sec=gi_elapsed
                    )
                # Clean up GIL evidence providers instantiated inside build_artifact
                for idx, prov in enumerate(gil_evidence_cleanup):
                    if prov is None or not hasattr(prov, "cleanup"):
                        continue
                    try:
                        prov.cleanup()
                    except Exception as cleanup_exc:
                        logger.debug(
                            "[%s] GIL evidence provider cleanup failed (%s): %s",
                            episode.idx,
                            idx,
                            cleanup_exc,
                        )
                break
            except GILGroundingUnsatisfiedError:
                raise
            except Exception as gi_exc:
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
                    continue
                if pipeline_metrics is not None:
                    pipeline_metrics.gi_failures += 1
                    pipeline_metrics.update_episode_status(
                        episode_id=episode_id_for_status, stage="gi_failed"
                    )
                logger.warning(
                    "[%s] GIL artifact generation failed (non-fatal): %s",
                    episode.idx,
                    gi_exc,
                    exc_info=True,
                )
                _capture_stage_exception(gi_exc, stage="gi")
    return bridge_gi_payload, gi_artifact_path, gi_cost, gi_elapsed, gi_meta, gi_path
