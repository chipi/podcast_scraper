"""The speaker stage after ASR: diarization, naming and the speech-coverage failover on a fresh
transcript (moved from ``workflow.episode_processor``, E04 S2).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from podcast_scraper import config
from podcast_scraper.models import TranscriptionJob
from podcast_scraper.speaker_stage.native import _apply_native_speaker_roster
from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.workflow.episode_processor import (
    _capture_stage_exception,
    _detection_report_of,
    _gap_clip_transcriber,
    _operator_renames,
    _split_diarize_naming,
    _transcribe_with_segments_maybe_chunked,
)

logger = logging.getLogger(__name__)


def _maybe_speech_coverage_failover(
    result: Dict[str, Any],
    media_for_transcription: str,
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    effective_output_dir: str,
    pipeline_metrics: Any,
    episode_duration_seconds: Optional[float],
) -> Dict[str, Any]:
    """ADR-131: re-transcribe on the failover model when the diarized transcript covers too little
    of the diarizer's SPEECH.

    ``speech_coverage = Σ(transcript segments) / Σ(diarization speech)`` (both merged). Unlike the
    raw ADR-123 gate (÷ total audio), music/ads/silence — which produce no diarization turn — are
    excluded, so a music-heavy episode turbo transcribed fully does not falsely failover, while the
    long-episode cliff (turbo dropping real speech a speaker was talking through) still does.

    Provider-agnostic (any ``DiarizationResult``). A **no-op** when there is no speech denominator
    (diarization off, or no speaker turns): the raw ``transcription_coverage_min`` gate governs that
    case instead, so nothing regresses. The failover result carries a ``speech_coverage_failover``
    breadcrumb + ``model_used`` for per-episode provenance.
    """
    min_cov = float(getattr(cfg, "transcription_speech_coverage_min", 0.0) or 0.0)
    fo_model = getattr(cfg, "transcription_coverage_failover_model", None)
    speech = float(result.get("diarization_speech_seconds") or 0.0)
    if speech <= 0:
        return result  # no speech denominator — defer to the raw-coverage gate

    from podcast_scraper.providers.ml.diarization.pipeline import (
        apply_diarization_to_result,
        merged_speech_seconds,
    )

    # The METRIC is recorded whether or not the gate is armed (#2187). It used to be computed
    # only past the gate's early return, so a profile that turns the gate off (prod_dgx_full:
    # ASR must stay constant) recorded `speech_coverage: null` while its registry comment said
    # the metric was still recorded.
    covered = merged_speech_seconds(result.get("segments") or [])
    speech_cov = min(1.0, covered / speech)
    result["asr_speech_coverage"] = round(speech_cov, 3)
    if min_cov <= 0 or not fo_model:
        return result
    primary_model = getattr(cfg, "dgx_whisper_model", None)
    if speech_cov >= min_cov:
        # Observable pass (ADR-131): log every evaluation so a run shows the gate ran + its
        # coverage, not only the rare failover ("the gate works, it just didn't need to fire").
        logger.info(
            "[%s] speech coverage %.1f%% >= %.1f%% — keeping primary %r (ADR-131 gate passed)",
            job.idx,
            speech_cov * 100,
            min_cov * 100,
            primary_model,
        )
        return result

    logger.info(
        "[%s] speech coverage %.1f%% < %.1f%% — primary %r dropped real speech; "
        "re-transcribing on failover model %s (ADR-131)",
        job.idx,
        speech_cov * 100,
        min_cov * 100,
        primary_model,
        fo_model,
    )
    from podcast_scraper.transcription.factory import create_transcription_provider
    from podcast_scraper.utils.provider_metrics import (
        apply_estimated_cost_if_missing,
        ProviderCallMetrics,
        transcription_model_for_cfg,
    )

    # Route the failover to the configured provider — parity with the ADR-123 raw gate's factory
    # logic (#1273): 'moss' re-transcribes on the DGX MOSS service (:8004) with moss_model, not the
    # speaches whisper service, so the MOSS model id is never sent to a server that cannot serve it.
    # None (default) keeps the historical whisper-on-whisper swap via dgx_whisper_model. Both gates
    # are disabled on the failover pass so it cannot recurse into another re-transcription.
    cov_provider = getattr(cfg, "transcription_coverage_failover_provider", None)
    model_field = "moss_model" if cov_provider == "moss" else "dgx_whisper_model"
    fo_updates: Dict[str, Any] = {
        model_field: fo_model,
        "transcription_coverage_min": 0.0,
        "transcription_speech_coverage_min": 0.0,
    }
    if cov_provider:
        fo_updates["transcription_provider"] = cov_provider
    fo_cfg = cfg.model_copy(update=fo_updates)
    fo_provider = create_transcription_provider(fo_cfg)
    fo_call_metrics = ProviderCallMetrics()
    fo_result, _ = _transcribe_with_segments_maybe_chunked(
        media_for_transcription,
        cfg=fo_cfg,
        job=job,
        transcription_provider=fo_provider,
        pipeline_metrics=pipeline_metrics,
        episode_duration_seconds=episode_duration_seconds,
        call_metrics=fo_call_metrics,
    )
    # RFC-109: the failover re-transcription is a SECOND ASR call. Local models cost 0; a cloud
    # failover model bills again, so its cost must be added to the ASR block (not just the primary).
    apply_estimated_cost_if_missing(
        fo_call_metrics,
        cfg=fo_cfg,
        provider_type=str(getattr(fo_cfg, "transcription_provider", None) or "whisper"),
        capability="transcription",
        model=transcription_model_for_cfg(fo_cfg),
        audio_minutes=(episode_duration_seconds / 60.0) if episode_duration_seconds else None,
    )
    fo_result["asr_failover_cost_usd"] = fo_call_metrics.estimated_cost
    # Diarization is audio-based (cache hit on the same audio) — only the transcript→speaker
    # alignment re-runs, so the failover pays for re-transcription, not re-diarization.
    fo_result = apply_diarization_to_result(
        fo_result,
        media_for_transcription,
        fo_cfg,
        job.detected_speaker_names,
        metadata_named=job.metadata_named,
        cache_dir=os.path.join(effective_output_dir, ".cache", "diarization"),
        feed_hosts=job.feed_hosts,
        unbindable_hosts=getattr(getattr(job, "episode", None), "unbindable_hosts", None),
        detection_report=_detection_report_of(job),
        speaker_renames=_operator_renames(job),
    )
    fo_speech = float(fo_result.get("diarization_speech_seconds") or speech)
    fo_cov = (
        min(1.0, merged_speech_seconds(fo_result.get("segments") or []) / fo_speech)
        if fo_speech > 0
        else None
    )
    fo_result["model_used"] = fo_result.get("model_used") or f"{fo_model}:speech_coverage_failover"
    if fo_cov is not None:
        fo_result["asr_speech_coverage"] = round(fo_cov, 3)
    fo_result["speech_coverage_failover"] = {
        "primary_model": primary_model,
        "primary_speech_coverage": round(speech_cov, 3),
        "failover_speech_coverage": round(fo_cov, 3) if fo_cov is not None else None,
        "speech_coverage_min": min_cov,
    }
    return fo_result


def speakers_after_asr(
    result: Any,
    media_for_transcription: Any,
    cfg: Any,
    job: Any,
    effective_output_dir: str,
    pipeline_metrics: Any,
    episode_duration_seconds: Any,
    transcription_provider: Any,
) -> Tuple[Any, Optional[float], Optional[float], Optional[str]]:
    """The speaker stage after ASR: diarize and name the voices (with the speech-coverage
    failover), or put a provider's own speaker labels through the roster. Returns the result
    and the diarization timings and error the manifest records."""
    dz_elapsed: Optional[float] = None
    dz_naming_elapsed: Optional[float] = None
    dz_error: Optional[str] = None
    if cfg.diarize:
        from podcast_scraper.exceptions import ProviderDependencyError
        from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

        dz_t0 = time.monotonic()
        dz_timings: Dict[str, float] = {}
        try:
            result = apply_diarization_to_result(
                result,
                media_for_transcription,
                cfg,
                job.detected_speaker_names,
                metadata_named=job.metadata_named,
                cache_dir=os.path.join(effective_output_dir, ".cache", "diarization"),
                feed_hosts=job.feed_hosts,
                unbindable_hosts=getattr(getattr(job, "episode", None), "unbindable_hosts", None),
                # ADR-137 — title + description feed the LLM's host/guest role determination.
                episode_title=job.ep_title,
                episode_description=getattr(job.episode, "description", None),
                detection_report=_detection_report_of(job),
                speaker_renames=_operator_renames(job),
                transcribe_clip=(
                    _gap_clip_transcriber(cfg, transcription_provider)
                    if cfg.transcription_recover_untranscribed_speech
                    else None
                ),
                recover_silent_gaps=cfg.transcription_recover_silent_gaps,
                timings=dz_timings,
            )
        except (ProviderDependencyError, ValueError, OSError, RuntimeError) as exc:
            # Broadened catch (Whisper-e2e diagnosis, #1180 follow-up).
            # A diarization failure MUST NOT lose the successfully-computed
            # transcript. HuggingFace Hub errors from an uncached pyannote
            # model raise OSError; torch/pyannote instantiation issues raise
            # RuntimeError; both were previously escaping to the outer
            # transcription except clause and getting mislabeled as
            # "Whisper transcription failed" while dropping the transcript.
            logger.warning(
                "[%s] Diarization failed; falling back to gap-based screenplay: %s",
                job.idx,
                format_exception_for_log(exc),
            )
            _capture_stage_exception(exc, stage="diarization")
            dz_error = type(exc).__name__
        dz_elapsed, dz_naming_elapsed = _split_diarize_naming(time.monotonic() - dz_t0, dz_timings)
        # ADR-131: speech-normalized quality gate — re-transcribe on the failover model if the
        # diarized transcript covers too little of the diarizer's SPEECH (music/ads excluded).
        # No-op unless configured + diarization produced a speech denominator.
        result = _maybe_speech_coverage_failover(
            result,
            media_for_transcription,
            cfg,
            job,
            effective_output_dir,
            pipeline_metrics,
            episode_duration_seconds,
        )
    else:
        # Native-screenplay provider (deepgram/moss) may have self-diarized without a roster
        # pass — route it through the single role authority (no-op otherwise). Guard + failure
        # handling live in the helper so this stays one branch.
        result = _apply_native_speaker_roster(result, cfg, job)
    return result, dz_elapsed, dz_naming_elapsed, dz_error
