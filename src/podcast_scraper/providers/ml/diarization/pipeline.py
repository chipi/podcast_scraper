"""Apply diarization to Whisper transcription results."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .... import config
from .alignment import align_segments_to_speakers
from .cache import (
    diarization_cache_dir_for_output,
    diarization_cache_path,
    load_cached_diarization,
    save_diarization_cache,
)
from .factory import create_diarization_provider
from .roster import build_speaker_diagnostics, resolve_speaker_roster

logger = logging.getLogger(__name__)


def _voice_texts_from_aligned(aligned: List[Any]) -> Dict[str, str]:
    """``voice_id -> concatenated text of its own turns`` (for own-turn self-intro naming, #876)."""
    chunks: Dict[str, List[str]] = {}
    for segment, speaker_id in aligned:
        txt = str(segment.get("text", "") or "") if isinstance(segment, dict) else ""
        if txt:
            chunks.setdefault(speaker_id, []).append(txt)
    return {v: " ".join(c) for v, c in chunks.items()}


def _ad_intervals(segments: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Ad regions of the episode as ``(start_s, end_s)`` time intervals.

    The ad detector works in *character* space over the transcript, while the roster reasons in
    *time* over diarization turns — this bridges the two. Without it the roster gets no ads at
    all, so a pre-roll ad read is indistinguishable from the host's intro and the episode's
    opening voice (the #1169 host rule) resolves to the **ad narrator**; the sponsor voice also
    never trips the ``COMMERCIAL_AD_FRACTION`` demotion. Both were live on real, ad-laden feeds.

    Returns ``[]`` when no ads are detected, which restores the previous (ad-blind) behaviour.
    """
    from ....gi.ad_regions import excise_ad_regions

    spans: List[Tuple[int, int, Dict[str, Any]]] = []
    parts: List[str] = []
    cursor = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "") or "")
        parts.append(text)
        spans.append((cursor, cursor + len(text), seg))
        cursor += len(text) + 1  # the space " ".join puts between segments

    try:
        _, _, meta = excise_ad_regions(" ".join(parts))
    except Exception as exc:  # noqa: BLE001 — ad detection must never break diarization
        logger.warning("Ad-region detection failed; diarizing without ad intervals: %s", exc)
        return []

    intervals: List[Tuple[float, float]] = []
    for char_start, char_end in meta.excised_ranges:
        covered = [
            seg
            for start, end, seg in spans
            if not (end <= char_start or start >= char_end)  # overlaps the ad span
        ]
        if not covered:
            continue
        intervals.append(
            (float(covered[0].get("start") or 0.0), float(covered[-1].get("end") or 0.0))
        )
    return intervals


def _enriched_segments(aligned: List[Any], roster: Any) -> List[Dict[str, Any]]:
    """Attach the resolved ``speaker`` + id-bearing ``speaker_label`` to each aligned segment.

    ``speaker_label`` stays the raw/real label (the GI mints person ids and the screenplay
    offsets from it). ``voice_type`` is an additive display hint so a surface can render
    "Brief speaker" / "Advertisement" for a cameo/ad voice without changing that id.
    """
    out: List[Dict[str, Any]] = []
    for segment, speaker_id in aligned:
        enriched = dict(segment)
        enriched["speaker"] = speaker_id
        enriched["speaker_label"] = roster.label_for(speaker_id)
        role = roster.by_voice.get(speaker_id)
        if role is not None and not role.named:
            if role.voice_type != "person":
                enriched["voice_type"] = role.voice_type
            if role.role == "host":
                enriched["speaker_role"] = "host"  # an unnamed host renders as "Host", not SPEAKER
        out.append(enriched)
    return out


def _resolve_diarization_cache_dir(cfg: config.Config, cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir:
        return cache_dir
    return diarization_cache_dir_for_output(cfg.output_dir)


def _resolve_voices_via_llm(
    cfg: config.Config,
    *,
    stated_names: List[str],
    voice_texts: Dict[str, str],
    known_hosts: List[str],
    ordered_turns: List[Tuple[str, str]],
) -> Dict[str, str]:
    """ADR-110 — match the stated names to the voices, using each voice's own words.

    Returns ``{}`` for every profile without an LLM. `airgapped`, `local`, `dev` and
    `reprocess_dgx_no_llm` run `speaker_detector_provider: spacy`, keep the deterministic cue
    matcher, and nothing about them changes.

    This never fails the episode. A speaker we cannot name costs an unnamed voice; a speaker we name
    WRONGLY puts words in a real person's mouth, and those are not symmetric (#876).
    """
    if not stated_names or not voice_texts:
        return {}
    if not bool(getattr(cfg, "speaker_resolution_llm", True)):
        return {}

    try:
        from ....speaker_detectors.resolution import (
            completion_fn_for,
            resolve_voices_from_conversation,
        )
        from ....summarization.factory import create_summarization_provider

        provider = create_summarization_provider(cfg)
        provider.initialize()
        complete = completion_fn_for(provider)
        if complete is None:
            logger.debug(
                "speaker resolution: %s has no completion endpoint — the deterministic cue "
                "matcher stays in charge",
                type(provider).__name__,
            )
            return {}
        return resolve_voices_from_conversation(
            stated_names,
            voice_texts,
            complete,
            known_hosts=known_hosts,
            ordered_turns=ordered_turns,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "speaker resolution unavailable (%s: %s); falling back to the deterministic cues",
            type(exc).__name__,
            exc,
        )
        return {}


def apply_diarization_to_result(
    result: dict,
    audio_path: str,
    cfg: config.Config,
    detected_speaker_names: Optional[List[str]],
    *,
    metadata_named: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """Enrich transcription segments with diarized speaker labels.

    ``metadata_named`` is every name the episode metadata stated, *before* corroboration filtered
    it. It never names a voice — it only lets the roster tell our own failures apart from the
    voices nobody could have named.
    """
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return result

    resolved_cache_dir = _resolve_diarization_cache_dir(cfg, cache_dir)
    diarization = None
    if resolved_cache_dir:
        cache_path = diarization_cache_path(audio_path, cfg, resolved_cache_dir)
        diarization = load_cached_diarization(cache_path)
        if diarization is not None:
            logger.info("Diarization cache hit: %s", os.path.basename(cache_path))

    if diarization is None:
        provider = create_diarization_provider(cfg)
        diarization = provider.diarize(
            audio_path,
            num_speakers=cfg.diarization_num_speakers,
            min_speakers=cfg.diarization_min_speakers,
            max_speakers=cfg.diarization_max_speakers,
        )
        if resolved_cache_dir:
            save_diarization_cache(
                diarization_cache_path(audio_path, cfg, resolved_cache_dir),
                diarization,
            )

    if not diarization.segments:
        # No speaker turns (silent/music-only audio, or a pyannote no-op). Returning
        # the result unchanged leaves segments without speaker_label, so the caller's
        # has_diarized_labels gate degrades to gap-based formatting instead of
        # attributing the whole episode to a phantom SPEAKER_00.
        logger.warning(
            "Diarization produced no speaker turns for %s; "
            "skipping speaker labels (gap-based formatting will be used).",
            os.path.basename(audio_path),
        )
        return result

    # Resolve every diarized voice once via the unified roster (#876): host = the opening
    # voice (#1169), named by transcript self-intro ("I'm Patrick O'Shaughnessy") → config
    # known_hosts; guests by talk-time; leftovers kept raw; a guest's name never lands on a
    # host. For network-published feeds the host name isn't in the metadata (the author tag is
    # the network), so the transcript self-intro the roster reads is the only reliable source.
    transcript_text = result.get("text") or " ".join(
        str(seg.get("text", "")) for seg in segments if isinstance(seg, dict)
    )
    # Align first so the roster can name a voice from its *own* turns' self-introduction (#876),
    # not only the episode-opening host intro.
    aligned = align_segments_to_speakers(segments, diarization)
    voice_texts = _voice_texts_from_aligned(aligned)
    guests = detected_speaker_names or []
    known_hosts = list(getattr(cfg, "known_hosts", None) or [])
    # Ordered turns let the roster use the host's introduction ("and now, Bobby Allen") to name the
    # voice that speaks NEXT — the only per-voice way to use an introduction.
    ordered_turns = [
        (str(speaker_id), str((seg or {}).get("text", ""))) for seg, speaker_id in aligned
    ]

    # ADR-110 — NOW we can hear them, so now we ask who they are.
    #
    # `detect_speakers` runs before the audio is even downloaded and its interface cannot take a
    # transcript, so it answers from show notes and returns the people the episode is ABOUT as
    # readily as the people in the room (#876: Elon Musk, named only as the man SUING OpenAI).
    # Here the voices exist. The model is shown each voice's own words plus the retrieved passages
    # where each stated name is actually spoken, and it may only MATCH a name from that closed list
    # — never author one. A voice it cannot place stays unnamed.
    # The HOSTS are candidates too. `detect_speakers` hands hosts back on a separate channel, so a
    # naive candidate list is guests-only — and then the voice holding 75% of a interview show has
    # no name it is allowed to be matched to.
    candidates = list(dict.fromkeys([*(metadata_named or ()), *guests, *known_hosts]))
    llm_voice_names = _resolve_voices_via_llm(
        cfg,
        stated_names=candidates,
        voice_texts=voice_texts,
        known_hosts=known_hosts,
        ordered_turns=ordered_turns,
    )

    roster = resolve_speaker_roster(
        diarization,
        transcript_text,
        detected_guests=guests,
        known_hosts=known_hosts,
        voice_texts=voice_texts,
        ordered_turns=ordered_turns,
        ad_intervals=_ad_intervals(segments),
        metadata_named=list(metadata_named or ()),
        llm_voice_names=llm_voice_names,
    )

    enriched_result = dict(result)
    enriched_result["segments"] = _enriched_segments(aligned, roster)
    # Diagnostics sidecar (what we tried / resolved / why each voice failed) — the caller
    # persists it next to the episode so unrecognized speakers are explainable without a re-run.
    enriched_result["speaker_diagnostics"] = build_speaker_diagnostics(
        diarization,
        roster,
        transcript_text=transcript_text,
        voice_texts=voice_texts,
        detected_guests=guests,
        known_hosts=known_hosts,
        metadata_named=list(metadata_named or ()),
        show_centric=bool(getattr(cfg, "show_centric", False)),
    )
    enriched_result["diarization_num_speakers"] = roster.num_speakers
    return enriched_result
