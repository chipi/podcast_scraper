"""Apply diarization to Whisper transcription results."""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import threading
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Tuple

from .... import config
from .alignment import align_segments_to_speakers
from .base import DiarizationResult
from .cache import (
    diarization_cache_dir_for_output,
    diarization_cache_path,
    load_cached_diarization,
    save_diarization_cache,
)
from .factory import create_diarization_provider
from .labeling_profile import DEFAULT_LABELING_PROFILE, get_profile
from .roster import (
    build_speaker_diagnostics,
    classify_voices,
    resolve_speaker_roster,
    SpeakerRoster,
)

logger = logging.getLogger(__name__)


def _voice_texts_from_aligned(aligned: List[Any]) -> Dict[str, str]:
    """``voice_id -> concatenated text of its own turns`` (for own-turn self-intro naming, #876)."""
    chunks: Dict[str, List[str]] = {}
    for segment, speaker_id in aligned:
        txt = str(segment.get("text", "") or "") if isinstance(segment, dict) else ""
        if txt:
            chunks.setdefault(speaker_id, []).append(txt)
    return {v: " ".join(c) for v, c in chunks.items()}


def _strip_ad_segments(
    aligned: List[Any], ad_intervals: Sequence[Tuple[float, float]]
) -> List[Any]:
    """Drop ``(segment, voice)`` pairs whose time falls inside a known ad interval.

    The ad regions are already computed (``_ad_intervals``); this reuses them so the text handed to
    the LLM speaker-resolver is a voice's REAL speech, not a sponsor read. Without it a voice whose
    diarized cluster contains the pre-roll ad is shown to the model reading "Ramp is the only
    platform…", which mismaps it (John Kim, prod-v2.4-100ep). A segment is an ad when its MIDPOINT
    lies in an ad interval. Empty ``ad_intervals`` → unchanged (ad-blind, previous behaviour).
    """
    if not ad_intervals:
        return aligned

    def _in_ad(segment: Any) -> bool:
        if not isinstance(segment, dict):
            return False
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return False
        mid = (float(start) + float(end)) / 2.0
        return any(a <= mid <= b for a, b in ad_intervals)

    return [(seg, spk) for seg, spk in aligned if not _in_ad(seg)]


def merged_speech_seconds(segments: Sequence[Any]) -> float:
    """ADR-131: total speech duration as the union of diarization turns (overlaps merged once).

    Works on any provider's ``DiarizationSegment`` list (start/end attrs) or on dict segments with
    ``start``/``end`` keys, so it is diarizer-agnostic. Merging overlaps matters: co-hosts talking
    over each other must not double-count. Returns 0.0 for an empty/degenerate diarization — the
    caller reads that as "no speech denominator" and defers to the raw-coverage gate.
    """

    def _bounds(s: Any) -> Optional[Tuple[float, float]]:
        try:
            if isinstance(s, dict):
                return float(s["start"]), float(s["end"])
            return float(s.start), float(s.end)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    ivs = sorted(b for b in (_bounds(s) for s in segments) if b is not None and b[1] > b[0])
    total = 0.0
    cur_start: Optional[float] = None
    cur_end = 0.0
    for start, end in ivs:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def _segment_end(s: Any) -> Optional[float]:
    try:
        return float(s["end"]) if isinstance(s, dict) else float(s.end)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _estimate_diarization_cost(
    diarization: DiarizationResult, cfg: Any, audio_seconds: Optional[float] = None
) -> Optional[float]:
    """Per-episode diarization cost in USD for the processing manifest (RFC-109 / ADR-132).

    Provider-agnostic: if the provider already set ``DiarizationResult.cost_usd`` it is trusted;
    otherwise the shared pricing layer (``capability="diarization"``) estimates cost per audio-min,
    as ASR does. Cloud diarizers (Deepgram/Gemini) have a pricing entry and get a real figure; local
    diarizers (pyannote/DGX/MOSS) have none, so the estimate stays ``None`` — a truthful "no billed
    cost", not a fabricated zero.

    ``audio_seconds`` is the billed unit (cloud diarizers bill on total audio). The caller passes
    the widest end across the ASR transcript + diarization turns — a closer proxy than diarization
    turns alone (which stop at the last speaker turn, undercounting trailing non-speech).
    """
    provider_set = getattr(diarization, "cost_usd", None)
    if provider_set is not None:
        return float(provider_set)
    provider_type = getattr(cfg, "diarization_provider", None)
    if not provider_type:
        return None

    span = audio_seconds
    if span is None:
        ends = [e for e in (_segment_end(s) for s in (diarization.segments or [])) if e is not None]
        span = max(ends) if ends else None
    if not span or span <= 0:
        return None
    from podcast_scraper.utils.provider_metrics import (
        apply_estimated_cost_if_missing,
        ProviderCallMetrics,
    )

    cm = ProviderCallMetrics()
    apply_estimated_cost_if_missing(
        cm,
        cfg=cfg,
        provider_type=str(provider_type),
        capability="diarization",
        model=getattr(diarization, "model_name", "") or "",
        audio_minutes=span / 60.0,
    )
    return cm.estimated_cost


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
        if role is not None:
            if not role.named:
                if role.voice_type != "person":
                    enriched["voice_type"] = role.voice_type
                if role.role == "host":
                    enriched["speaker_role"] = "host"  # an unnamed host renders as "Host"
            elif role.role in ("host", "guest"):
                # Invariant: a NAMED voice is always host or guest today (roster.py only builds
                # named SpeakerRoles with those roles; "unknown" is always unnamed). If that ever
                # changes, the metadata reader silently falls back to the detected_guests heuristic.
                # Persist the roster's host/guest role for NAMED voices too, so the durable segments
                # sidecar carries the role truth that metadata / context-digest read downstream.
                # Without this the reader had to guess host-vs-guest from the pre-diarization
                # detected_guests hint, and defaulted every named voice to "host" — turning guests
                # (Brundage, Karpathy, ...) into hosts across ~half the corpus.
                enriched["speaker_role"] = role.role
        out.append(enriched)
    return out


def _resolve_diarization_cache_dir(cfg: config.Config, cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir:
        return cache_dir
    return diarization_cache_dir_for_output(cfg.output_dir)


# Cache maps output_dir -> (transcript_count_at_build, shingles). The count lets a batch's LATER
# episodes rebuild the index once earlier episodes have written their transcripts (D3): keying on
# out_dir alone froze the index at the first episode's state — on a fresh feed's first full pass
# that is near-empty, so the mid-roll-ad rule ran blind for the whole batch.
_recurring_cache: Dict[str, tuple[int, set]] = {}
_recurring_lock = threading.Lock()


def _feed_recurring_text(cfg: config.Config) -> set:
    """The script THIS FEED repeats across its own episodes (#1188).

    A mid-roll house ad is short but sits in the middle, so the edge rule cannot see it, and it
    carries no sponsor language, so the keyword patterns score zero. It gives itself away only
    across episodes: it is read from the same script every week. `Jonathan Knight` (NYT Games) got
    into 7 of 10 Hard Fork episodes as a named person because no single episode could tell.

    Built from the transcripts already on disk for this output dir, and REBUILT when more
    transcripts have since landed (so a batch's later episodes see the passages its earlier ones
    just wrote). Fewer than three transcripts and there is nothing to compare, so the rule abstains.
    """
    out_dir = str(getattr(cfg, "output_dir", "") or "")
    if not out_dir:
        return set()
    paths = [
        p
        for p in Path(out_dir).glob("**/transcripts/*.txt")
        if ".adfree" not in p.name and ".cleaned" not in p.name
    ]
    count = len(paths)
    cached = _recurring_cache.get(out_dir)
    if cached is not None and count <= cached[0]:
        return cached[1]  # no new transcripts since the last build
    with _recurring_lock:
        cached = _recurring_cache.get(out_dir)
        if cached is not None and count <= cached[0]:
            return cached[1]
        try:
            from ....speaker_detectors.boilerplate import shingles_from_transcript_files

            shingles = shingles_from_transcript_files(paths) if count >= 3 else set()
            if shingles:
                logger.info(
                    "#1188: indexed %d repeated passages across %d transcripts of this feed — "
                    "a voice that only reads them is a recording, not a person",
                    len(shingles),
                    count,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("recurring-text index unavailable (%s); mid-roll ad rule abstains", exc)
            shingles = set()
        _recurring_cache[out_dir] = (count, shingles)
        return shingles


def _resolution_attribution(baseline: Any, final: Any) -> Dict[str, Any]:
    """How much of the final naming/role came from the deterministic cues vs the LLM (ADR-137).

    ``baseline`` is the roster with the LLM inputs emptied (pure cues); ``final`` is the shipped
    roster. Counts on each side answer "before/after the LLM"; the per-voice diff is the LLM's
    marginal contribution — names it added to voices the cues left raw, and roles it changed.
    """

    def _counts(r: Any) -> Dict[str, int]:
        vs = list(r.by_voice.values())
        return {
            "named": sum(1 for v in vs if v.named),
            "hosts": sum(1 for v in vs if v.role == "host"),
            "guests": sum(1 for v in vs if v.role == "guest"),
        }

    names_added: List[Dict[str, str]] = []
    names_removed: List[Dict[str, str]] = []
    roles_changed: List[Dict[str, Optional[str]]] = []
    for vid, fin in final.by_voice.items():
        base = baseline.by_voice.get(vid)
        if base is None:
            continue
        if fin.named and not base.named:
            names_added.append({"voice": vid, "name": fin.name})
        # A name the cues established but the LLM path dropped — a regression, never intended
        # (ADR-137: the LLM is additive). Post-reconciliation this must be empty; tracked so a leak
        # is visible in the sidecar instead of silent (it used to be — only additions were counted).
        if base.named and not fin.named:
            names_removed.append({"voice": vid, "name": base.name})
        if fin.role != base.role:
            roles_changed.append({"voice": vid, "from": base.role, "to": fin.role})
    return {
        "deterministic": _counts(baseline),
        "final": _counts(final),
        "llm_delta": {
            "names_added": names_added,
            "names_removed": names_removed,
            "roles_changed": roles_changed,
        },
    }


def _reconcile_non_regression(baseline: Any, final: Any) -> Tuple[Any, List[str]]:
    """ADR-137 non-regression guard: the LLM path is contracted to be ADDITIVE — it may add names
    to voices the cues left raw and correct roles, but it must NEVER erase a name the deterministic
    cues already established. When applying the LLM inputs un-names such a voice (measured at
    18/104 episodes on prod-v2.4-100ep), restore that voice's full baseline resolution.

    Returns the (rebuilt when needed) roster and the list of restored voice ids. ``final`` is
    frozen, so a rebuilt roster is returned rather than mutated in place.
    """
    restored: List[str] = []
    merged = dict(final.by_voice)
    for vid, base_role in baseline.by_voice.items():
        fin_role = merged.get(vid)
        if base_role.named and (fin_role is None or not fin_role.named):
            merged[vid] = base_role
            restored.append(vid)
    if not restored:
        return final, []
    return dataclasses.replace(final, by_voice=merged), restored


def _labeled_intro_block(
    aligned: List[Tuple[dict, Any]],
    real_voices: AbstractSet[str],
    *,
    max_words: int = 500,
) -> str:
    """The first ~``max_words`` of the diarized transcript, speaker-labeled, restricted to REAL
    voices for the LLM role call (ADR-137).

    Ad / cameo / commercial voices are excluded via ``real_voices`` — the shared cleaning
    classification (:func:`roster.classify_voices`), the SAME source the roster uses, so the intro
    is cleaned once and never replicated. This is the role-bearing input: the intro is where a show
    says who hosts and who is visiting.
    """
    lines: List[str] = []
    words = 0
    for seg, spk in aligned:
        if str(spk) not in real_voices:
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not text:
            continue
        lines.append(f"{spk}: {text}")
        words += len(text.split())
        if words >= max_words:
            break
    return "\n".join(lines)


def _resolve_voices_via_llm(
    cfg: config.Config,
    *,
    stated_names: List[str],
    voice_texts: Dict[str, str],
    known_hosts: List[str],
    ordered_turns: List[Tuple[str, str]],
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    intro_block: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """ADR-110/ADR-137 — match stated names to voices AND decide host/guest, from the conversation.

    Returns ``({voice: name}, {voice: role})``. Empty for every profile without an LLM: `airgapped`,
    `local`, `dev` and `reprocess_dgx_no_llm` run `speaker_detector_provider: spacy`, keep the
    deterministic cue matcher, and nothing about them changes.

    This never fails the episode. A speaker we cannot name costs an unnamed voice; a speaker we name
    WRONGLY puts words in a real person's mouth, and those are not symmetric (#876).
    """
    # Role-only mode (ADR-137): run even with no candidate names, as long as there are voices and
    # some role context (title/description/intro) — a no-stated-host show still needs host/guest.
    if not voice_texts:
        return {}, {}
    if not stated_names and not (episode_title or episode_description or intro_block):
        return {}, {}
    if not bool(getattr(cfg, "speaker_resolution_llm", True)):
        return {}, {}

    try:
        from ....speaker_detectors.resolution import (
            completion_fn_for,
            resolve_voices_and_roles,
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
            return {}, {}
        resolved = resolve_voices_and_roles(
            stated_names,
            voice_texts,
            complete,
            known_hosts=known_hosts,
            ordered_turns=ordered_turns,
            episode_title=episode_title,
            episode_description=episode_description,
            intro_block=intro_block,
        )
        names = {v: lv.name for v, lv in resolved.items() if lv.name}
        roles = {v: lv.role for v, lv in resolved.items() if lv.role}
        return names, roles
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "speaker resolution unavailable (%s: %s); falling back to the deterministic cues",
            type(exc).__name__,
            exc,
        )
        return {}, {}


def apply_diarization_to_result(
    result: dict,
    audio_path: str,
    cfg: config.Config,
    detected_speaker_names: Optional[List[str]],
    *,
    metadata_named: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    precomputed_diarization: Optional[DiarizationResult] = None,
    feed_hosts: Optional[List[str]] = None,
    bypass_cache_read: bool = False,
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    detection_ran: Optional[bool] = None,
) -> dict:
    """Enrich transcription segments with diarized speaker labels.

    ``metadata_named`` is every name the episode metadata stated, *before* corroboration filtered
    it. It never names a voice — it only lets the roster tell our own failures apart from the
    voices nobody could have named.

    ``precomputed_diarization`` supplies the diarized voices directly, skipping the
    cache/provider (audio) path — used by ``pipeline_stage=relabel_only`` to re-resolve
    names on an existing corpus's frozen ``SPEAKER_NN`` diarization, no audio / re-diarize.

    ``feed_hosts`` are the host names the feed's own blurb states (via
    ``detect_hosts_from_feed``); merged with ``cfg.known_hosts`` to anchor the roster and
    canonicalize ASR-garbled host surnames.

    ``detection_ran`` reports whether the speaker-detection stage executed for this episode
    (#1647). It is passed straight to the diagnostics: an unnamed voice reads as "nobody names
    them" only if detection looked, and as an unmeasured gap if it did not. ``None`` = caller
    did not say.
    """
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return result

    resolved_cache_dir = _resolve_diarization_cache_dir(cfg, cache_dir)
    diarization = precomputed_diarization
    # rediarize_only (v2.2) passes bypass_cache_read=True so a re-diarize is genuinely fresh even
    # when the diarizer config is unchanged — otherwise the audio-hash cache would return the old
    # diarization and the re-diarize would no-op. The fresh result is still cached below.
    if diarization is None and resolved_cache_dir and not bypass_cache_read:
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
    # known_hosts anchors the roster: it names the opening voice and lets
    # _canonicalize_to_known_host repair an ASR-garbled spoken surname ("Kevin Russo" -> "Kevin
    # Roose"). cfg.known_hosts is the manual show-level override; feed_hosts is what the feed's own
    # blurb states ("journalists Kevin Roose and Casey Newton explore..."), detected via
    # detect_hosts_from_feed by the caller. Without this the roster ran with an empty anchor on
    # every feed that did not hard-code cfg.known_hosts.
    known_hosts = list(
        dict.fromkeys(list(getattr(cfg, "known_hosts", None) or []) + list(feed_hosts or []))
    )
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
    ad_intervals = _ad_intervals(segments)
    recurring_text = _feed_recurring_text(cfg)
    dz_provider = getattr(cfg, "diarization_provider", None)
    # Versioned labeling profile (ADR-140) — one knob-bundle drives cleaning + naming and is stamped
    # on the sidecar. Resolved HERE (before the cleaning pass) so both the cleaning and the roster
    # read the same knobs. A typo'd id is rejected fail-fast by the Config validator (F6), so this
    # fallback is defense-in-depth for a caller that bypassed Config construction.
    try:
        _labeling_profile = get_profile(getattr(cfg, "labeling_profile", None) or "naming-4")
    except ValueError:
        logger.warning(
            "unknown labeling_profile %r; using the production default",
            getattr(cfg, "labeling_profile", None),
        )
        _labeling_profile = DEFAULT_LABELING_PROFILE
    # ADR-137 — ONE deterministic cleaning pass, right after diarization, shared by the LLM call and
    # the roster so "which voices are ad / cameo / commercial vs real" is defined in a single place.
    cleaning = classify_voices(
        diarization,
        ad_intervals,
        voice_texts=voice_texts,
        ordered_turns=ordered_turns,
        recurring_text=recurring_text,
        diarization_provider=dz_provider,
        cameo_max_talk_s=_labeling_profile.cameo_max_talk_s,
    )
    # The intro (title + description + the cleaned, labeled first minutes) is where a show states
    # who hosts and who is visiting; it lets the same call decide host/guest, not just name. The LLM
    # is asked only about REAL voices — never a cameo/commercial/ad, on which it abstains anyway.
    # Feed the LLM a voice's REAL speech, not sponsor reads: the ad regions are already known
    # (ad_intervals), so strip those segments from the intro + per-voice samples the resolver sees.
    # The FULL voice_texts above still feed classify_voices (it needs the ad text to type ad
    # voices); only the LLM's input is ad-stripped. Fixes the John Kim mismap (ad-read SPEAKER_00).
    adfree_aligned = _strip_ad_segments(aligned, ad_intervals)
    intro_block = _labeled_intro_block(adfree_aligned, cleaning.real)
    real_voice_texts = {
        v: t for v, t in _voice_texts_from_aligned(adfree_aligned).items() if v in cleaning.real
    }
    llm_voice_names, llm_voice_roles = _resolve_voices_via_llm(
        cfg,
        stated_names=candidates,
        voice_texts=real_voice_texts,
        known_hosts=known_hosts,
        ordered_turns=ordered_turns,
        episode_title=episode_title,
        episode_description=episode_description,
        intro_block=intro_block,
    )

    _md_named = list(metadata_named or ())

    def _run_roster(
        names: Optional[Dict[str, str]], roles: Optional[Dict[str, str]]
    ) -> SpeakerRoster:
        return resolve_speaker_roster(
            diarization,
            transcript_text,
            detected_guests=guests,
            known_hosts=known_hosts,
            voice_texts=voice_texts,
            ordered_turns=ordered_turns,
            ad_intervals=ad_intervals,
            metadata_named=_md_named,
            llm_voice_names=names,
            llm_voice_roles=roles,
            cleaning=cleaning,
            recurring_text=recurring_text,
            diarization_provider=dz_provider,
            profile=_labeling_profile,
        )

    roster = _run_roster(llm_voice_names, llm_voice_roles)
    # ADR-137 attribution — how much the LLM did vs the deterministic cues. A second roster pass
    # with the LLM inputs emptied is the pure-cue BASELINE; the diff against the shipped roster is
    # the LLM's marginal contribution. The baseline pass is deterministic (no network), so it is
    # cheap, and it only runs when the LLM actually produced something.
    resolution_attribution: Optional[Dict[str, Any]] = None
    if llm_voice_names or llm_voice_roles:
        baseline_roster = _run_roster({}, {})
        # Enforce the ADDITIVE contract: the LLM path must never un-name a voice the cues resolved.
        roster, restored_names = _reconcile_non_regression(baseline_roster, roster)
        resolution_attribution = _resolution_attribution(baseline_roster, roster)
        resolution_attribution["llm_delta"]["names_restored"] = restored_names
        _d = resolution_attribution["llm_delta"]
        logger.info(
            "resolution attribution: LLM added %d name(s), changed %d role(s) vs the "
            "deterministic baseline",
            len(_d["names_added"]),
            len(_d["roles_changed"]),
        )
        if restored_names:
            logger.warning(
                "resolution non-regression: restored %d deterministic name(s) the LLM path "
                "dropped (voices: %s)",
                len(restored_names),
                ", ".join(restored_names),
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
        profile=_labeling_profile,
        detection_ran=detection_ran,
    )
    if resolution_attribution is not None:
        enriched_result["speaker_diagnostics"]["resolution_attribution"] = resolution_attribution
    enriched_result["diarization_num_speakers"] = roster.num_speakers
    # ADR-132 provenance: the ACTUAL diarization model served (e.g. pyannote/speaker-diarization-
    # community-1), so the processing manifest records which model produced the speaker turns — the
    # diarization analogue of the ASR model_used, previously absent from the manifest's diar block.
    enriched_result["diarization_model_name"] = getattr(diarization, "model_name", None) or None
    # ADR-131: the diarizer's total SPEECH duration (Σ merged speaker turns) — the denominator for
    # the speech-normalized coverage gate. Provider-agnostic (any DiarizationResult). Non-speech
    # (music/ads/silence) has no speaker turn, so it is excluded here, unlike raw audio duration.
    enriched_result["diarization_speech_seconds"] = merged_speech_seconds(diarization.segments)
    # RFC-109 / ADR-132: per-episode diarization cost. Cloud diarizers (Deepgram/Gemini) bill per
    # audio-minute; local diarizers (pyannote/DGX/MOSS) have no pricing entry -> None. Bill on the
    # widest end across the ASR transcript + diarization turns (closest in-memory audio proxy).
    _audio_ends = [
        e
        for e in (
            _segment_end(s)
            for s in list(result.get("segments") or []) + list(diarization.segments or [])
        )
        if e is not None
    ]
    enriched_result["diarization_cost_usd"] = _estimate_diarization_cost(
        diarization, cfg, audio_seconds=(max(_audio_ends) if _audio_ends else None)
    )
    return enriched_result
