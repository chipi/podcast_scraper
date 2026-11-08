"""Speaker reprocess stages: relabel_only and rediarize_only (moved verbatim from
``workflow.episode_processor``, E04 S2).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from podcast_scraper import config
from podcast_scraper.models import TranscriptionJob
from podcast_scraper.utils import filesystem
from podcast_scraper.workflow.episode_processor import (
    _detection_report_of,
    _duration_from_media_file,
    _existing_transcript_for,
    _format_transcript_if_needed,
    _ingest_audio_bytes,
    _invalidate_translation,
    _operator_renames,
    _preprocess_audio_if_needed,
    _produce_transcript_sidecars,
    _record_episode_diarization,
    _record_unresolved_transcript,
    _relabel_cleaned_transcript,
    _save_speaker_diagnostics_file,
    _save_transcript_segments_file,
    _split_diarize_naming,
    _write_processing_manifest,
    rediarize_audio_mismatch,
)

logger = logging.getLogger(__name__)


def _feed_hosts_from_sibling_metadata(txt_path: Path) -> List[str]:
    """Read the feed-stated host names from a transcript's sibling metadata JSON.

    ``<run>/transcripts/<name>.txt`` -> ``<run>/metadata/<name>.metadata.json``. The feed blurb
    usually names the hosts; ``detect_hosts_from_feed`` extracts them. Any failure (missing file,
    malformed JSON, no feed block) returns ``[]`` — the roster then falls back to self-intro only.
    """
    import json as _json

    from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_feed

    stem = txt_path.name[: -len(".txt")] if txt_path.name.endswith(".txt") else txt_path.stem
    md_path = txt_path.parent.parent / filesystem.METADATA_SUBDIR / f"{stem}.metadata.json"
    try:
        feed = _json.loads(md_path.read_text(encoding="utf-8")).get("feed", {})
    except (OSError, ValueError):
        return []
    if not isinstance(feed, dict):
        return []
    return sorted(
        detect_hosts_from_feed(
            feed.get("title"),
            feed.get("description"),
            feed.get("authors") or [],
            language=str(feed.get("language") or ""),
        )
    )


def _rewrite_speaker_record_in_place(
    txt_path: Path,
    job: TranscriptionJob,  # type: ignore[valid-type]
    effective_output_dir: str,
    rel_path: str,
    stage: str,
    feed_hosts: Sequence[str],
) -> bool:
    """Rewrite ``content.speakers`` from the labels this reprocess stage just wrote (#2075).

    THE RECORD IS THE POINT OF THIS ARC, and until now no reprocess stage touched it. `relabel_only`
    (and through it `rediarize_only` / `retranscript_only`) rewrote the transcript, the segments,
    the diagnostics, the ad-free and cleaned copies and the manifest — and left `metadata.json`
    exactly as it was. Measured on two DGX episodes: `metadata.json` byte-identical before and
    after, still the pre-1.2.0 ``{id, name, role}`` shape with no ``placed``/``voices``. A relabel
    that CHANGED a name would therefore leave the record naming the old person, which is precisely
    the divergence the record exists to prevent.

    Rebuilt through the SAME path a full run uses, reading the segments and diagnostics this stage
    has just written to disk, so the two cannot drift. Only the three roster keys are touched: a
    frozen stage has no new information about anything else in the artifact, and a wider write
    would re-open summaries and derived fields it deliberately skipped.

    Returns True when the record was rewritten. Never raises: a failure here must not lose the
    relabel that already succeeded — it is logged, and the audit reports the stale record.
    """
    import json as _json

    from podcast_scraper.identity.roster_provenance import roster_source
    from podcast_scraper.workflow.metadata_generation import _build_speaker_record

    stem = txt_path.name[: -len(".txt")] if txt_path.name.endswith(".txt") else txt_path.stem
    md_dir = txt_path.parent.parent / filesystem.METADATA_SUBDIR
    md_path = md_dir / f"{stem}.metadata.json"
    if not md_path.is_file():
        # THE TWO FILENAMES DO NOT HAVE TO MATCH. The metadata name truncates the title and the
        # transcript name does not, so on a downloaded transcript the stems differ by the title's
        # tail alone — 280 of 2,298 records on the production snapshot. Deriving the record from
        # the transcript stem therefore missed them silently ("no sibling metadata"), and a
        # relabel that renamed a speaker left the record naming the old person: the exact
        # divergence this function exists to prevent. Same predicate the audit and the transcript
        # resolver use, so all three agree on what "the same episode" means.
        match = next(
            (
                candidate
                for candidate in sorted(md_dir.glob("*.metadata.json"))
                if filesystem.names_the_same_episode(candidate.name[: -len(".metadata.json")], stem)
            ),
            None,
        )
        if match is None:
            logger.warning(
                "[%s] %s: no metadata in %s names the same episode as %s — "
                "the speaker record was NOT updated",
                job.idx,
                stage,
                md_dir,
                stem,
            )
            return False
        md_path = match
    try:
        payload = _json.loads(md_path.read_text(encoding="utf-8"))
        feed_block = payload.get("feed") if isinstance(payload.get("feed"), dict) else {}
        content = payload.get("content")
        if not isinstance(content, dict):
            raise ValueError("artifact has no content block")
        speakers, num_speakers = _build_speaker_record(
            effective_output_dir,
            rel_path,
            # THE SAME host anchor the relabel itself resolved with, not the live one. The stage
            # prefers the frozen sibling metadata over live feed detection so a relabel of a stored
            # corpus is reproducible; rebuilding the record from `job.feed_hosts` would describe a
            # different resolution from the one that produced the labels on disk.
            list(feed_hosts or []),
            list(job.detected_speaker_names or []),
            (feed_block or {}).get("title"),
        )
        before = content.get("speakers")
        content["speakers"] = [sp.model_dump(exclude_none=False) for sp in speakers]
        source = roster_source(speakers, diarized=any(sp.placed for sp in speakers))
        if source is not None:
            content["speakers_source"] = source
        if num_speakers is not None:
            content["diarization_num_speakers"] = num_speakers
        # REPOINT THE RECORD AT THE TRANSCRIPT WE ACTUALLY JUST WROTE. On a #2082 episode the
        # stored pointer names another episode's file; the resolver now refuses to follow it, but
        # leaving it in place means the pairing audit keeps reporting the episode forever and
        # "done when the audit exits 0" is unreachable by repair. The value is relative to the RUN
        # dir, which is the metadata file's grandparent — the same join the audit reads it back
        # with.
        run_dir = md_path.parent.parent
        try:
            pointer = str(Path(txt_path).resolve().relative_to(run_dir.resolve()))
        except ValueError:
            pointer = ""
        if pointer and content.get("transcript_file_path") != pointer:
            logger.info(
                "[%s] %s: repointing transcript_file_path %r -> %r",
                job.idx,
                stage,
                content.get("transcript_file_path"),
                pointer,
            )
            content["transcript_file_path"] = pointer
        md_path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "[%s] %s: could not rewrite the speaker record at %s (%s) — the transcript was "
            "relabelled but the record still describes the OLD naming",
            job.idx,
            stage,
            md_path,
            exc,
        )
        return False
    logger.info(
        "[%s] %s: speaker record rewritten — %d entries (%d placed), was %d",
        job.idx,
        stage,
        len(speakers),
        sum(1 for sp in speakers if sp.placed),
        len(before or []) if isinstance(before, list) else 0,
    )
    return True


def _embeddings_carried_over(
    txt_path: Path, cluster_ids: Mapping[str, str]
) -> Tuple[Dict[str, List[float]], str]:
    """The stored voice embeddings, re-keyed to the voice ids a relabel assigns.

    A relabel rewrites the diagnostics sidecar from a fresh roster, and it renumbers the voices
    (``_anon``: first appearance order), so without this it would silently drop the embeddings
    the original diarization stored. Only a voice whose stored id is in ``cluster_ids`` keeps its
    vector; a corpus whose segments carry names instead of raw ids never had any. Returns the
    vectors and the model that made them (vectors of different models are not comparable).
    """
    import json as _json

    from podcast_scraper.providers.ml.diarization.base import clean_embeddings

    sidecar = txt_path.with_name(txt_path.name[: -len(".txt")] + ".speakers.diagnostics.json")
    try:
        stored = _json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, ""
    block = stored.get("voice_embeddings") if isinstance(stored, dict) else None
    if not isinstance(block, dict):
        return {}, ""
    vectors = clean_embeddings(block.get("by_voice"))
    carried = {cluster_ids[v]: vec for v, vec in vectors.items() if v in cluster_ids}
    return carried, str(block.get("model_name") or "") if carried else ""


def _relabel_existing_transcript(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider,
    pipeline_metrics,
) -> tuple[bool, Optional[str], int]:
    """pipeline_stage=relabel_only: re-resolve speaker names on the existing on-disk
    transcript + frozen ``SPEAKER_NN`` diarization, re-render the screenplay, and overwrite
    the transcript / segments / ad-free base in place. No audio, no re-ASR, no re-diarize.
    """
    import json as _json

    from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
    from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

    # effective_output_dir is the *new* run dir this invocation created; the existing corpus
    # transcript lives in a sibling run_<old-tag>/transcripts/, and this stage OVERWRITES it in
    # place. Which file that is comes from the episode's own metadata record — see
    # `_existing_transcript_for` for why searching for it by index prefix was wrong.
    txt_path = _existing_transcript_for(job, effective_output_dir, "relabel_only")
    if txt_path is None:
        _record_unresolved_transcript(job, cfg, pipeline_metrics, "relabel_only")
        return False, None, 0
    # SWAP THE SOURCE BACK FIRST. On a translated episode the canonical pair is the ENGLISH render
    # (D-44): reading it named the speakers on English text with the source language's vocabulary,
    # and the post-processing swap-back then moved that English aside — the relabel's work with
    # it. Radio Ambulante (es, 2026-10-09) ended a relabel with 0 named entries that way.
    # translate_only already swaps back before handing off here; this makes relabel_only do the
    # same, and makes the later swap-back a no-op. The translation memory is kept (D-33), so the
    # English re-render that follows costs no GPU.
    from podcast_scraper.languages import resolve_config_language
    from podcast_scraper.workflow.transcript_resolution import _canonical_relpath

    _root = str(txt_path.parent.parent)
    _raw_lang, _relabel_language, _lang_src = resolve_config_language(cfg)
    _invalidate_translation(
        _canonical_relpath(os.path.relpath(str(txt_path), _root)), _root, _relabel_language
    )
    seg_path = txt_path.with_name(txt_path.name[: -len(".txt")] + ".segments.json")
    if not seg_path.exists():
        logger.warning(
            "[%s] relabel_only: transcript has no .segments.json; cannot relabel", job.idx
        )
        _record_unresolved_transcript(job, cfg, pipeline_metrics, "relabel_only")
        return False, None, 0

    text = txt_path.read_text(encoding="utf-8")
    segs = _json.loads(seg_path.read_text(encoding="utf-8"))
    # Frozen-clustering source. A finished corpus stores the diarization identity in
    # ``speaker_label`` (``speaker`` is None), and for already-resolved voices that label is v2's
    # RESOLVED NAME ("Amy Lawrence"), not a raw ``SPEAKER_NN``. Relabel must re-resolve from the
    # transcript, NOT inherit v2's names — so remap every distinct label (name or SPEAKER_NN) to a
    # fresh anonymous ``SPEAKER_NN`` in first-appearance order. This preserves the clustering (same
    # label -> same voice) while stripping v2's naming, so the roster decides afresh.
    _cluster_ids: Dict[str, str] = {}

    def _anon(label: str) -> str:
        if label not in _cluster_ids:
            _cluster_ids[label] = f"SPEAKER_{len(_cluster_ids):02d}"
        return _cluster_ids[label]

    def _identity(s: dict) -> Optional[str]:
        # ``is not None`` so a native diarizer's int-0 speaker id is not dropped by a falsy ``or``.
        v = s.get("speaker")
        if v is None:
            v = s.get("speaker_label")
        return str(v) if v is not None else None

    dsegs = [
        DiarizationSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            speaker=_anon(_identity(s)),  # type: ignore[arg-type]
        )
        for s in segs
        if isinstance(s, dict) and _identity(s) is not None
    ]
    if not dsegs:
        logger.warning(
            "[%s] relabel_only: segments carry no speaker identity; nothing to relabel", job.idx
        )
        _record_unresolved_transcript(
            job,
            cfg,
            pipeline_metrics,
            "relabel_only",
            error_type="NoSpeakerIdentity",
            detail="segments carry no speaker identity; nothing to relabel",
        )
        return False, None, 0
    _carried, _carried_model = _embeddings_carried_over(txt_path, _cluster_ids)
    diar = DiarizationResult(
        segments=dsegs,
        num_speakers=len(_cluster_ids),
        speaker_embeddings=_carried,
        speaker_embeddings_model=_carried_model,
    )

    # THE PUBLISHER'S LABELS SURVIVE A RELABEL. `_anon` above deliberately strips every identity
    # so names are re-resolved from evidence — correct for names WE inferred, wrong for names the
    # SOURCE stated. `stated_speaker` carries those (written by `_roster_native_segments`), and
    # they are handed back to the roster at top precedence, exactly as on the first pass.
    #
    # TWO CARRIERS, because two different writers produce this file and only one of them is new.
    # `stated_speaker` is written by the download path added alongside this. But `retranscript_only`
    # — the stage that repairs the 129 production episodes whose spans were never stored at all —
    # writes the PARSED CUES verbatim: `{start, end, text, speaker}`, where `speaker` is the
    # publisher's own label and there is no `stated_speaker` to find. Reading only the new key
    # meant the repair ran, logged "rewrote ... with 2 speaker(s); relabelling", and published
    # `SPEAKER_00` / `SPEAKER_01` — every stated name discarded, which is the whole point of the
    # repair. Verified by simulating the stage's exact output before this line existed.
    #
    # ONLY `speaker`, NEVER `speaker_label`. That is the same split the clustering comment above
    # draws: `speaker` is what the SOURCE said and is not ours to revise; `speaker_label` is OUR
    # resolved answer from a previous run, and re-deriving it is what relabel is FOR. Treating the
    # latter as stated would freeze v2's names forever.
    from podcast_scraper.providers.ml.diarization.roster import _is_person_voice_label

    _stated_relabel: Dict[str, str] = {}
    for s in segs:
        if not isinstance(s, dict):
            continue
        stated = s.get("stated_speaker")
        if not stated:
            raw_voice = s.get("speaker")
            if raw_voice is not None and _is_person_voice_label(str(raw_voice)):
                stated = str(raw_voice).strip()
        ident = _identity(s)
        if stated and ident is not None:
            _stated_relabel.setdefault(_anon(ident), str(stated))
    result: dict = {
        "text": text,
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
            for s in segs
            if isinstance(s, dict)
        ],
    }
    # The corpus's own stored metadata carries the feed blurb that names the hosts
    # ("journalists Kevin Roose and Casey Newton explore..."). Feed it to the roster so an
    # ASR-garbled spoken surname ("Kevin Russo") canonicalizes to the feed's spelling. Read from
    # the sibling <run>/metadata/<name>.metadata.json; absence is non-fatal (relabel still runs).
    #
    # Q3 (advisor review): keep the FROZEN sibling metadata as the host anchor — a relabel of a
    # stored corpus must be reproducible, not track live feed drift. But when the sibling is
    # missing/unreadable it silently returns [], the WORST anchor state, while the live detection
    # (job.feed_hosts, statement + NER fallback + corroboration) was already computed — so fall back
    # to it only then. Log any sibling-vs-live divergence so the freeze-vs-live choice can later be
    # decided from data rather than assumption.
    sibling_hosts = _feed_hosts_from_sibling_metadata(txt_path)
    live_hosts = list(getattr(job, "feed_hosts", None) or [])
    feed_hosts = sibling_hosts or live_hosts
    if sibling_hosts and live_hosts and sorted(sibling_hosts) != sorted(live_hosts):
        logger.info(
            "[%s] relabel_only: feed_hosts divergence — sibling-metadata %s vs live %s (using "
            "sibling for a reproducible relabel)",
            job.idx,
            sibling_hosts,
            live_hosts,
        )
    naming_t0 = time.monotonic()
    result = apply_diarization_to_result(
        result,
        "",
        cfg,
        job.detected_speaker_names,
        metadata_named=job.metadata_named,
        precomputed_diarization=diar,
        stated_voice_names=_stated_relabel or None,
        feed_hosts=feed_hosts,
        unbindable_hosts=getattr(getattr(job, "episode", None), "unbindable_hosts", None),
        # ADR-137 — title + description feed the LLM's host/guest role determination and gate
        # role-only resolution. FULL passes both; relabel_only omitting them resolved on a strictly
        # weaker prompt ("(not provided)"), the structural half of the relabel!=full confound.
        episode_title=job.ep_title,
        episode_description=getattr(job.episode, "description", None),
        detection_ran=getattr(job, "speaker_detection_ran", None),
        detection_report=_detection_report_of(job),
        speaker_renames=_operator_renames(job),
    )
    naming_elapsed = time.monotonic() - naming_t0
    # AND THEY SURVIVE THE NEXT RELABEL TOO. The roster is handed a stripped
    # ``{start, end, text}`` view, so whatever it returns carries no ``stated_speaker`` — writing
    # that straight back would make the FIRST relabel keep the publisher's names and the SECOND
    # fall back to the feed. Re-attach from the stored segments, in the order they were passed.
    _old_dicts = [s for s in segs if isinstance(s, dict)]
    _new_dicts = result.get("segments") if isinstance(result, dict) else None
    if isinstance(_new_dicts, list):
        for _old, _new in zip(_old_dicts, _new_dicts):
            if isinstance(_new, dict) and _old.get("stated_speaker"):
                _new["stated_speaker"] = _old["stated_speaker"]
    new_text = _format_transcript_if_needed(
        result, cfg, job.detected_speaker_names, transcription_provider
    )
    txt_path.write_text(new_text, encoding="utf-8")
    rel_path = os.path.relpath(str(txt_path), effective_output_dir)
    new_segs = result.get("segments") if isinstance(result, dict) else None
    if isinstance(new_segs, list) and new_segs:
        _save_transcript_segments_file(new_segs, rel_path, effective_output_dir)
        _save_speaker_diagnostics_file(
            result.get("speaker_diagnostics") if isinstance(result, dict) else None,
            rel_path,
            effective_output_dir,
        )
        _produce_transcript_sidecars(
            cfg, new_text, new_segs, rel_path, effective_output_dir, episode=job.episode
        )
        _relabel_cleaned_transcript(txt_path, segs, new_segs, job.idx)
        # The record follows the labels, or the two describe different episodes (#2075).
        _rewrite_speaker_record_in_place(
            txt_path, job, effective_output_dir, rel_path, "relabel_only", feed_hosts
        )
    # advisor #2: relabel rewrites naming on disk — the manifest MUST record the new naming
    # method_version, or "reprocess episodes below naming-3" never converges. result has no ASR/
    # diarization fields (frozen), so only the naming block is written + a pipeline_stage emitted.
    _write_processing_manifest(
        result, cfg, job, rel_path, effective_output_dir, naming_elapsed=naming_elapsed
    )
    logger.info("[%s] relabel_only: re-resolved speaker names in place -> %s", job.idx, rel_path)
    return True, rel_path, 0


def _rediarize_existing_transcript(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider,
    pipeline_metrics,
) -> tuple[bool, Optional[str], int]:
    """pipeline_stage=rediarize_only (v2.2): re-diarize the downloaded audio with the profile's
    diarizer (DGX pyannote) and align the FRESH voices to the existing on-disk ASR transcript —
    reusing its text + timestamps, so NO re-ASR — then re-resolve names, re-render the screenplay,
    and overwrite in place. GI/KG cascade. The decoupled sibling of ``relabel_only``: same
    read/render/save machinery, but the diarization is regenerated from audio rather than frozen.
    """
    import json as _json

    from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

    audio_path = job.temp_media
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("[%s] rediarize_only: no downloaded audio; cannot re-diarize", job.idx)
        _record_unresolved_transcript(
            job,
            cfg,
            pipeline_metrics,
            "rediarize_only",
            error_type="NoAudio",
            detail="no downloaded audio; cannot re-diarize",
        )
        return False, None, 0

    # Locate the existing transcript — same discovery as relabel, from the episode's own record.
    txt_path = _existing_transcript_for(job, effective_output_dir, "rediarize_only")
    if txt_path is None:
        # Same vanish-between-selection-and-summary hole the relabel path had.
        _record_unresolved_transcript(job, cfg, pipeline_metrics, "rediarize_only")
        return False, None, 0
    seg_path = txt_path.with_name(txt_path.name[: -len(".txt")] + ".segments.json")

    text = txt_path.read_text(encoding="utf-8")
    segs = _json.loads(seg_path.read_text(encoding="utf-8"))
    # Reuse the existing ASR text + per-segment timestamps; the fresh diarization aligns to THESE.
    result: dict = {
        "text": text,
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
            for s in segs
            if isinstance(s, dict)
        ],
    }
    if not result["segments"]:
        logger.warning("[%s] rediarize_only: transcript has no segments to align", job.idx)
        _record_unresolved_transcript(
            job,
            cfg,
            pipeline_metrics,
            "rediarize_only",
            error_type="NoSegments",
            detail="transcript has no segments to align",
        )
        return False, None, 0

    mismatch = rediarize_audio_mismatch(
        os.path.getsize(audio_path),
        _ingest_audio_bytes(txt_path),
        _duration_from_media_file(job),
        max(float(s.get("end") or 0.0) for s in result["segments"]),
    )
    if mismatch:
        logger.warning("[%s] rediarize_only: refused, %s", job.idx, mismatch)
        _record_unresolved_transcript(
            job, cfg, pipeline_metrics, "rediarize_only", error_type="AudioChanged", detail=mismatch
        )
        return False, None, 0

    feed_hosts = _feed_hosts_from_sibling_metadata(txt_path)
    # The audio a full run diarizes: the profile's preprocessing preset (resample, re-encode,
    # loudness), as in transcribe_media_to_text. The raw download gave pyannote different audio
    # from every episode the pipeline had processed, so fresh and stored voices were not
    # comparable.
    media = _preprocess_audio_if_needed(job, cfg, audio_path, pipeline_metrics)
    # No precomputed diarization + bypass_cache_read -> apply_diarization diarizes the AUDIO fresh
    # with the profile's diarizer and aligns the new voices to result["segments"] (the ASR text).
    dz_t0 = time.monotonic()
    dz_timings: Dict[str, float] = {}
    result = apply_diarization_to_result(
        result,
        media,
        cfg,
        job.detected_speaker_names,
        metadata_named=job.metadata_named,
        feed_hosts=feed_hosts,
        unbindable_hosts=getattr(getattr(job, "episode", None), "unbindable_hosts", None),
        # ADR-137: the naming step reads the same title and description as on a full run.
        episode_title=job.ep_title,
        episode_description=getattr(job.episode, "description", None),
        bypass_cache_read=True,
        detection_report=_detection_report_of(job),
        speaker_renames=_operator_renames(job),
        timings=dz_timings,
    )
    dz_elapsed, naming_elapsed = _split_diarize_naming(time.monotonic() - dz_t0, dz_timings)
    new_text = _format_transcript_if_needed(
        result, cfg, job.detected_speaker_names, transcription_provider
    )
    txt_path.write_text(new_text, encoding="utf-8")
    rel_path = os.path.relpath(str(txt_path), effective_output_dir)
    new_segs = result.get("segments") if isinstance(result, dict) else None
    if isinstance(new_segs, list) and new_segs:
        _save_transcript_segments_file(new_segs, rel_path, effective_output_dir)
        _save_speaker_diagnostics_file(
            result.get("speaker_diagnostics") if isinstance(result, dict) else None,
            rel_path,
            effective_output_dir,
        )
        _produce_transcript_sidecars(
            cfg, new_text, new_segs, rel_path, effective_output_dir, episode=job.episode
        )
        # Fresh voices AND fresh names: the record must describe the diarization that now exists.
        _rewrite_speaker_record_in_place(
            txt_path, job, effective_output_dir, rel_path, "rediarize_only", feed_hosts
        )
    # advisor #2: rediarize regenerates diarization + naming on disk — record both into run metrics
    # and the manifest (fresh diarization + naming blocks + pipeline_stage), else the rerun is
    # invisible in diarization_* metrics and the manifest keeps the old versions.
    _record_episode_diarization(pipeline_metrics, result)
    _write_processing_manifest(
        result,
        cfg,
        job,
        rel_path,
        effective_output_dir,
        pipeline_metrics=pipeline_metrics,
        diarization_elapsed=dz_elapsed,
        naming_elapsed=naming_elapsed,
    )
    logger.info("[%s] rediarize_only: re-diarized + re-resolved in place -> %s", job.idx, rel_path)
    return True, rel_path, 0
