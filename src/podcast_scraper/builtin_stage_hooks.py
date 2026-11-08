"""Today's in-tree stages, on the pipeline's stage hooks (E04 S2).

Each entry imports its stage inside its body, so this module loads without the stage code. An
entry goes when its stage moves to an extension, which then contributes it instead.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .languages import transcription_language
from .stage_hooks import PipelineHooks, RosterSourceValue

logger = logging.getLogger(__name__)


def _validate_grounding_dependencies(cfg: Any) -> None:
    from podcast_scraper.gi.deps import validate_gil_grounding_dependencies

    validate_gil_grounding_dependencies(cfg)


def _write_ad_signatures(corpus_root: Path) -> None:
    """Cross-show ad signatures: read segments, not the LanceDB index. Never raises; a missing or
    stale file makes the rule abstain."""
    from podcast_scraper.providers.ml.diarization.ad_signatures import write_for_corpus

    write_for_corpus(Path(corpus_root))


def _measure_naming_coverage(corpus_parent: Path) -> None:
    """The served corpus's naming coverage, once per batch, for the API's /metrics (#2304 S8).
    Absolute, not a diff: an episode no code version names was invisible to every other check."""
    from podcast_scraper.quality.naming_coverage import measure_and_record

    measure_and_record(Path(corpus_parent))


def _build_audio_references(corpus_root: str, run_dir: str) -> None:
    """Each new episode's audio reference, from the audio it was just transcribed from (RFC-129,
    #2359). Before eviction: that is the last moment the exact audio is on local disk. Never
    fatal: a run without numpy or ffmpeg simply has no references."""
    try:
        from podcast_scraper.audio_sync.references import write_for_run

        write_for_run(corpus_root, run_dir)
    except Exception:  # pragma: no cover - a finalize step must never break a finished run
        logger.warning("audio references: end-of-run pass failed (non-fatal)", exc_info=True)


def _cluster_corpus_topics(cfg: Any, output_dir: str) -> None:
    """#1058 chunk 3: corpus-level Topic clustering over every kg.json on disk. Gated on
    cfg.kg_topic_corpus_clustering (default off; airgapped* overlays flip it on). Non-fatal: a
    failure here doesn't bring down a successful run."""
    if not getattr(cfg, "kg_topic_corpus_clustering", False) or cfg.dry_run:
        return
    try:
        from podcast_scraper.kg.topic_clustering import cluster_and_apply_corpus_topics

        summary = cluster_and_apply_corpus_topics(Path(output_dir))
        logger.info(
            "corpus topic clustering: clusters=%d concept_topics_added=%d "
            "related_to_edges_added=%d artifacts_mutated=%d",
            summary.clusters_found,
            summary.concept_topics_added,
            summary.related_to_edges_added,
            summary.artifacts_mutated,
        )
    except Exception as cluster_exc:
        logger.warning(
            "corpus topic clustering failed (non-fatal): %s",
            cluster_exc,
            exc_info=True,
        )


def _resolve_ner_prepass(
    transcript_text: str,
    cfg: Optional[Any],
    kg_extraction_provider: Any,
    max_entities: int,
) -> tuple[List[Dict[str, str]], str]:
    """#1035 — NER pre-pass: returns (hints, prompt_version).

    When ``cfg.kg_extraction_use_ner_prepass`` is False (the Field default is
    ``True``; see config.py) or the NER pipeline isn't available, returns
    ``([], "v4")`` — caller renders the v4 prompt with no hints. When enabled
    and the spaCy model resolves, returns the deduped + capped PERSON+ORG
    candidate list and prompt ``"v5"``.

    NB (#1540): the shipped cloud image is ``[llm,search]`` — spaCy lives only
    in ``[ml]`` — so on cloud_balanced this pass load-fails and silently
    downgrades to v4. Whether that materially hurts cloud KG entity recall
    (the #1035 0%→100% win was measured on DGX models, not the litellm cloud
    model) is an open question tracked in the ML-opt-in discussion issue.

    Reuses ``kg_extraction_provider._spacy_nlp`` when cached (Issue #387)
    or falls back to a fresh ``get_ner_model(cfg)`` load. Any failure
    silently downgrades to v4 — entity recall stays at the pre-#1035
    baseline, never worse.
    """
    if cfg is None or not getattr(cfg, "kg_extraction_use_ner_prepass", False):
        return [], "v4"
    nlp = getattr(kg_extraction_provider, "_spacy_nlp", None)
    if nlp is None:
        try:
            from podcast_scraper.providers.ml.speaker_detection import get_ner_model

            nlp = get_ner_model(cfg)
        except Exception as exc:
            logger.warning(
                "#1035 NER pre-pass: could not load spaCy model (%s); falling back to v4 prompt",
                exc,
            )
            return [], "v4"
    if nlp is None:
        logger.debug("#1035 NER pre-pass: cfg.ner_model not set; falling back to v4 prompt")
        return [], "v4"

    from podcast_scraper.kg.ner_prepass import extract_kg_ner_hints

    cap = max(1, min(int(max_entities) * 3, 40))
    hints = extract_kg_ner_hints(transcript_text, nlp, max_candidates=cap)
    return hints, "v5"


def _collapse_mentions_into_speakers(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """A person MENTIONED in the episode they also SPOKE in is one Person node, not two.

    MEASURED on 400 sampled prod episodes: 20 distinct pairs where one spelling is a speaker and
    a near-identical one is a `mentioned` Person in the SAME episode —

        Joe Weisenthal (host) / Joe Wiesenthal      x5   Odd Lots
        Peter Attia, MD (host) / Peter Attia        x4   The Peter Attia Drive
        Tracy Alloway (host) / Tracy Allaway        x3   Odd Lots
        Theo Jaffe / Theo Jaffee                    x3   The a16z Show
        Mark Galeotti (host) / Mark Galliotti       x2   In Moscow's Shadows
        ...15 more

    Neither existing mechanism closes this. `_one_name_per_person` (the roster) operates on
    diarized VOICES and never sees a mention; `_dedupe_nodes_by_id` merges only nodes that already
    share a slug, and `joe-weisenthal` / `joe-wiesenthal` do not.

    THE SPEAKER'S SPELLING WINS. A voice is established evidence — somebody spoke and the roster
    resolved who. A mention is a name extracted from prose, where the ASR's rendering is whatever
    it heard. So the mention is rewritten onto the speaker's id, and every edge that referenced it
    is remapped rather than left dangling.

    THREE GUARDS, because merging two real people is worse than leaving a duplicate (#876):

    * only mention -> SPEAKER, never mention -> mention. The dangerous shape is several mentions of
      one family: in `ChinaTalk — "North Korea's Messiah"`, `Kim Jong-il`, `Kim Jong-un` and
      `Kim Il-sung` are three different people, all mentions, and `same_person` scores the first
      two at 0.857. Comparing only against speakers keeps them apart — that episode's only speaker
      is Jordan Schneider.
    * exactly ONE speaker may match. Two matches means nothing says which, and a guess there is
      precisely the failure this guards.
    * a generational disagreement settles it outright. `same_person`'s subset rule reads
      `Sam Lee` as `Sam Lee Jr`, but that suffix exists to tell a father from a son.
    """
    from podcast_scraper.graph_id_utils import GENERATIONAL_SUFFIXES
    from podcast_scraper.kg.pipeline import _remap_nodes
    from podcast_scraper.kg.speaker_coherence import same_person

    def _generation(name: str) -> str:
        toks = [x for x in name.lower().replace(".", " ").replace(",", " ").split() if x]
        return toks[-1] if toks and toks[-1] in GENERATIONAL_SUFFIXES else ""

    people = [n for n in nodes if isinstance(n, dict) and n.get("type") == "Person"]
    speakers = [
        n for n in people if str((n.get("properties") or {}).get("role") or "") in ("host", "guest")
    ]
    if not speakers:
        return nodes, edges

    remap: Dict[str, str] = {}
    for node in people:
        props = node.get("properties") or {}
        if str(props.get("role") or "") in ("host", "guest"):
            continue
        name = str(props.get("name") or "").strip()
        nid = str(node.get("id") or "")
        if not name or not nid:
            continue
        hits = [
            s
            for s in speakers
            if _generation(name) == _generation(str((s.get("properties") or {}).get("name") or ""))
            and same_person(name, str((s.get("properties") or {}).get("name") or ""))
        ]
        if len(hits) != 1:
            continue
        target = str(hits[0].get("id") or "")
        if target and target != nid:
            remap[nid] = target
            logger.info(
                "kg build_artifact: %r (mentioned) is %r (%s) — collapsing onto %s",
                name,
                (hits[0].get("properties") or {}).get("name"),
                (hits[0].get("properties") or {}).get("role"),
                target,
            )

    return _remap_nodes(nodes, edges, remap)


def _adfree_base(
    text: str,
    segments: Optional[List[Dict[str, Any]]],
    rel_transcript_path: str,
    output_dir: str,
    *,
    extra_cue_patterns: Optional[List[str]] = None,
) -> Optional[Tuple[str, Any]]:
    from podcast_scraper.workflow.adfree_transcript import produce_adfree_artifacts

    return produce_adfree_artifacts(
        text, segments, rel_transcript_path, output_dir, extra_cue_patterns=extra_cue_patterns
    )


def _excise_ads(
    text: str, *, segments: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, Optional[List[Dict[str, Any]]], Any]:
    from podcast_scraper.gi.ad_regions import excise_ad_regions

    return excise_ad_regions(text, segments=segments)


def _write_turns_artifacts(
    cfg: Any,
    text: str,
    segments: List[Dict[str, Any]],
    rel_transcript_path: str,
    effective_output_dir: str,
    *,
    adfree: Any = None,
    episode: Any = None,
) -> None:
    """Write ``turns.json`` for each variant and record the manifest ``turns`` block (RFC-123 §3).

    Best-effort as a whole: turns have no consumers in v1, so nothing here may cost an episode.
    """
    from podcast_scraper.workflow import processing_manifest as pm
    from podcast_scraper.workflow.transcript_resolution import adfree_transcript_relpath
    from podcast_scraper.workflow.turns_artifact import turns_manifest_metrics, write_turns_artifact

    language = transcription_language(cfg)
    turns_t0 = time.monotonic()
    try:
        raw_outcome = write_turns_artifact(
            text, segments, rel_transcript_path, effective_output_dir, language=language
        )
        adfree_outcome = None
        if adfree is not None:
            adfree_rel = adfree_transcript_relpath(rel_transcript_path)
            adfree_outcome = write_turns_artifact(
                adfree.text,
                adfree.segments,
                adfree_rel,
                effective_output_dir,
                language=language,
            )
    except Exception:  # noqa: BLE001 - a sidecar with no readers must never lose an episode
        logger.warning("turns: could not build turns for %s", rel_transcript_path, exc_info=True)
        return

    if raw_outcome.relpath:
        logger.info(
            "    saved turns artifact: %s (%d turns)", raw_outcome.relpath, raw_outcome.count
        )
    episode_id = None
    if episode is not None:
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        episode_id, _ = get_episode_id_from_episode(episode, cfg.rss_url or "")
    from podcast_scraper.utils import correlation

    pm.update_stage(
        effective_output_dir,
        rel_transcript_path,
        "turns",
        pm.stage_block(
            ran=True,
            method_version=pm.METHOD_VERSIONS["turns"],
            duration_s=time.monotonic() - turns_t0,
            # Measured and genuinely free: pure Python, milliseconds per episode. Not ``None``,
            # which would mean nobody measured it.
            cost_usd=0.0,
            metrics=turns_manifest_metrics(raw_outcome, adfree_outcome),
        ),
        episode_id=episode_id,
        feed_id=getattr(cfg, "rss_url", None),
        run_id=correlation.get_run_id() or getattr(cfg, "run_id", None),
    )


def _screenplay(segments: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_with_offsets,
    )

    return format_diarized_screenplay_with_offsets(segments)


def _speaker_turns(
    offset_segments: List[Dict[str, Any]], *, screenplay_text: str, max_sentence_words: int
) -> Any:
    from podcast_scraper.providers.ml.diarization.turns import build_turns

    return build_turns(
        offset_segments, screenplay_text=screenplay_text, max_sentence_words=max_sentence_words
    )


#: A single token with a dot INSIDE it, e.g. ``Latent.Space``. Only applied to split fragments.
_DOMAIN_TOKEN = re.compile(r"^[^\s.]+(?:\.[^\s.]+)+$")


def _split_person_list(text: str) -> List[str]:
    """A person name with a conjunction, split by the host-name splitter (#2197): a suffix stays
    with its name, an organisation or domain-shaped fragment is dropped, repeats collapse, and
    a name that does not split stays whole."""
    from podcast_scraper.speaker_detectors.hosts import has_org_markers, split_author_names

    parts: List[str] = []
    for part in split_author_names(text):
        if not any(ch.isalpha() for ch in part) or has_org_markers(part):
            continue
        if _DOMAIN_TOKEN.match(part) or part in parts:
            continue
        parts.append(part)
    if len(split_author_names(text)) < 2:
        return [text]
    return parts


def _feed_hosts(
    cfg: Any, feed: Any, episodes: List[Any], pipeline_metrics: Any = None, *, speaker_detector: Any
) -> Any:
    from podcast_scraper.speaker_stage import detection

    return detection.detect_feed_hosts_and_patterns(
        cfg, feed, episodes, pipeline_metrics, speaker_detector=speaker_detector
    )


def _episode_hosts(host_detection_result: Any, episode: Any, episode_people: Any = ()) -> Any:
    from podcast_scraper.speaker_stage import detection

    return detection.hosts_for_episode(host_detection_result, episode, episode_people)


def _episode_speakers(
    episode: Any,
    cfg: Any,
    host_detection_result: Any,
    pipeline_metrics: Any,
    *,
    skip_speaker_detection: bool = False,
    skip_reason: Optional[str] = None,
    skip_detail: Optional[Dict[str, Any]] = None,
) -> Any:
    from podcast_scraper.speaker_stage import detection

    return detection._detect_speakers_for_episode(
        episode,
        cfg,
        host_detection_result,
        pipeline_metrics,
        skip_speaker_detection=skip_speaker_detection,
        skip_reason=skip_reason,
        skip_detail=skip_detail,
    )


def _stated_show_hosts(
    title: Any, description: Any, language: str, authors: List[str]
) -> List[Tuple[str, str]]:
    """The feed statement's hosts (sorted), the names it refused, then each author tag's people
    that are neither the show nor an organisation: the order ``show.json`` lists them in."""
    from podcast_scraper.speaker_detectors.hosts import (
        hosts_from_feed_statement,
        is_network_or_org_author,
        names_the_show,
        refused_feed_statement_names,
        split_author_names,
    )

    out: List[Tuple[str, str]] = []
    for name in sorted(hosts_from_feed_statement(title, description, language)):
        out.append((name, "feed_statement"))
    for name in refused_feed_statement_names(title, description, language):
        out.append((name, "feed_statement_refused"))
    for tag in authors:
        for name in split_author_names(tag, language or None):
            if (
                name
                and not names_the_show(name, title, language or None)
                and not is_network_or_org_author(name, language or None)
            ):
                out.append((name, "author_tag"))
    return out


def _person_matcher() -> Any:
    from podcast_scraper.kg.speaker_coherence import same_person

    return same_person


def _speaker_reprocess(
    mode: str,
    job: Any,
    cfg: Any,
    run_suffix: Optional[str],
    output_dir: str,
    transcription_provider: Any,
    pipeline_metrics: Any,
) -> Tuple[bool, Optional[str], int]:
    from podcast_scraper.speaker_stage import reprocess

    run = {
        "relabel": reprocess._relabel_existing_transcript,
        "rediarize": reprocess._rediarize_existing_transcript,
    }[mode]
    return run(job, cfg, run_suffix, output_dir, transcription_provider, pipeline_metrics)


def _speakers_after_asr(*args: Any) -> Tuple[Any, Optional[float], Optional[float], Optional[str]]:
    from podcast_scraper.speaker_stage import after_asr

    return after_asr.speakers_after_asr(*args)


def _anon_transcript(
    segments: List[Dict[str, Any]], rel_transcript_path: str, output_dir: str
) -> Optional[str]:
    from podcast_scraper.speaker_stage import sidecars

    return sidecars._write_anon_transcript(segments, rel_transcript_path, output_dir)


def _render_speakers(segments: List[Dict[str, Any]]) -> Optional[str]:
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_from_segments,
    )

    return format_diarized_screenplay_from_segments(segments)


def _person_voice_label(label: str) -> bool:
    from podcast_scraper.providers.ml.diarization.roster import _is_person_voice_label

    return _is_person_voice_label(label)


def _native_roster(result: Dict[str, Any], cfg: Any, **context: Any) -> Dict[str, Any]:
    from podcast_scraper.speaker_stage import native

    return native._roster_native_segments(result, cfg, **context)


def _carry_speaker_identity(new: Any, old: Any) -> List[Dict[str, Any]]:
    from podcast_scraper.providers.ml.diarization.alignment import carry_speaker_identity

    return carry_speaker_identity(new, old)


def _episode_insights(**context: Any) -> Tuple[Any, ...]:
    from podcast_scraper.insight_stage import episode

    return episode.build_episode_insights(**context)


def _gi_kg_passes(**context: Any) -> Tuple[Any, ...]:
    from podcast_scraper.insight_stage import passes

    return passes.run_gi_kg_passes(**context)


def _speaker_record(
    output_dir: Optional[str],
    transcript_file_path: Optional[str],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    feed_title: Optional[str],
    language: Optional[str] = None,
) -> Tuple[List[Any], Optional[int]]:
    from podcast_scraper.speaker_stage import record

    return record._build_speaker_record(
        output_dir,
        transcript_file_path,
        detected_hosts,
        detected_guests,
        feed_title,
        language=language,
    )


def _speaker_lists_for_graph(
    speakers: Any, feed_title: Optional[str] = None, language: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    from podcast_scraper.speaker_stage import record

    return record._speaker_lists_for_graph(speakers, feed_title=feed_title, language=language)


def _roster_source(speakers: Any, *, diarized: bool) -> Optional[RosterSourceValue]:
    from podcast_scraper.identity.roster_provenance import roster_source

    return roster_source(speakers, diarized=diarized)


class _BundledNaming:
    """Today's bundled transcript naming (speaker_stage.bundled_speaker_naming), for one call."""

    def __init__(
        self, cfg: Any, output_dir: str, transcript_file_path: Optional[str], labels: List[str]
    ) -> None:
        self._cfg, self._output_dir, self._transcript = cfg, output_dir, transcript_file_path
        self._labels = labels

    def params(self) -> Dict[str, Any]:
        from podcast_scraper.providers.ml.diarization.transcript_naming import PARAMS_LABELS

        return {PARAMS_LABELS: self._labels}

    def record(self, call: str, speakers: Any) -> None:
        from podcast_scraper.speaker_stage import bundled_speaker_naming as bsn

        bsn.record(self._cfg, self._output_dir, self._transcript, call, speakers)

    def take_speakers(self, result: Any) -> Any:
        from podcast_scraper.speaker_stage import bundled_speaker_naming as bsn

        return bsn.take_speakers(result)


def _bundled_naming(
    cfg: Any, pipeline_mode: str, output_dir: str, transcript_file_path: Optional[str]
) -> Optional[_BundledNaming]:
    from podcast_scraper.speaker_stage import bundled_speaker_naming as bsn

    labels = bsn.labels_for_call(cfg, pipeline_mode, output_dir, transcript_file_path)
    if not labels:
        return None
    return _BundledNaming(cfg, output_dir, transcript_file_path, labels)


def _default_speaker_names() -> List[str]:
    from podcast_scraper.providers.ml.speaker_detection import DEFAULT_SPEAKER_NAMES

    return list(DEFAULT_SPEAKER_NAMES)


def _ner_model(cfg: Any) -> Any:
    from podcast_scraper.providers.ml.speaker_detection import get_ner_model

    return get_ner_model(cfg)


def _create_speaker_detector(cfg: Any) -> Any:
    from podcast_scraper.speaker_detectors import factory

    return factory.create_speaker_detector(cfg)


def _validate_labeling_profile(profile_id: str) -> None:
    from podcast_scraper.providers.ml.diarization.labeling_profile import get_profile

    get_profile(profile_id)


def _gi_index_rows(
    episode_root: Path,
    metadata_path: Path,
    doc: Dict[str, Any],
    *,
    scope_tag: str,
    episode_id: str,
    raw_feed_id: Any,
    published: Any,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    from podcast_scraper.insight_stage.index_rows import gi_index_rows

    return gi_index_rows(
        episode_root,
        metadata_path,
        doc,
        scope_tag=scope_tag,
        episode_id=episode_id,
        raw_feed_id=raw_feed_id,
        published=published,
    )


BUILTIN = PipelineHooks(
    name="builtin",
    run_started=(_validate_grounding_dependencies,),
    corpus_derived=(_write_ad_signatures,),
    batch_finalized=(_measure_naming_coverage,),
    run_finalized=(_build_audio_references,),
    index_rows=(_gi_index_rows,),
    run_completed=(_cluster_corpus_topics,),
    kg_prompt_hints=_resolve_ner_prepass,
    adfree_base=_adfree_base,
    excise_ads=_excise_ads,
    variant_turns=_write_turns_artifacts,
    screenplay=_screenplay,
    speaker_turns=_speaker_turns,
    feed_hosts=_feed_hosts,
    episode_hosts=_episode_hosts,
    episode_speakers=_episode_speakers,
    stated_show_hosts=_stated_show_hosts,
    person_matcher=_person_matcher,
    speaker_reprocess=_speaker_reprocess,
    speakers_after_asr=_speakers_after_asr,
    anon_transcript=_anon_transcript,
    render_speakers=_render_speakers,
    person_voice_label=_person_voice_label,
    native_roster=_native_roster,
    carry_speaker_identity=_carry_speaker_identity,
    episode_insights=_episode_insights,
    gi_kg_passes=_gi_kg_passes,
    speaker_record=_speaker_record,
    speaker_lists_for_graph=_speaker_lists_for_graph,
    roster_source=_roster_source,
    bundled_naming=_bundled_naming,
    default_speaker_names=_default_speaker_names,
    ner_model=_ner_model,
    create_speaker_detector=_create_speaker_detector,
    validate_labeling_profile=_validate_labeling_profile,
    kg_graph_passes=(_collapse_mentions_into_speakers,),
    split_person_list=_split_person_list,
)
