"""Today's in-tree stages, on the pipeline's stage hooks (E04 S2).

Each entry imports its stage inside its body, so this module loads without the stage code. An
entry goes when its stage moves to an extension, which then contributes it instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .stage_hooks import PipelineHooks

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


BUILTIN = PipelineHooks(
    name="builtin",
    run_started=(_validate_grounding_dependencies,),
    corpus_derived=(_write_ad_signatures,),
    batch_finalized=(_measure_naming_coverage,),
    run_finalized=(_build_audio_references,),
    run_completed=(_cluster_corpus_topics,),
    kg_prompt_hints=_resolve_ner_prepass,
    kg_graph_passes=(_collapse_mentions_into_speakers,),
)
