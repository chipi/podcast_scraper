"""Where the pipeline calls a stage it does not own (E04 S2).

The run and the multi-feed batch call these points by name and never import the stage code
behind them. Each point runs the built-in hooks first, then each installed extension's, in load
order. A hook owns its own guard: a finalize step that must never fail the run catches its own
errors, and a start-up check that must stop the run raises. The runner adds no guard of its own,
so moving a hook behind the seam changes nothing about how it fails.

The built-in hooks are today's in-tree stages (:mod:`.builtin_stage_hooks`). When a stage's code
moves to an extension, its built-in entry goes and the extension's ``pipeline_hooks`` carries it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

#: ``(cfg)``. Before any feed is fetched; raise to stop the run (a missing dependency).
RunStartedHook = Callable[[Any], None]
#: ``(corpus_root)``. Corpus-wide derivations from what is on disk, once the run's (or the
#: batch's) index is built: single-run finalize at the run's corpus root, or batch finalize at
#: the corpus parent. Must not raise.
CorpusDerivedHook = Callable[[Path], None]
#: ``(corpus_parent)``. Once per multi-feed batch, after every feed finished, before the
#: enrichment follow-up is queued. Must not raise.
BatchFinalizedHook = Callable[[Path], None]
#: ``(corpus_root, run_dir)``. End of a run, before local audio is evicted: the last moment the
#: run's audio is on local disk. Must not raise.
RunFinalizedHook = Callable[[str, str], None]
#: ``(cfg, output_dir)``. After every episode of the run is finalized and on disk, before the
#: run is marked done. Must not raise.
RunCompletedHook = Callable[[Any, str], None]
#: ``(transcript_text, cfg, kg_extraction_provider, max_entities) -> (hints, prompt_version)``.
#: Entity hints for the KG extraction prompt. One owner answers: the first that is set. Without
#: one the prompt is the plain ``v4`` with no hints.
KgPromptHints = Callable[[str, Any, Any, int], Tuple[List[Dict[str, str]], str]]
#: ``(nodes, edges) -> (nodes, edges)``. A pass over one episode's KG graph before the final
#: dedup; each pass gets the previous one's output.
#: ``(text, segments, rel_transcript_path, output_dir, *, extra_cue_patterns) ->
#: (adfree_relpath, artifacts) | None``. Builds and saves the episode's ``<base>.adfree.*`` (D-44).
#: One owner; without one no ad-free base is written.
AdfreeBase = Callable[..., Optional[Tuple[str, Any]]]
#: ``(text, *, segments) -> (text, segments, metadata)``: pre-roll and post-roll ads cut from a
#: transcript held in memory. One owner; without one the text is returned as it came, with
#: :class:`NoAdExcision`.
ExciseAds = Callable[..., Tuple[str, Optional[List[Dict[str, Any]]], Any]]
#: ``(cfg, text, segments, rel_transcript_path, output_dir, *, adfree, episode)``. Writes each
#: transcript variant's ``turns.json`` and the manifest's ``turns`` block (RFC-123 §3). Without an
#: owner no turns are written.
VariantTurns = Callable[..., None]
#: ``(segments) -> (text, offset_segments)``: segments rendered as speaker lines, with each
#: segment's char range in that text. One owner; without one there is no screenplay.
Screenplay = Callable[[List[Dict[str, Any]]], Tuple[str, List[Dict[str, Any]]]]
#: ``(offset_segments, *, screenplay_text, max_sentence_words) -> turns``: the turns of a
#: screenplay. One owner.
SpeakerTurns = Callable[..., Any]
#: ``(text) -> names``: the people one extracted person name lists (``Kahneman and Tversky``).
#: Called only for a name with a conjunction. One owner; without one it stays one name.
SplitPersonList = Callable[[str], List[str]]
KgGraphPass = Callable[
    [List[Dict[str, Any]], List[Dict[str, Any]]],
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]],
]


@dataclass(frozen=True)
class NoAdExcision:
    """What :func:`excise_ads` reports when no stage cuts ads: nothing was removed."""

    source_length: int = 0
    preroll_cut_end: Optional[int] = None
    postroll_cut_start: Optional[int] = None
    chars_removed: int = 0
    preroll_pattern_hits: int = 0
    postroll_pattern_hits: int = 0


@dataclass(frozen=True)
class PipelineHooks:
    """The stage hooks one owner contributes; every part is optional."""

    name: str
    run_started: Sequence[RunStartedHook] = field(default_factory=tuple)
    corpus_derived: Sequence[CorpusDerivedHook] = field(default_factory=tuple)
    batch_finalized: Sequence[BatchFinalizedHook] = field(default_factory=tuple)
    run_finalized: Sequence[RunFinalizedHook] = field(default_factory=tuple)
    run_completed: Sequence[RunCompletedHook] = field(default_factory=tuple)
    kg_prompt_hints: Optional[KgPromptHints] = None
    adfree_base: Optional[AdfreeBase] = None
    excise_ads: Optional[ExciseAds] = None
    variant_turns: Optional[VariantTurns] = None
    screenplay: Optional[Screenplay] = None
    speaker_turns: Optional[SpeakerTurns] = None
    kg_graph_passes: Sequence[KgGraphPass] = field(default_factory=tuple)
    split_person_list: Optional[SplitPersonList] = None


def _all() -> Iterator[PipelineHooks]:
    from podcast_scraper.extensions import load_extensions

    from .builtin_stage_hooks import BUILTIN

    yield BUILTIN
    for ext in load_extensions():
        if ext.pipeline_hooks is not None:
            yield ext.pipeline_hooks


def run_started(cfg: Any) -> None:
    for hooks in _all():
        for hook in hooks.run_started:
            hook(cfg)


def corpus_derived(corpus_root: Path) -> None:
    for hooks in _all():
        for hook in hooks.corpus_derived:
            hook(corpus_root)


def batch_finalized(corpus_parent: Path) -> None:
    for hooks in _all():
        for hook in hooks.batch_finalized:
            hook(corpus_parent)


def run_finalized(corpus_root: str, run_dir: str) -> None:
    for hooks in _all():
        for hook in hooks.run_finalized:
            hook(corpus_root, run_dir)


def run_completed(cfg: Any, output_dir: str) -> None:
    for hooks in _all():
        for hook in hooks.run_completed:
            hook(cfg, output_dir)


def kg_prompt_hints(
    transcript_text: str, cfg: Any, kg_extraction_provider: Any, max_entities: int
) -> Tuple[List[Dict[str, str]], str]:
    for hooks in _all():
        if hooks.kg_prompt_hints is not None:
            return hooks.kg_prompt_hints(transcript_text, cfg, kg_extraction_provider, max_entities)
    return [], "v4"


def adfree_base(
    text: str,
    segments: Optional[List[Dict[str, Any]]],
    rel_transcript_path: str,
    output_dir: str,
    *,
    extra_cue_patterns: Optional[List[str]] = None,
) -> Optional[Tuple[str, Any]]:
    for hooks in _all():
        if hooks.adfree_base is not None:
            return hooks.adfree_base(
                text,
                segments,
                rel_transcript_path,
                output_dir,
                extra_cue_patterns=extra_cue_patterns,
            )
    return None


def excise_ads(
    text: str, *, segments: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, Optional[List[Dict[str, Any]]], Any]:
    for hooks in _all():
        if hooks.excise_ads is not None:
            return hooks.excise_ads(text, segments=segments)
    return text, segments, NoAdExcision(source_length=len(text))


def write_variant_turns(
    cfg: Any,
    text: str,
    segments: List[Dict[str, Any]],
    rel_transcript_path: str,
    output_dir: str,
    *,
    adfree: Any = None,
    episode: Any = None,
) -> None:
    for hooks in _all():
        if hooks.variant_turns is not None:
            hooks.variant_turns(
                cfg, text, segments, rel_transcript_path, output_dir, adfree=adfree, episode=episode
            )
            return


def screenplay(segments: List[Dict[str, Any]]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    for hooks in _all():
        if hooks.screenplay is not None:
            return hooks.screenplay(segments)
    return None


def speaker_turns(
    offset_segments: List[Dict[str, Any]], *, screenplay_text: str, max_sentence_words: int
) -> Any:
    for hooks in _all():
        if hooks.speaker_turns is not None:
            return hooks.speaker_turns(
                offset_segments,
                screenplay_text=screenplay_text,
                max_sentence_words=max_sentence_words,
            )
    raise RuntimeError("no speaker stage builds turns")


def split_person_list(text: str) -> List[str]:
    for hooks in _all():
        if hooks.split_person_list is not None:
            return hooks.split_person_list(text)
    return [text]


def kg_graph_passes(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    for hooks in _all():
        for graph_pass in hooks.kg_graph_passes:
            nodes, edges = graph_pass(nodes, edges)
    return nodes, edges


__all__ = [
    "AdfreeBase",
    "ExciseAds",
    "NoAdExcision",
    "BatchFinalizedHook",
    "CorpusDerivedHook",
    "KgGraphPass",
    "KgPromptHints",
    "PipelineHooks",
    "Screenplay",
    "SpeakerTurns",
    "SplitPersonList",
    "VariantTurns",
    "RunCompletedHook",
    "RunFinalizedHook",
    "RunStartedHook",
    "adfree_base",
    "batch_finalized",
    "corpus_derived",
    "excise_ads",
    "kg_graph_passes",
    "kg_prompt_hints",
    "run_completed",
    "run_finalized",
    "run_started",
    "screenplay",
    "speaker_turns",
    "split_person_list",
    "write_variant_turns",
]
