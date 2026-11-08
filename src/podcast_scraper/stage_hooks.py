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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

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
#: ``(episode_root, metadata_path, doc, *, scope_tag, episode_id, raw_feed_id, published) ->
#: [(doc_id, text, metadata)]``: search-index rows a stage adds for its own artifacts.
IndexRowsHook = Callable[..., List[Tuple[str, str, Dict[str, Any]]]]
#: ``(episode_root, metadata_path, doc) -> [path]``: a stage's artifacts the search index is built
#: from, so a newer one makes the index stale.
IndexSourcePathsHook = Callable[[Path, Path, Dict[str, Any]], List[Path]]
#: ``(kind, suffix)``: a per-episode artifact kind a stage writes (``("gi", ".gi.json")``).
ArtifactKind = Tuple[str, str]
#: ``(args, log) -> exit code``: the enrich-edges pass over a corpus (``args`` as the CLI parses).
EnrichEdges = Callable[[Any, Any], int]
#: ``(episode_root, metadata_path, doc) -> (quote_text_by_insight, quote_times_by_insight)``.
InsightGrounding = Callable[
    [Path, Path, Dict[str, Any]],
    Tuple[Dict[str, str], Dict[str, Tuple[float, Optional[float]]]],
]
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
#: ``(cfg, feed, episodes, pipeline_metrics, *, speaker_detector) -> HostDetectionResult``: the
#: feed's hosts and the patterns per-episode detection reads. One owner; without one, no hosts.
FeedHosts = Callable[..., Any]
#: ``(host_detection_result, episode, episode_people) -> set of names``: the hosts one episode
#: is anchored on. One owner; without one, none.
EpisodeHosts = Callable[..., Set[str]]
#: ``(episode, cfg, host_detection_result, pipeline_metrics, *, skip_speaker_detection,
#: skip_reason, skip_detail) -> DetectedSpeakers | None``: the episode's guests and every stated
#: name. One owner; without one, ``None`` (nothing detected).
EpisodeSpeakers = Callable[..., Any]
#: ``(title, description, language, authors) -> [(name, source), ...]``: the hosts a feed
#: states, in order, each with the field that states it (``feed_statement``,
#: ``feed_statement_refused``, ``author_tag``). One owner; without one, none.
StatedShowHosts = Callable[[Any, Any, str, List[str]], List[Tuple[str, str]]]
#: ``(a, b) -> bool``: whether two spellings name one person.
SamePerson = Callable[[str, str], bool]
#: ``() -> SamePerson``: the person match, built once so a caller comparing names in a loop
#: pays no lookup per pair. One owner; without one, only an identical string matches.
PersonMatcher = Callable[[], SamePerson]
#: ``(mode, job, cfg, run_suffix, output_dir, transcription_provider, pipeline_metrics) ->
#: (ok, transcript_relpath, bytes)``: a speaker reprocess of an episode already on disk, ``mode``
#: ``relabel`` (frozen diarization, names re-resolved) or ``rediarize`` (voices from the audio
#: again). One owner; without one :func:`speaker_reprocess` returns ``None`` and the caller
#: records the refusal.
SpeakerReprocess = Callable[..., Tuple[bool, Optional[str], int]]
#: ``(result, media, cfg, job, output_dir, pipeline_metrics, episode_duration_seconds,
#: transcription_provider) -> (result, diarization_s, naming_s, diarization_error)``: the speaker
#: stage on a fresh ASR result. One owner; without one the result is kept as it came (no
#: speakers) and nothing is timed.
SpeakersAfterAsr = Callable[..., Tuple[Any, Optional[float], Optional[float], Optional[str]]]
#: ``(segments, rel_transcript_path, output_dir) -> relpath | None``: the pre-naming render,
#: ``<base>.anon.txt`` (D-40). One owner; without one none is written.
AnonTranscript = Callable[[List[Dict[str, Any]], str, str], Optional[str]]
#: ``(segments) -> text | None``: roster-labelled segments rendered as speaker lines. One owner;
#: without one, ``None`` (the caller keeps the plain transcript).
RenderSpeakers = Callable[[List[Dict[str, Any]]], Optional[str]]
#: ``(label) -> bool``: whether a transcript's voice label names a person. One owner; without one
#: no label does.
PersonVoiceLabel = Callable[[str], bool]
#: ``(result, cfg, **context) -> result``: the roster pass over a transcript that names its own
#: speakers (a publisher's ``<v Name>`` cues). One owner; without one the result is kept as it came.
NativeRoster = Callable[..., Dict[str, Any]]
#: ``(new_segments, old_segments) -> new_segments``: each new segment (one per input, in order)
#: takes the speaker identity of the old segments it overlaps. One owner; without one the new
#: segments come back as copies, unlabelled.
CarrySpeakerIdentity = Callable[[Sequence[Any], Sequence[Any]], List[Dict[str, Any]]]
#: ``(**context) -> (bridge_gi_payload, gi_artifact_path, gi_cost, gi_elapsed, gi_meta, gi_path)``:
#: the episode's grounded insights (``gi.json``), built and written. One owner; without one,
#: ``None``: no GI, and the metadata record carries none.
EpisodeInsights = Callable[..., Tuple[Any, ...]]
#: ``(**context) -> (bridge_gi_payload, bridge_kg_payload)``: once both layers exist, the passes
#: that join them (KG organisations, person-id scoping and merges, typed mentions, topic
#: alignment), then the bridge and the context digest. One owner; without one, ``None``: the
#: payloads are kept as they are and no bridge is written.
GiKgPasses = Callable[..., Tuple[Any, ...]]
#: ``(output_dir, transcript_file_path, detected_hosts, detected_guests, feed_title, language) ->
#: (speakers, num_speakers)``: the episode's speaker record (``content.speakers``). One owner;
#: without one the record is empty.
SpeakerRecord = Callable[..., Tuple[List[Any], Optional[int]]]
#: ``(speakers, feed_title, language) -> (hosts, guests)``: the people a voice was matched to, for
#: the graph. One owner; without one, none.
SpeakerListsForGraph = Callable[..., Tuple[List[str], List[str]]]
#: Where a speaker record came from: resolved from the audio, or the pre-diarization guess.
RosterSourceValue = Literal["diarized", "hint"]
#: ``(speakers, *, diarized) -> RosterSourceValue | None``: where the record came from. One owner;
#: without one, ``None`` (no roster to claim an origin for).
RosterSourceHook = Callable[..., Optional[RosterSourceValue]]


class BundledNaming(Protocol):
    """Transcript naming inside one bundled summary call."""

    def params(self) -> Dict[str, Any]:
        """What to add to the call's params (the speaker labels to judge)."""

    def record(self, call: str, speakers: Any) -> None:
        """Write the call's answer into the speaker sidecar. Never raises."""

    def take_speakers(self, result: Any) -> Any:
        """Pop the speakers answer from a bundled result, in place."""


#: ``(cfg, pipeline_mode, output_dir, transcript_file_path) -> BundledNaming | None``: a naming
#: session for this bundled call, or None when there is nothing to judge. One owner; without one,
#: None (no prompt changes).
BundledNamingHook = Callable[..., Optional[BundledNaming]]
#: ``() -> names``: the names the speaker detector writes when it fails. One owner; without one,
#: none.
DefaultSpeakerNames = Callable[[], Sequence[str]]
#: ``(cfg) -> nlp | None``: the configured spaCy NER model. One owner; without one, None.
NerModel = Callable[[Any], Any]
#: ``(cfg) -> detector``: the speaker detector the run uses for hosts and guests. One owner;
#: without one, None (no detector).
CreateSpeakerDetector = Callable[[Any], Any]
#: ``(profile_id) -> None``: raise ValueError when a labeling profile id is not registered. One
#: owner; without one every id is accepted (no speaker stage reads it).
ValidateLabelingProfile = Callable[[str], None]
#: ``(corpus_parent, cfg, log)``: once a multi-feed batch with results is finalized, from the CLI
#: (the identity ledger refresh); run by :func:`podcast_scraper.cli_hooks.batch_completed`.
BatchCompletedHook = Callable[[Any, Any, Any], Any]
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
    index_rows: Sequence[IndexRowsHook] = field(default_factory=tuple)
    insight_grounding: Optional[InsightGrounding] = None
    enrich_edges: Optional[EnrichEdges] = None
    artifact_kinds: Sequence[ArtifactKind] = field(default_factory=tuple)
    index_source_paths: Sequence[IndexSourcePathsHook] = field(default_factory=tuple)
    run_completed: Sequence[RunCompletedHook] = field(default_factory=tuple)
    kg_prompt_hints: Optional[KgPromptHints] = None
    adfree_base: Optional[AdfreeBase] = None
    excise_ads: Optional[ExciseAds] = None
    variant_turns: Optional[VariantTurns] = None
    screenplay: Optional[Screenplay] = None
    speaker_turns: Optional[SpeakerTurns] = None
    feed_hosts: Optional[FeedHosts] = None
    episode_hosts: Optional[EpisodeHosts] = None
    episode_speakers: Optional[EpisodeSpeakers] = None
    stated_show_hosts: Optional[StatedShowHosts] = None
    person_matcher: Optional[PersonMatcher] = None
    speaker_reprocess: Optional[SpeakerReprocess] = None
    speakers_after_asr: Optional[SpeakersAfterAsr] = None
    anon_transcript: Optional[AnonTranscript] = None
    render_speakers: Optional[RenderSpeakers] = None
    person_voice_label: Optional[PersonVoiceLabel] = None
    native_roster: Optional[NativeRoster] = None
    carry_speaker_identity: Optional[CarrySpeakerIdentity] = None
    episode_insights: Optional[EpisodeInsights] = None
    gi_kg_passes: Optional[GiKgPasses] = None
    speaker_record: Optional[SpeakerRecord] = None
    speaker_lists_for_graph: Optional[SpeakerListsForGraph] = None
    roster_source: Optional[RosterSourceHook] = None
    bundled_naming: Optional[BundledNamingHook] = None
    default_speaker_names: Optional[DefaultSpeakerNames] = None
    ner_model: Optional[NerModel] = None
    create_speaker_detector: Optional[CreateSpeakerDetector] = None
    validate_labeling_profile: Optional[ValidateLabelingProfile] = None
    batch_completed: Sequence[BatchCompletedHook] = field(default_factory=tuple)
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
    """Run every ``run_started`` stage hook: the built-in ones, then each extension's, in load
    order."""
    for hooks in _all():
        for hook in hooks.run_started:
            hook(cfg)


def corpus_derived(corpus_root: Path) -> None:
    """Run every ``corpus_derived`` stage hook: the built-in ones, then each extension's, in load
    order."""
    for hooks in _all():
        for hook in hooks.corpus_derived:
            hook(corpus_root)


def batch_finalized(corpus_parent: Path) -> None:
    """Run every ``batch_finalized`` stage hook: the built-in ones, then each extension's, in
    load order."""
    for hooks in _all():
        for hook in hooks.batch_finalized:
            hook(corpus_parent)


def index_rows(
    episode_root: Path,
    metadata_path: Path,
    doc: Dict[str, Any],
    *,
    scope_tag: str,
    episode_id: str,
    raw_feed_id: Any,
    published: Any,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Every ``index_rows`` stage hook's rows, the built-in ones first, then each extension's, in
    load order; without a stage that adds rows, none (the index holds public's own)."""
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    for hooks in _all():
        for hook in hooks.index_rows:
            rows.extend(
                hook(
                    episode_root,
                    metadata_path,
                    doc,
                    scope_tag=scope_tag,
                    episode_id=episode_id,
                    raw_feed_id=raw_feed_id,
                    published=published,
                )
            )
    return rows


def index_source_paths(episode_root: Path, metadata_path: Path, doc: Dict[str, Any]) -> List[Path]:
    """Every ``index_source_paths`` stage hook's paths, built-in first, then each extension's;
    without one, none (staleness follows public's own artifacts)."""
    out: List[Path] = []
    for hooks in _all():
        for hook in hooks.index_source_paths:
            out.extend(hook(episode_root, metadata_path, doc))
    return out


def artifact_kinds() -> List[ArtifactKind]:
    """Every stage's per-episode artifact kinds, built-in first, then each extension's; without a
    stage, none (corpus readers see public's own kinds only)."""
    return [kind for hooks in _all() for kind in hooks.artifact_kinds]


def enrich_edges(args: Any, log: Any) -> Optional[int]:
    """The ``enrich_edges`` stage hook's exit code: the first owner that sets it answers;
    without one, ``None`` (there is no ``gi.json`` to enrich)."""
    for hooks in _all():
        if hooks.enrich_edges is not None:
            return hooks.enrich_edges(args, log)
    return None


def insight_grounding(
    episode_root: Path, metadata_path: Path, doc: Dict[str, Any]
) -> Tuple[Dict[str, str], Dict[str, Tuple[float, Optional[float]]]]:
    """Each insight's grounding quote, ``(text by insight id, (start_s, end_s) by insight id)``:
    the first owner that sets it answers; without one, none (no insight is linked)."""
    for hooks in _all():
        if hooks.insight_grounding is not None:
            return hooks.insight_grounding(episode_root, metadata_path, doc)
    return {}, {}


def run_finalized(corpus_root: str, run_dir: str) -> None:
    """Run every ``run_finalized`` stage hook: the built-in ones, then each extension's, in load
    order."""
    for hooks in _all():
        for hook in hooks.run_finalized:
            hook(corpus_root, run_dir)


def run_completed(cfg: Any, output_dir: str) -> None:
    """Run every ``run_completed`` stage hook: the built-in ones, then each extension's, in load
    order."""
    for hooks in _all():
        for hook in hooks.run_completed:
            hook(cfg, output_dir)


def kg_prompt_hints(
    transcript_text: str, cfg: Any, kg_extraction_provider: Any, max_entities: int
) -> Tuple[List[Dict[str, str]], str]:
    """The ``kg_prompt_hints`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``([], 'v4')``."""
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
    """The ``adfree_base`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
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
    """The ``excise_ads`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``(text, segments,
    NoAdExcision(source_length=len(text)))``."""
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
    """Run every ``write_variant_turns`` stage hook: the built-in ones, then each extension's, in
    load order."""
    for hooks in _all():
        if hooks.variant_turns is not None:
            hooks.variant_turns(
                cfg, text, segments, rel_transcript_path, output_dir, adfree=adfree, episode=episode
            )
            return


def screenplay(segments: List[Dict[str, Any]]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """The ``screenplay`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.screenplay is not None:
            return hooks.screenplay(segments)
    return None


def speaker_turns(
    offset_segments: List[Dict[str, Any]], *, screenplay_text: str, max_sentence_words: int
) -> Any:
    """The ``speaker_turns`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.speaker_turns is not None:
            return hooks.speaker_turns(
                offset_segments,
                screenplay_text=screenplay_text,
                max_sentence_words=max_sentence_words,
            )
    raise RuntimeError("no speaker stage builds turns")


def split_person_list(text: str) -> List[str]:
    """The ``split_person_list`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``[text]``."""
    for hooks in _all():
        if hooks.split_person_list is not None:
            return hooks.split_person_list(text)
    return [text]


def feed_hosts(
    cfg: Any,
    feed: Any,
    episodes: List[Any],
    pipeline_metrics: Any = None,
    *,
    speaker_detector: Any = None,
) -> Any:
    """The ``feed_hosts`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``HostDetectionResult(cached_hosts=set(),
    heuristics=None)``."""
    for hooks in _all():
        if hooks.feed_hosts is not None:
            return hooks.feed_hosts(
                cfg, feed, episodes, pipeline_metrics, speaker_detector=speaker_detector
            )
    from podcast_scraper.workflow.types import HostDetectionResult

    return HostDetectionResult(cached_hosts=set(), heuristics=None)


def episode_hosts(
    host_detection_result: Any, episode: Any, episode_people: Iterable[str] = ()
) -> Set[str]:
    """The ``episode_hosts`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``set()``."""
    for hooks in _all():
        if hooks.episode_hosts is not None:
            return hooks.episode_hosts(host_detection_result, episode, episode_people)
    return set()


def episode_speakers(
    episode: Any,
    cfg: Any,
    host_detection_result: Any,
    pipeline_metrics: Any,
    *,
    skip_speaker_detection: bool = False,
    skip_reason: Optional[str] = None,
    skip_detail: Optional[Dict[str, Any]] = None,
) -> Any:
    """The ``episode_speakers`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.episode_speakers is not None:
            return hooks.episode_speakers(
                episode,
                cfg,
                host_detection_result,
                pipeline_metrics,
                skip_speaker_detection=skip_speaker_detection,
                skip_reason=skip_reason,
                skip_detail=skip_detail,
            )
    return None


def stated_show_hosts(
    title: Any, description: Any, language: str, authors: List[str]
) -> List[Tuple[str, str]]:
    """The ``stated_show_hosts`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``[]``."""
    for hooks in _all():
        if hooks.stated_show_hosts is not None:
            return hooks.stated_show_hosts(title, description, language, authors)
    return []


def person_matcher() -> SamePerson:
    """The owner's person match, resolved once: a caller comparing names in a loop calls this
    once and the returned function per pair."""
    for hooks in _all():
        if hooks.person_matcher is not None:
            return hooks.person_matcher()
    return _identical


def _identical(a: str, b: str) -> bool:
    return a == b


def same_person(a: str, b: str) -> bool:
    """Run every ``same_person`` stage hook: the built-in ones, then each extension's, in load
    order."""
    return person_matcher()(a, b)


def speaker_reprocess(
    mode: str,
    job: Any,
    cfg: Any,
    run_suffix: Optional[str],
    output_dir: str,
    transcription_provider: Any,
    pipeline_metrics: Any,
) -> Optional[Tuple[bool, Optional[str], int]]:
    """The ``speaker_reprocess`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.speaker_reprocess is not None:
            return hooks.speaker_reprocess(
                mode, job, cfg, run_suffix, output_dir, transcription_provider, pipeline_metrics
            )
    return None


def speakers_after_asr(
    result: Any,
    media_for_transcription: Any,
    cfg: Any,
    job: Any,
    output_dir: str,
    pipeline_metrics: Any,
    episode_duration_seconds: Any,
    transcription_provider: Any,
) -> Tuple[Any, Optional[float], Optional[float], Optional[str]]:
    """The ``speakers_after_asr`` stage hook: the first owner that sets it answers (built-in,
    then extensions in load order); without one, ``(result, None, None, None)``."""
    for hooks in _all():
        if hooks.speakers_after_asr is not None:
            return hooks.speakers_after_asr(
                result,
                media_for_transcription,
                cfg,
                job,
                output_dir,
                pipeline_metrics,
                episode_duration_seconds,
                transcription_provider,
            )
    return result, None, None, None


def anon_transcript(
    segments: List[Dict[str, Any]], rel_transcript_path: str, output_dir: str
) -> Optional[str]:
    """The ``anon_transcript`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.anon_transcript is not None:
            return hooks.anon_transcript(segments, rel_transcript_path, output_dir)
    return None


def render_speakers(segments: List[Dict[str, Any]]) -> Optional[str]:
    """The ``render_speakers`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.render_speakers is not None:
            return hooks.render_speakers(segments)
    return None


def person_voice_label(label: str) -> bool:
    """The ``person_voice_label`` stage hook: the first owner that sets it answers (built-in,
    then extensions in load order); without one, ``False``."""
    for hooks in _all():
        if hooks.person_voice_label is not None:
            return hooks.person_voice_label(label)
    return False


def native_roster(result: Dict[str, Any], cfg: Any, **context: Any) -> Dict[str, Any]:
    """The ``native_roster`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``result``."""
    for hooks in _all():
        if hooks.native_roster is not None:
            return hooks.native_roster(result, cfg, **context)
    return result


def carry_speaker_identity(new: Sequence[Any], old: Sequence[Any]) -> List[Dict[str, Any]]:
    """The ``carry_speaker_identity`` stage hook: the first owner that sets it answers (built-in,
    then extensions in load order); without one, ``[dict(seg) for seg in new]``."""
    for hooks in _all():
        if hooks.carry_speaker_identity is not None:
            return hooks.carry_speaker_identity(new, old)
    return [dict(seg) for seg in new]


def episode_insights(**context: Any) -> Optional[Tuple[Any, ...]]:
    """The ``episode_insights`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.episode_insights is not None:
            return hooks.episode_insights(**context)
    return None


def gi_kg_passes(**context: Any) -> Optional[Tuple[Any, ...]]:
    """The ``gi_kg_passes`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.gi_kg_passes is not None:
            return hooks.gi_kg_passes(**context)
    return None


def speaker_record(
    output_dir: Optional[str],
    transcript_file_path: Optional[str],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    feed_title: Optional[str],
    language: Optional[str] = None,
) -> Tuple[List[Any], Optional[int]]:
    """The ``speaker_record`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``([], None)``."""
    for hooks in _all():
        if hooks.speaker_record is not None:
            return hooks.speaker_record(
                output_dir,
                transcript_file_path,
                detected_hosts,
                detected_guests,
                feed_title,
                language=language,
            )
    return [], None


def speaker_lists_for_graph(
    speakers: Any, feed_title: Optional[str] = None, language: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """The ``speaker_lists_for_graph`` stage hook: the first owner that sets it answers
    (built-in, then extensions in load order); without one, ``([], [])``."""
    for hooks in _all():
        if hooks.speaker_lists_for_graph is not None:
            return hooks.speaker_lists_for_graph(speakers, feed_title=feed_title, language=language)
    return [], []


def roster_source(speakers: Any, *, diarized: bool) -> Optional[RosterSourceValue]:
    """The ``roster_source`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.roster_source is not None:
            return hooks.roster_source(speakers, diarized=diarized)
    return None


def bundled_naming(
    cfg: Any, pipeline_mode: str, output_dir: str, transcript_file_path: Optional[str]
) -> Optional[BundledNaming]:
    """The ``bundled_naming`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.bundled_naming is not None:
            return hooks.bundled_naming(cfg, pipeline_mode, output_dir, transcript_file_path)
    return None


def default_speaker_names() -> Sequence[str]:
    """The ``default_speaker_names`` stage hook: the first owner that sets it answers (built-in,
    then extensions in load order); without one, ``()``."""
    for hooks in _all():
        if hooks.default_speaker_names is not None:
            return hooks.default_speaker_names()
    return ()


def ner_model(cfg: Any) -> Any:
    """The ``ner_model`` stage hook: the first owner that sets it answers (built-in, then
    extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.ner_model is not None:
            return hooks.ner_model(cfg)
    return None


def create_speaker_detector(cfg: Any) -> Any:
    """The ``create_speaker_detector`` stage hook: the first owner that sets it answers
    (built-in, then extensions in load order); without one, ``None``."""
    for hooks in _all():
        if hooks.create_speaker_detector is not None:
            return hooks.create_speaker_detector(cfg)
    return None


def validate_labeling_profile(profile_id: str) -> None:
    """Run every ``validate_labeling_profile`` stage hook: the built-in ones, then each
    extension's, in load order."""
    for hooks in _all():
        if hooks.validate_labeling_profile is not None:
            hooks.validate_labeling_profile(profile_id)
            return


def kg_graph_passes(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run every ``kg_graph_passes`` stage hook: the built-in ones, then each extension's, in
    load order."""
    for hooks in _all():
        for graph_pass in hooks.kg_graph_passes:
            nodes, edges = graph_pass(nodes, edges)
    return nodes, edges


__all__ = [
    "AnonTranscript",
    "AdfreeBase",
    "EpisodeHosts",
    "EpisodeInsights",
    "EpisodeSpeakers",
    "ExciseAds",
    "FeedHosts",
    "GiKgPasses",
    "NoAdExcision",
    "BatchCompletedHook",
    "BatchFinalizedHook",
    "BundledNaming",
    "BundledNamingHook",
    "CarrySpeakerIdentity",
    "CorpusDerivedHook",
    "CreateSpeakerDetector",
    "DefaultSpeakerNames",
    "KgGraphPass",
    "KgPromptHints",
    "NativeRoster",
    "NerModel",
    "PersonMatcher",
    "PersonVoiceLabel",
    "PipelineHooks",
    "RenderSpeakers",
    "RosterSourceHook",
    "RosterSourceValue",
    "SamePerson",
    "Screenplay",
    "SpeakerTurns",
    "SpeakerListsForGraph",
    "SpeakerRecord",
    "SpeakerReprocess",
    "ValidateLabelingProfile",
    "SpeakersAfterAsr",
    "SplitPersonList",
    "StatedShowHosts",
    "VariantTurns",
    "RunCompletedHook",
    "RunFinalizedHook",
    "RunStartedHook",
    "adfree_base",
    "anon_transcript",
    "artifact_kinds",
    "batch_finalized",
    "enrich_edges",
    "index_rows",
    "index_source_paths",
    "insight_grounding",
    "bundled_naming",
    "carry_speaker_identity",
    "corpus_derived",
    "create_speaker_detector",
    "default_speaker_names",
    "episode_hosts",
    "episode_insights",
    "episode_speakers",
    "excise_ads",
    "feed_hosts",
    "gi_kg_passes",
    "kg_graph_passes",
    "kg_prompt_hints",
    "native_roster",
    "ner_model",
    "person_voice_label",
    "render_speakers",
    "roster_source",
    "run_completed",
    "run_finalized",
    "run_started",
    "person_matcher",
    "same_person",
    "screenplay",
    "speaker_lists_for_graph",
    "speaker_record",
    "speaker_reprocess",
    "speaker_turns",
    "speakers_after_asr",
    "split_person_list",
    "validate_labeling_profile",
    "stated_show_hosts",
    "write_variant_turns",
]
