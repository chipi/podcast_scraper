"""The pipeline's stage hooks (E04 S2): the run and the batch reach stages they do not own only
through these points, built-in hooks first, then each extension's, with no guard added."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import patch

import pytest

from podcast_scraper import builtin_stage_hooks, stage_hooks
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.stage_hooks import PipelineHooks

pytestmark = pytest.mark.unit

_WORKFLOW = Path(stage_hooks.__file__).parent / "workflow"


def _recording(calls: List[Tuple[str, Tuple[Any, ...]]], who: str) -> PipelineHooks:
    def rec(point: str) -> Any:
        def hook(*args: Any) -> None:
            calls.append((f"{who}.{point}", args))

        return hook

    return PipelineHooks(
        name=who,
        run_started=(rec("run_started"),),
        corpus_derived=(rec("corpus_derived"),),
        batch_finalized=(rec("batch_finalized"),),
        run_finalized=(rec("run_finalized"),),
        run_completed=(rec("run_completed"),),
    )


def test_built_in_hooks_run_before_each_extension_in_load_order(tmp_path: Path) -> None:
    calls: List[Tuple[str, Tuple[Any, ...]]] = []
    exts = [
        Extension(name="a", pipeline_hooks=_recording(calls, "a")),
        Extension(name="none"),
        Extension(name="b", pipeline_hooks=_recording(calls, "b")),
    ]
    with (
        patch.object(builtin_stage_hooks, "BUILTIN", _recording(calls, "builtin")),
        use_extensions(exts),
    ):
        stage_hooks.run_started("cfg")
        stage_hooks.corpus_derived(tmp_path)
        stage_hooks.batch_finalized(tmp_path)
        stage_hooks.run_finalized("root", "run")
        stage_hooks.run_completed("cfg", "out")

    assert calls == [
        ("builtin.run_started", ("cfg",)),
        ("a.run_started", ("cfg",)),
        ("b.run_started", ("cfg",)),
        ("builtin.corpus_derived", (tmp_path,)),
        ("a.corpus_derived", (tmp_path,)),
        ("b.corpus_derived", (tmp_path,)),
        ("builtin.batch_finalized", (tmp_path,)),
        ("a.batch_finalized", (tmp_path,)),
        ("b.batch_finalized", (tmp_path,)),
        ("builtin.run_finalized", ("root", "run")),
        ("a.run_finalized", ("root", "run")),
        ("b.run_finalized", ("root", "run")),
        ("builtin.run_completed", ("cfg", "out")),
        ("a.run_completed", ("cfg", "out")),
        ("b.run_completed", ("cfg", "out")),
    ]


def test_a_raising_hook_stops_the_point_as_it_did_inline() -> None:
    """The runner adds no guard: a start-up check that raised before the seam still stops the run,
    and the hooks after it do not run."""
    later: List[str] = []

    def missing_dependency(cfg: Any) -> None:
        raise ImportError("grounding needs sentence-transformers")

    builtin = PipelineHooks(name="builtin", run_started=(missing_dependency,))
    ext = Extension(name="x", pipeline_hooks=PipelineHooks(name="x", run_started=(later.append,)))
    with patch.object(builtin_stage_hooks, "BUILTIN", builtin), use_extensions([ext]):
        with pytest.raises(ImportError, match="sentence-transformers"):
            stage_hooks.run_started("cfg")
    assert later == []


def test_the_built_in_hooks_call_todays_stages(tmp_path: Path) -> None:
    with (
        patch("podcast_scraper.gi.deps.validate_gil_grounding_dependencies") as deps,
        patch("podcast_scraper.providers.ml.diarization.ad_signatures.write_for_corpus") as ads,
        patch("podcast_scraper.quality.naming_coverage.measure_and_record") as coverage,
        patch("podcast_scraper.audio_sync.references.write_for_run") as refs,
        use_extensions([]),
    ):
        stage_hooks.run_started("cfg")
        stage_hooks.corpus_derived(tmp_path)
        stage_hooks.batch_finalized(tmp_path)
        stage_hooks.run_finalized("root", "run")
    deps.assert_called_once_with("cfg")
    ads.assert_called_once_with(tmp_path)
    coverage.assert_called_once_with(tmp_path)
    refs.assert_called_once_with("root", "run")


def test_a_failing_audio_reference_pass_never_fails_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("podcast_scraper.audio_sync.references.write_for_run", side_effect=OSError("disk")),
        use_extensions([]),
    ):
        stage_hooks.run_finalized("root", "run")
    assert "audio references: end-of-run pass failed (non-fatal)" in caplog.text


@pytest.mark.parametrize(
    "module, moved",
    [
        (
            "orchestration.py",
            {
                "gi.deps",
                "audio_sync.references",
                "diarization.ad_signatures",
                "kg.topic_clustering",
            },
        ),
        ("corpus_operations.py", {"quality.naming_coverage", "diarization.ad_signatures"}),
    ],
)
def test_the_run_and_the_batch_no_longer_import_the_hooked_stages(module: str, moved: set) -> None:
    tree = ast.parse((_WORKFLOW / module).read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported for s in moved if m.endswith(s)}


class _Cfg:
    def __init__(self, on: bool, dry_run: bool = False) -> None:
        self.kg_topic_corpus_clustering = on
        self.dry_run = dry_run


@pytest.mark.parametrize("cfg", [_Cfg(False), _Cfg(True, dry_run=True)])
def test_corpus_topic_clustering_runs_only_when_on_and_not_a_dry_run(cfg: _Cfg) -> None:
    with (
        patch("podcast_scraper.kg.topic_clustering.cluster_and_apply_corpus_topics") as cluster,
        use_extensions([]),
    ):
        stage_hooks.run_completed(cfg, "out")
    cluster.assert_not_called()


def test_corpus_topic_clustering_runs_and_a_failure_is_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("podcast_scraper.kg.topic_clustering.cluster_and_apply_corpus_topics") as cluster,
        use_extensions([]),
    ):
        stage_hooks.run_completed(_Cfg(True), "out")
    cluster.assert_called_once_with(Path("out"))
    with (
        patch(
            "podcast_scraper.kg.topic_clustering.cluster_and_apply_corpus_topics",
            side_effect=RuntimeError("boom"),
        ),
        use_extensions([]),
    ):
        stage_hooks.run_completed(_Cfg(True), "out")
    assert "corpus topic clustering failed (non-fatal): boom" in caplog.text


def test_kg_prompt_hints_take_the_first_owner_and_default_to_plain_v4() -> None:
    def owner(answer: Tuple[List[Any], str]) -> Any:
        return lambda text, cfg, provider, max_entities: answer

    first = PipelineHooks(name="first", kg_prompt_hints=owner(([{"name": "A"}], "v5")))
    second = Extension(
        name="second",
        pipeline_hooks=PipelineHooks(name="second", kg_prompt_hints=owner(([], "never"))),
    )
    with patch.object(builtin_stage_hooks, "BUILTIN", first), use_extensions([second]):
        assert stage_hooks.kg_prompt_hints("t", None, None, 5) == ([{"name": "A"}], "v5")
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        assert stage_hooks.kg_prompt_hints("t", None, None, 5) == ([], "v4")


def test_kg_graph_passes_chain_each_output_into_the_next() -> None:
    def add(tag: str) -> Any:
        return lambda nodes, edges: (nodes + [{"id": tag}], edges + [{"from": tag}])

    builtin = PipelineHooks(name="b", kg_graph_passes=(add("b1"), add("b2")))
    ext = Extension(name="x", pipeline_hooks=PipelineHooks(name="x", kg_graph_passes=(add("x"),)))
    with patch.object(builtin_stage_hooks, "BUILTIN", builtin), use_extensions([ext]):
        nodes, edges = stage_hooks.kg_graph_passes([], [])
    assert [n["id"] for n in nodes] == ["b1", "b2", "x"]
    assert [e["from"] for e in edges] == ["b1", "b2", "x"]


def test_the_kg_pipeline_no_longer_imports_the_advanced_kg_passes() -> None:
    kg_pipeline = Path(stage_hooks.__file__).parent / "kg" / "pipeline.py"
    tree = ast.parse(kg_pipeline.read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {
        m
        for m in imported
        for s in ("ner_prepass", "speaker_coherence", "speaker_detection")
        if m.endswith(s)
    }


def test_without_an_ad_stage_no_ad_free_base_is_written_and_nothing_is_cut() -> None:
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        assert stage_hooks.adfree_base("text", [], "t.txt", "/out") is None
        segs = [{"text": "x"}]
        text, out_segs, meta = stage_hooks.excise_ads("sponsored by x", segments=segs)
    assert (text, out_segs) == ("sponsored by x", segs)
    assert meta == stage_hooks.NoAdExcision(source_length=len("sponsored by x"))
    assert meta.chars_removed == 0 and meta.preroll_cut_end is None


def test_the_built_in_ad_stage_calls_todays_code() -> None:
    with (
        patch(
            "podcast_scraper.workflow.adfree_transcript.produce_adfree_artifacts",
            return_value=("t.adfree.txt", "artifacts"),
        ) as produce,
        patch(
            "podcast_scraper.gi.ad_regions.excise_ad_regions", return_value=("cut", None, "meta")
        ) as excise,
        use_extensions([]),
    ):
        got = stage_hooks.adfree_base("text", [], "t.txt", "/out", extra_cue_patterns=["p"])
        cut = stage_hooks.excise_ads("text", segments=None)
    assert got == ("t.adfree.txt", "artifacts")
    produce.assert_called_once_with("text", [], "t.txt", "/out", extra_cue_patterns=["p"])
    assert cut == ("cut", None, "meta")
    excise.assert_called_once_with("text", segments=None)


@pytest.mark.parametrize(
    "module",
    [
        "workflow/episode_processor.py",
        "translation/artifacts.py",
        "cleaning/pattern_based.py",
        "workflow/metadata_generation.py",
    ],
)
def test_ad_cleaning_callers_no_longer_import_the_ad_stage(module: str) -> None:
    tree = ast.parse((Path(stage_hooks.__file__).parent / module).read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if m.endswith(("adfree_transcript", "gi.ad_regions"))}


def test_without_a_speaker_stage_there_are_no_turns_and_no_screenplay() -> None:
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        assert stage_hooks.screenplay([{"speaker": "A", "text": "hi"}]) is None
        stage_hooks.write_variant_turns(None, "hi", [], "t.txt", "/out")
        with pytest.raises(RuntimeError, match="no speaker stage"):
            stage_hooks.speaker_turns([], screenplay_text="", max_sentence_words=0)


def test_the_built_in_speaker_stage_renders_and_builds_turns_with_todays_code() -> None:
    with (
        patch(
            "podcast_scraper.providers.ml.diarization.formatting."
            "format_diarized_screenplay_with_offsets",
            return_value=("A: hi", [{"char_start": 0}]),
        ) as render,
        patch(
            "podcast_scraper.providers.ml.diarization.turns.build_turns", return_value="turns"
        ) as build,
        use_extensions([]),
    ):
        assert stage_hooks.screenplay([{"x": 1}]) == ("A: hi", [{"char_start": 0}])
        got = stage_hooks.speaker_turns([{"y": 2}], screenplay_text="A: hi", max_sentence_words=7)
    render.assert_called_once_with([{"x": 1}])
    build.assert_called_once_with([{"y": 2}], screenplay_text="A: hi", max_sentence_words=7)
    assert got == "turns"


@pytest.mark.parametrize(
    "module, moved",
    [
        ("workflow/episode_processor.py", ("turns_artifact", "diarization.turns")),
        ("workflow/translation_stage.py", ("diarization.turns", "diarization.formatting")),
        ("translation/artifacts.py", ("diarization.turns", "diarization.formatting")),
    ],
)
def test_turn_and_screenplay_callers_no_longer_import_them(
    module: str, moved: Tuple[str, ...]
) -> None:
    tree = ast.parse((Path(stage_hooks.__file__).parent / module).read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if m.endswith(moved)}


def test_without_a_speaker_stage_a_feed_has_no_hosts_and_an_episode_no_speakers() -> None:
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        result = stage_hooks.feed_hosts(object(), object(), [], None, speaker_detector=None)
        assert result.cached_hosts == set() and result.heuristics is None
        assert stage_hooks.episode_hosts(result, object(), ["Jane Roe"]) == set()
        assert stage_hooks.episode_speakers(object(), object(), result, None) is None


def test_the_built_in_speaker_stage_detects_with_todays_code() -> None:
    with (
        patch(
            "podcast_scraper.speaker_stage.detection.detect_feed_hosts_and_patterns",
            return_value="feed",
        ) as feed,
        patch(
            "podcast_scraper.speaker_stage.detection.hosts_for_episode", return_value={"A"}
        ) as hosts,
        patch(
            "podcast_scraper.speaker_stage.detection._detect_speakers_for_episode",
            return_value="speakers",
        ) as speakers,
        use_extensions([]),
    ):
        assert stage_hooks.feed_hosts("cfg", "f", [], "m", speaker_detector="d") == "feed"
        assert stage_hooks.episode_hosts("r", "e", ["p"]) == {"A"}
        got = stage_hooks.episode_speakers("e", "cfg", "r", "m", skip_reason="big")
    feed.assert_called_once_with("cfg", "f", [], "m", speaker_detector="d")
    hosts.assert_called_once_with("r", "e", ["p"])
    speakers.assert_called_once_with(
        "e",
        "cfg",
        "r",
        "m",
        skip_speaker_detection=False,
        skip_reason="big",
        skip_detail=None,
    )
    assert got == "speakers"


@pytest.mark.parametrize(
    "module",
    [
        "workflow/stages/processing.py",
        "workflow/stages/metadata.py",
        "workflow/stages/transcription.py",
    ],
)
def test_the_processing_stages_no_longer_import_host_or_guest_detection(module: str) -> None:
    tree = ast.parse((Path(stage_hooks.__file__).parent / module).read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {
        m
        for m in imported
        for s in ("speaker_detectors", "speaker_coherence", "diarization.roster")
        if s in m
    }


def test_without_a_speaker_stage_a_show_states_no_hosts_and_names_match_only_exactly() -> None:
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        assert stage_hooks.stated_show_hosts("T", "Hosted by Jane Roe", "en", ["Jane Roe"]) == []
        match = stage_hooks.person_matcher()
        assert match("Jane Roe", "Jane Roe") and not match("Joe Weisenthal", "Joe Wiesenthal")


def test_the_built_in_person_matcher_is_todays_same_person_itself() -> None:
    from podcast_scraper.kg.speaker_coherence import same_person

    with use_extensions([]):
        assert stage_hooks.person_matcher() is same_person
        assert stage_hooks.same_person("Joe Weisenthal", "Joe Wiesenthal")


def test_show_metadata_no_longer_imports_host_rules() -> None:
    src = (Path(stage_hooks.__file__).parent / "workflow" / "show_metadata.py").read_text(
        encoding="utf-8"
    )
    imported = {(n.module or "") for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if "speaker_detectors" in m or "speaker_coherence" in m}


@pytest.mark.parametrize("mode", ["relabel", "rediarize"])
def test_without_a_speaker_stage_a_reprocess_is_refused_and_counted(mode: str) -> None:
    from podcast_scraper.workflow import episode_processor as ep

    class _Job:
        idx = 7

    recorded: List[Tuple[Any, ...]] = []

    def _record(job: Any, cfg: Any, metrics: Any, stage: str, **kw: Any) -> None:
        recorded.append((stage, kw.get("error_type")))

    with (
        patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")),
        use_extensions([]),
        patch.object(ep, "_record_unresolved_transcript", _record),
    ):
        assert stage_hooks.speaker_reprocess(mode, _Job(), None, None, "/o", None, None) is None
        got = ep._speaker_reprocess(mode, _Job(), None, None, "/out", None, None)
    assert got == (False, None, 0)
    assert recorded == [(f"{mode}_only", "NoSpeakerStage")]


@pytest.mark.parametrize(
    "mode, fn",
    [("relabel", "_relabel_existing_transcript"), ("rediarize", "_rediarize_existing_transcript")],
)
def test_the_built_in_reprocess_runs_todays_stage_for_the_mode(mode: str, fn: str) -> None:
    with (
        patch(f"podcast_scraper.speaker_stage.reprocess.{fn}", return_value=(True, "r", 3)) as run,
        use_extensions([]),
    ):
        got = stage_hooks.speaker_reprocess(mode, "job", "cfg", "sfx", "/out", "prov", "m")
    run.assert_called_once_with("job", "cfg", "sfx", "/out", "prov", "m")
    assert got == (True, "r", 3)


def test_without_a_speaker_stage_the_asr_result_is_kept_as_it_came() -> None:
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}]}
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        got = stage_hooks.speakers_after_asr(result, "a.wav", None, None, "/o", None, 60.0, None)
    assert got == (result, None, None, None)
    assert got[0] is result


def test_the_built_in_speaker_stage_after_asr_gets_every_input() -> None:
    args = ("result", "a.wav", "cfg", "job", "/o", "metrics", 60.0, "provider")
    with (
        patch(
            "podcast_scraper.speaker_stage.after_asr.speakers_after_asr",
            return_value=("r2", 1.0, 0.5, None),
        ) as run,
        use_extensions([]),
    ):
        assert stage_hooks.speakers_after_asr(*args) == ("r2", 1.0, 0.5, None)
    run.assert_called_once_with(*args)


def test_transcription_no_longer_imports_the_diarization_pipeline() -> None:
    src = (Path(stage_hooks.__file__).parent / "workflow" / "episode_processor.py").read_text(
        encoding="utf-8"
    )
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "transcribe_media_to_text"
    )
    imported = {(n.module or "") for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if "diarization" in m}


def test_without_a_speaker_stage_no_anon_render_and_no_speaker_lines(tmp_path: Path) -> None:
    segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker_label": "SPEAKER_00"}]
    with patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")), use_extensions([]):
        assert stage_hooks.anon_transcript(segs, "t.txt", str(tmp_path)) is None
        assert stage_hooks.render_speakers(segs) is None
    assert list(tmp_path.iterdir()) == []


def test_the_built_in_speaker_stage_renders_and_writes_the_anon_transcript_with_todays_code() -> (
    None
):
    with (
        patch(
            "podcast_scraper.providers.ml.diarization.formatting."
            "format_diarized_screenplay_from_segments",
            return_value="A: hi",
        ) as render,
        patch(
            "podcast_scraper.speaker_stage.sidecars._write_anon_transcript",
            return_value="t.anon.txt",
        ) as anon,
        use_extensions([]),
    ):
        assert stage_hooks.render_speakers([{"x": 1}]) == "A: hi"
        assert stage_hooks.anon_transcript([{"x": 1}], "t.txt", "/o") == "t.anon.txt"
    render.assert_called_once_with([{"x": 1}])
    anon.assert_called_once_with([{"x": 1}], "t.txt", "/o")
