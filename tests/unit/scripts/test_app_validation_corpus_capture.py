"""`--capture-summaries` must capture the English render, or say it did not.

THE FAILURE THIS GUARDS WAS SILENT. After D-44 moved the English body to the canonical
``<stem>.txt`` and the source to ``<stem>.<lang>.txt``, `_load_pipeline_outputs` kept reading
``<stem>.en.txt``, found nothing, and `_capture_english_renders` skipped every episode as "not
translated" — while `--capture-summaries` reported success. It went unnoticed for a day.

So the run tree here is NOT laid out by hand. It is written by the pipeline's own D-44 code
(`_swap_in_translation`, `write_analysis_base`, `write_translation_json`), so the next layout
change moves this fixture with it, and a builder that has not followed goes red here instead of
capturing nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.translation import artifacts
from podcast_scraper.translation.artifacts import TranslationDocument
from podcast_scraper.workflow.transcript_resolution import _cleaned_transcript_relpath

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_app_validation_corpus.py"
_spec = importlib.util.spec_from_file_location("_bavc_capture", _SCRIPT)
assert _spec and _spec.loader
_bavc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bavc)

_SOURCE = "Hola, soy Lucía Herrera. Hoy hablamos de senderos.\n"
_ENGLISH = "Hi, I'm Lucía Herrera. Today we talk about trails.\n"
_TITLE_EN = "Building Trails That Last"


def _segments(text: str) -> list[dict[str, Any]]:
    return [{"start": 0.0, "end": 4.0, "text": text.strip(), "speaker": "SPEAKER_00"}]


def _metadata(guid: str, transcript_rel: str, language: str) -> dict[str, Any]:
    return {
        "episode": {"guid": guid, "duration_seconds": 361.0, "language": language},
        "feed": {"language": language},
        "content": {"transcript_file_path": transcript_rel},
        "summary": {"title": "T", "raw_text": "A summary.", "bullets": ["b1"]},
    }


def _kg(topics: list[str]) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": t, "type": "Topic", "properties": {"label": t.split(":", 1)[1]}} for t in topics
        ]
        + [{"id": "person:lucia-herrera", "type": "Person", "properties": {}}]
    }


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """One translated Spanish episode and one English episode, as a real run leaves them."""
    root = tmp_path / "run"
    run_dir = root / "feeds" / "p10" / "run_20261003"
    (run_dir / "transcripts").mkdir(parents=True)
    (run_dir / "metadata").mkdir()
    out = str(run_dir)

    # --- p10_e02: Spanish, translated. The source is written where ASR writes it — the
    # canonical path — and the pipeline's own swap then moves it aside.
    es_rel = "transcripts/p10_e02.txt"
    (run_dir / es_rel).write_text(_SOURCE, encoding="utf-8")
    (run_dir / "transcripts/p10_e02.segments.json").write_text(
        json.dumps(_segments(_SOURCE)), encoding="utf-8"
    )
    assert artifacts._swap_in_translation(
        es_rel,
        out,
        "es-ES",
        target_text=_ENGLISH,
        target_segments=_segments(_ENGLISH),
        unit_count=1,
    )
    assert artifacts.write_analysis_base(es_rel, out)
    assert artifacts.write_translation_json(
        TranslationDocument(source_language="es", title_en=_TITLE_EN), es_rel, out
    )
    # The summary stage writes the cleaned body inline (no writer function to call), so only its
    # NAME can come from the pipeline — the resolver's own helper.
    (run_dir / _cleaned_transcript_relpath(es_rel)).write_text(_ENGLISH, encoding="utf-8")
    # Run by-products the capture must NOT take (see CAPTURED_STACK_SUFFIXES).
    (run_dir / "transcripts/p10_e02.anon.txt").write_text("anon", encoding="utf-8")
    (run_dir / "transcripts/p10_e02.manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "metadata/p10_e02.metadata.json").write_text(
        json.dumps(_metadata("p10_e02", es_rel, "es-ES")), encoding="utf-8"
    )
    (run_dir / "metadata/p10_e02.kg.json").write_text(
        json.dumps(_kg(["topic:trail-building", "topic:erosion"])), encoding="utf-8"
    )

    # --- p01_e01: English. No ledger, no swap; it must have no render.
    en_rel = "transcripts/p01_e01.txt"
    (run_dir / en_rel).write_text("Welcome back.\n", encoding="utf-8")
    (run_dir / "metadata/p01_e01.metadata.json").write_text(
        json.dumps(_metadata("p01_e01", en_rel, "en")), encoding="utf-8"
    )
    return root


@pytest.fixture
def capture_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    dirs = {
        "render": tmp_path / "pipeline-renders",
        "kg": tmp_path / "pipeline-kg",
        "summary": tmp_path / "pipeline-summaries",
    }
    monkeypatch.setattr(_bavc, "CAPTURED_RENDER_DIR", dirs["render"])
    monkeypatch.setattr(_bavc, "CAPTURED_KG_DIR", dirs["kg"])
    monkeypatch.setattr(_bavc, "CAPTURED_SUMMARY_DIR", dirs["summary"])
    return dirs


class TestLoadPipelineOutputs:
    def test_the_english_render_is_read_from_the_canonical_path(self, run_root: Path) -> None:
        # THE regression: this came back "" for every episode after D-44.
        out = _bavc._load_pipeline_outputs(run_root)
        assert out["p10_e02"]["english_render"] == _ENGLISH
        assert out["p10_e02"]["english_segments"][0]["text"] == _ENGLISH.strip()

    def test_the_source_suffix_is_written_for_the_replay(self, run_root: Path) -> None:
        # `_replay_english_stack` read this for a day before anything wrote it. Region subtag
        # stripped: the feed declares es-ES, the file the swap wrote is `.es.txt`.
        out = _bavc._load_pipeline_outputs(run_root)
        assert out["p10_e02"]["source_suffix"] == "es.txt"

    def test_the_stack_is_exactly_the_allow_list(self, run_root: Path) -> None:
        stack = _bavc._load_pipeline_outputs(run_root)["p10_e02"]["english_stack"]
        assert set(stack) == set(_bavc.CAPTURED_STACK_SUFFIXES), (
            "the run is missing a suffix the capture expects, or the allow-list grew — either "
            f"way the pipeline's layout and the builder's disagree: {sorted(stack)}"
        )
        assert stack["txt"] == _ENGLISH
        assert "anon.txt" not in stack and "manifest.json" not in stack

    def test_the_ledger_supplies_title_and_status(self, run_root: Path) -> None:
        ep = _bavc._load_pipeline_outputs(run_root)["p10_e02"]
        assert ep["translated_title"] == _TITLE_EN
        ledger = json.loads(
            (run_root / "feeds/p10/run_20261003/transcripts/p10_e02.translation.json").read_text(
                encoding="utf-8"
            )
        )
        assert ep["translation_status"] == ledger.get("status")

    def test_kg_topics_are_read_and_other_nodes_are_not(self, run_root: Path) -> None:
        topics = _bavc._load_pipeline_outputs(run_root)["p10_e02"]["kg_topics"]
        assert [t["id"] for t in topics] == ["topic:trail-building", "topic:erosion"]

    def test_an_english_episode_has_no_render(self, run_root: Path) -> None:
        ep = _bavc._load_pipeline_outputs(run_root)["p01_e01"]
        assert ep["english_render"] == ""
        assert ep["english_stack"] == {}
        assert ep["source_suffix"] is None


class TestCaptureRoundTrip:
    """Capture, then replay into a fresh directory: the replay must reproduce the swap."""

    def test_the_render_is_captured(self, run_root: Path, capture_dirs: dict[str, Path]) -> None:
        outputs = _bavc._load_pipeline_outputs(run_root)
        written, skipped = _bavc._capture_english_renders(outputs, "v3", run_root)
        # Before the fix this was (0, [both episodes]) and the command still reported success.
        assert written == 1
        assert [s.split()[0] for s in skipped] == ["p01_e01"]
        assert (capture_dirs["render"] / "v3" / "p10_e02.json").is_file()

    def test_the_replay_reproduces_the_d44_layout(
        self, run_root: Path, capture_dirs: dict[str, Path], tmp_path: Path
    ) -> None:
        outputs = _bavc._load_pipeline_outputs(run_root)
        _bavc._capture_english_renders(outputs, "v3", run_root)

        # The builder writes the SOURCE body to the canonical names, then replays.
        replay = tmp_path / "replay"
        replay.mkdir()
        (replay / "p10_e02.txt").write_text(_SOURCE, encoding="utf-8")
        (replay / "p10_e02.segments.json").write_text(
            json.dumps(_segments(_SOURCE)), encoding="utf-8"
        )
        n = _bavc._replay_english_stack(replay, "p10_e02", "v3")

        assert n == len(_bavc.CAPTURED_STACK_SUFFIXES)
        assert (replay / "p10_e02.txt").read_text(encoding="utf-8") == _ENGLISH
        assert (replay / "p10_e02.es.txt").read_text(encoding="utf-8") == _SOURCE
        assert (replay / "p10_e02.es.segments.json").is_file()
        # The ANALYSIS base every GI/KG/index reader resolves to. Without it the corpus claims
        # `translated` and analyses the source text.
        assert (replay / "p10_e02.adfree.txt").is_file()

    def test_kg_topics_round_trip(self, run_root: Path, capture_dirs: dict[str, Path]) -> None:
        outputs = _bavc._load_pipeline_outputs(run_root)
        written, skipped = _bavc._capture_kg_topics(outputs, "v3", run_root)
        assert written == 1 and [s.split()[0] for s in skipped] == ["p01_e01"]
        assert _bavc._captured_topics_for("p10_e02", "v3") == [
            "topic:trail-building",
            "topic:erosion",
        ]

    def test_capture_or_remind_writes_all_three_legs(
        self, run_root: Path, capture_dirs: dict[str, Path], tmp_path: Path
    ) -> None:
        # The one gesture the operator runs. Each leg once had a reader and no working writer.
        outputs = _bavc._load_pipeline_outputs(run_root)
        gt = tmp_path / "gt"
        gt.mkdir()
        _bavc._capture_or_remind(
            capture=True,
            pipeline_outputs=outputs,
            from_pipeline=list(outputs),
            version="v3",
            gt_dir=gt,
            run_root=run_root,
        )
        for leg in ("render", "kg", "summary"):
            assert (capture_dirs[leg] / "v3" / "p10_e02.json").is_file(), leg
