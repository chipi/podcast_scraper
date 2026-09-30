"""A re-run whose extraction fails must not overwrite a working knowledge graph.

THE INCIDENT (prod, 2026-09-29). ``relabel_only`` rewrites kg.json in place. Two relabel runs
(``d65c2fcf`` pdrl.fm, ``74245ab1`` WSJ) hit an extraction that returned nothing, and the empty
``provider:extraction_failed`` graph replaced working NVFP4 graphs. Both jobs reported
``succeeded``. A repair run made the corpus worse, silently.

Both sides are built with the real ``build_artifact`` — the working graph from a provider that
answers, the failed one from a provider that returns None the way the real one does — so the
guard is tested against the exact provenance strings the pipeline stamps, not hand-written ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.kg.io import previous_artifact_to_keep, write_artifact
from podcast_scraper.kg.pipeline import build_artifact

pytestmark = pytest.mark.integration


class _Answers:
    summary_model = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"

    def extract_kg_graph(self, *_a: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "topics": [{"label": "tungsten supply"}, {"label": "export controls"}],
            "entities": [{"name": "China", "entity_kind": "organization"}],
        }


class _ReturnsNothing:
    summary_model = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"

    def extract_kg_graph(self, *_a: Any, **_kw: Any) -> None:
        return None


def _build(provider: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from podcast_scraper.kg import pipeline

    monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
    return build_artifact(
        "ep:x",
        "transcript text",
        podcast_id="podcast:p1",
        episode_title="The Tungsten Market",
        kg_extraction_provider=provider,
    )


def _on_disk(tmp_path: Path, payload: dict[str, Any]) -> Path:
    p = tmp_path / "0004 - ep.kg.json"
    write_artifact(p, payload, validate=True)
    return p


def test_a_failed_rerun_keeps_the_working_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE regression: the old graph must come back, topics intact."""
    path = _on_disk(tmp_path, _build(_Answers(), monkeypatch))
    failed = _build(_ReturnsNothing(), monkeypatch)
    assert failed["extraction"]["model_version"] == "provider:extraction_failed"

    kept = previous_artifact_to_keep(path, failed)

    assert kept is not None, "a failed re-run would have replaced a working graph with an empty one"
    labels = [n["properties"]["label"] for n in kept["nodes"] if n["type"] == "Topic"]
    assert labels == ["tungsten supply", "export controls"]


def test_a_successful_rerun_is_written_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _on_disk(tmp_path, _build(_Answers(), monkeypatch))
    assert previous_artifact_to_keep(path, _build(_Answers(), monkeypatch)) is None


def test_a_first_run_has_nothing_to_keep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No previous file: the honest empty graph is written, as before."""
    missing = tmp_path / "never-written.kg.json"
    assert previous_artifact_to_keep(missing, _build(_ReturnsNothing(), monkeypatch)) is None


def test_a_previous_failure_is_not_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _on_disk(tmp_path, _build(_ReturnsNothing(), monkeypatch))
    assert previous_artifact_to_keep(path, _build(_ReturnsNothing(), monkeypatch)) is None


def test_a_fabricated_graph_is_not_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``topic_labels`` graphs are summary bullets posing as topics (ADR-156): empty beats them."""
    fabricated = _build(_Answers(), monkeypatch)
    fabricated["extraction"]["model_version"] = "topic_labels"
    path = _on_disk(tmp_path, fabricated)
    assert previous_artifact_to_keep(path, _build(_ReturnsNothing(), monkeypatch)) is None


def test_an_unreadable_previous_file_is_not_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "torn.kg.json"
    path.write_text('{"extraction": {"model_version": "provider:x"', encoding="utf-8")
    assert previous_artifact_to_keep(path, _build(_ReturnsNothing(), monkeypatch)) is None
