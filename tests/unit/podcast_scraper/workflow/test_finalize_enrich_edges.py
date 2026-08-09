"""_finalize_pipeline auto-runs enrich-edges inline (incremental-add P1.3).

Edges (HAS_EPISODE / MENTIONS / SPOKEN_BY) went stale on every add because enrich-edges was a
manual CLI verb, never auto-invoked. finalize now runs it inline with the result SURFACED on
``pipeline_metrics.edges_enriched`` (not fire-and-forget). Tests: the end-to-end derivation on a
real one-episode corpus, and the surfaced-error contract (failure sets the flag False, no raise).
"""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow import orchestration

pytestmark = pytest.mark.unit


def _build_corpus(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"title": "Test Show"},
                "episode": {"episode_id": "ep1"},
                "content": {
                    "transcript_file_path": "transcript.txt",
                    "detected_hosts": [],
                    "detected_guests": ["Elon Musk"],
                },
                "grounded_insights": {"artifact_path": "ep1.gi.json"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ep1.gi.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "model_version": "t",
                "prompt_version": "t",
                "episode_id": "ep1",
                "nodes": [
                    {"id": "episode:ep1", "type": "Episode", "properties": {}},
                    {
                        "id": "insight:1",
                        "type": "Insight",
                        "properties": {"text": "Elon Musk plans to list SpaceX."},
                    },
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ep1.kg.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "person:elon-musk",
                        "type": "Entity",
                        "properties": {"name": "Elon Musk", "kind": "person"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "transcript.txt").write_text("Speaker 1: Hello. Speaker 2: Elon Musk.", "utf-8")


def test_finalize_enrich_edges_populates_gi_edges(tmp_path):
    _build_corpus(tmp_path)
    metrics = SimpleNamespace()
    orchestration._finalize_enrich_edges(str(tmp_path), metrics)

    art = json.loads((tmp_path / "ep1.gi.json").read_text(encoding="utf-8"))
    edge_types = {e["type"] for e in art["edges"]}
    assert "HAS_EPISODE" in edge_types  # no manual enrich-edges pass needed
    assert edge_types & {"MENTIONS_PERSON", "MENTIONS_ORG"}
    assert metrics.edges_enriched is True  # surfaced on the metrics


def test_finalize_enrich_edges_surfaces_nonzero_rc(monkeypatch, tmp_path):
    # _finalize_enrich_edges imports run_enrich_edges_cli from cli_handlers at call time.
    from podcast_scraper.search import cli_handlers

    monkeypatch.setattr(cli_handlers, "run_enrich_edges_cli", lambda *a, **k: 2)
    metrics = SimpleNamespace()
    orchestration._finalize_enrich_edges(str(tmp_path), metrics)
    assert metrics.edges_enriched is False


def test_finalize_enrich_edges_surfaces_exception_without_raising(monkeypatch, tmp_path):
    from podcast_scraper.search import cli_handlers

    def _boom(*_a, **_k):
        raise RuntimeError("edge derivation exploded")

    monkeypatch.setattr(cli_handlers, "run_enrich_edges_cli", _boom)
    metrics = SimpleNamespace()
    orchestration._finalize_enrich_edges(str(tmp_path), metrics)  # must NOT raise
    assert metrics.edges_enriched is False


def test_finalize_enrich_edges_calls_with_output_dir(monkeypatch, tmp_path):
    captured = {}

    def _rec(args: Namespace, _logger) -> int:
        captured["output_dir"] = args.output_dir
        captured["no_speaker"] = args.no_speaker
        return 0

    from podcast_scraper.search import cli_handlers

    monkeypatch.setattr(cli_handlers, "run_enrich_edges_cli", _rec)
    metrics = SimpleNamespace()
    orchestration._finalize_enrich_edges(str(tmp_path), metrics)
    assert captured["output_dir"] == str(tmp_path)
    assert captured["no_speaker"] is False
    assert metrics.edges_enriched is True
