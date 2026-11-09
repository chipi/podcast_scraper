"""The insight-cluster commands after their move to the insight stage (E04 S4): ``insight-clusters``
and the ``gi`` views over its output (``clusters``, ``explore-quotes``, ``topic-insights``)."""

from __future__ import annotations

import json
import logging
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from podcast_scraper.gi.explore import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS
from podcast_scraper.insight_stage import cluster_cli

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "app-validation-corpus"
    / "v3"
    / "feeds"
    / "p01"
    / "run_20260101-000000"
    / "metadata"
    / "p01_e01.gi.json"
)
LOG = logging.getLogger(__name__)


def _about_pair() -> tuple[str, str]:
    """``(insight_id, topic_label)`` of an insight ABOUT a topic in the fixture episode."""
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    topics = {n["id"]: n for n in doc["nodes"] if n.get("type") == "Topic"}
    for e in doc.get("edges") or []:
        if e.get("type") == "ABOUT" and e.get("to") in topics:
            return str(e["from"]), str(topics[e["to"]]["properties"]["label"])
    raise AssertionError("fixture episode has no insight ABOUT a topic")


def _corpus(tmp_path: Path) -> tuple[Path, str]:
    """A corpus with the fixture episode and an insight_clusters.json clustering its insight."""
    meta = tmp_path / "feeds" / "p01" / "run_1" / "metadata"
    meta.mkdir(parents=True)
    shutil.copy2(FIXTURE, meta / FIXTURE.name)
    insight_id, label = _about_pair()
    (tmp_path / "search").mkdir()
    payload = {
        "clusters": [
            {
                "cluster_id": "ic:1",
                "member_count": 2,
                "cross_episode": True,
                "canonical_insight": "The canonical insight text",
                "episode_ids": ["ep-a", "ep-b"],
                "members": [{"insight_id": insight_id, "episode_id": "ep-a"}],
            },
            {
                "cluster_id": "ic:2",
                "member_count": 1,
                "cross_episode": False,
                "canonical_insight": "A singleton",
                "episode_ids": ["ep-c"],
                "members": [{"insight_id": "insight:other", "episode_id": "ep-c"}],
            },
        ]
    }
    (tmp_path / "search" / "insight_clusters.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return tmp_path, label


def test_insight_clusters_reports_the_built_payload(tmp_path: Path) -> None:
    args = cluster_cli.parse_insight_clusters_argv(["--output-dir", str(tmp_path)])
    built = {"insight_count": 3, "cluster_count": 1, "cross_episode_clusters": 1, "singletons": 0}
    with patch(
        "podcast_scraper.insight_stage.insight_clusters.build_insight_clusters_for_corpus",
        return_value=built,
    ) as build:
        assert cluster_cli.run_insight_clusters_cli(args, LOG) == EXIT_SUCCESS
    assert build.call_args.args[0] == str(tmp_path)


@pytest.mark.parametrize(
    ("error", "code"),
    [(FileNotFoundError("no gi"), EXIT_NO_ARTIFACTS), (ValueError("x"), EXIT_INVALID_ARGS)],
)
def test_insight_clusters_maps_failures_to_exit_codes(
    tmp_path: Path, error: Exception, code: int
) -> None:
    args = cluster_cli.parse_insight_clusters_argv(["--output-dir", str(tmp_path)])
    with patch(
        "podcast_scraper.insight_stage.insight_clusters.build_insight_clusters_for_corpus",
        side_effect=error,
    ):
        assert cluster_cli.run_insight_clusters_cli(args, LOG) == code


def test_insight_clusters_needs_an_output_dir() -> None:
    assert (
        cluster_cli.run_insight_clusters_cli(Namespace(output_dir=None), LOG) == EXIT_INVALID_ARGS
    )


@pytest.mark.parametrize("fmt", ["pretty", "json"])
def test_clusters_lists_the_top_clusters(
    tmp_path: Path, fmt: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _label = _corpus(tmp_path)
    args = cluster_cli.parse_cluster_browse_argv(
        ["--output-dir", str(root), "--top", "5", "--format", fmt]
    )
    assert cluster_cli.run_cluster_browse_cli(args, LOG) == EXIT_SUCCESS
    assert "canonical insight text" in capsys.readouterr().out.lower()


def test_clusters_without_the_file_is_no_artifacts(tmp_path: Path) -> None:
    args = cluster_cli.parse_cluster_browse_argv(["--output-dir", str(tmp_path)])
    assert cluster_cli.run_cluster_browse_cli(args, LOG) == EXIT_NO_ARTIFACTS


@pytest.mark.parametrize("fmt", ["pretty", "json"])
def test_topic_insights_finds_the_cluster_of_a_topics_insight(
    tmp_path: Path, fmt: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root, label = _corpus(tmp_path)
    args = cluster_cli.parse_topic_insights_argv(
        ["--topic", label, "--output-dir", str(root), "--format", fmt]
    )
    assert cluster_cli.run_topic_insights_cli(args, LOG) == EXIT_SUCCESS
    assert "canonical insight text" in capsys.readouterr().out.lower()


def test_topic_insights_without_clusters_is_no_artifacts(tmp_path: Path) -> None:
    args = cluster_cli.parse_topic_insights_argv(["--topic", "x", "--output-dir", str(tmp_path)])
    assert cluster_cli.run_topic_insights_cli(args, LOG) == EXIT_NO_ARTIFACTS


@pytest.mark.parametrize("fmt", ["pretty", "json"])
def test_explore_quotes_prints_quote_hits(
    tmp_path: Path, fmt: str, capsys: pytest.CaptureFixture[str]
) -> None:
    hit = {
        "text": "a quoted line",
        "score": 0.9,
        "metadata": {"episode_id": "e1", "speaker_id": "p1"},
    }
    args = cluster_cli.parse_explore_quotes_argv(
        ["--query", "q", "--output-dir", str(tmp_path), "--format", fmt]
    )
    with patch(
        "podcast_scraper.search.corpus_search.run_corpus_search",
        return_value=SimpleNamespace(error=None, detail=None, results=[hit]),
    ) as search:
        assert cluster_cli.run_explore_quotes_cli(args, LOG) == EXIT_SUCCESS
    assert search.call_args.kwargs["doc_types"] == ["quote"]
    assert "a quoted line" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "code"), [("no_index", EXIT_NO_ARTIFACTS), ("embed_failed", EXIT_INVALID_ARGS)]
)
def test_explore_quotes_maps_search_errors(tmp_path: Path, error: str, code: int) -> None:
    args = cluster_cli.parse_explore_quotes_argv(["--query", "q", "--output-dir", str(tmp_path)])
    with patch(
        "podcast_scraper.search.corpus_search.run_corpus_search",
        return_value=SimpleNamespace(error=error, detail="d", results=[]),
    ):
        assert cluster_cli.run_explore_quotes_cli(args, LOG) == code


def test_explore_quotes_needs_a_query_and_a_corpus() -> None:
    assert (
        cluster_cli.run_explore_quotes_cli(Namespace(query=" ", output_dir="x"), LOG)
        == EXIT_INVALID_ARGS
    )
    assert (
        cluster_cli.run_explore_quotes_cli(Namespace(query="q", output_dir=None), LOG)
        == EXIT_INVALID_ARGS
    )
