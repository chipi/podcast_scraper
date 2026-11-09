"""The ``gi`` verbs (validate, inspect, show-insight, explore, query, export) on a real fixture
corpus, in process: the insight stage's CLI after its move out of ``cli.py`` (E04 S7)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from podcast_scraper.insight_stage import cli

pytestmark = pytest.mark.unit

CORPUS = Path(__file__).resolve().parents[3] / "fixtures" / "app-validation-corpus" / "v3"
EPISODE = CORPUS / "feeds" / "p01" / "run_20260101-000000" / "metadata" / "p01_e01.gi.json"
LOG = logging.getLogger(__name__)


def _gi(*argv: str) -> int:
    rc: int = cli._run_gi(cli._parse_gi_args(list(argv)), log=LOG)
    return rc


def _first(kind: str) -> dict:
    doc = json.loads(EPISODE.read_text(encoding="utf-8"))
    node: dict = next(n for n in doc["nodes"] if n.get("type") == kind)
    return node


def _insight_topic_label() -> str:
    """The label of a topic an insight is ABOUT (an unlinked topic has nothing to explore)."""
    doc = json.loads(EPISODE.read_text(encoding="utf-8"))
    topics = {n["id"]: n for n in doc["nodes"] if n.get("type") == "Topic"}
    for e in doc.get("edges") or []:
        if str(e.get("type", "")).upper() == "ABOUT" and e.get("to") in topics:
            label: str = topics[e["to"]]["properties"]["label"]
            return label
    raise AssertionError("fixture episode has no insight ABOUT a topic")


def test_validate_accepts_the_fixture_artifacts() -> None:
    assert _gi("validate", str(EPISODE), "--quiet") == 0
    assert _gi("validate", str(EPISODE), "--strict") == 0


def test_validate_refuses_a_file_that_is_not_gi(tmp_path: Path) -> None:
    bad = tmp_path / "x.gi.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _gi("validate", str(bad), "--quiet") != 0


def test_inspect_one_episode_in_both_formats(capsys: pytest.CaptureFixture[str]) -> None:
    assert _gi("inspect", "--episode-path", str(EPISODE)) == 0
    assert _gi("inspect", "--episode-path", str(EPISODE), "--format", "json", "--no-stats") == 0
    out = capsys.readouterr().out
    assert out.strip()


def test_show_insight_finds_the_insight_and_its_quotes(capsys: pytest.CaptureFixture[str]) -> None:
    iid = _first("Insight")["id"]
    assert _gi("show-insight", "--id", iid, "--episode-path", str(EPISODE)) == 0
    assert _gi("show-insight", "--id", iid, "--output-dir", str(CORPUS), "--format", "json") == 0
    assert iid in capsys.readouterr().out


def test_show_insight_of_an_unknown_id_fails() -> None:
    assert _gi("show-insight", "--id", "insight:nope", "--episode-path", str(EPISODE)) != 0


def test_explore_a_topic_across_the_corpus(tmp_path: Path) -> None:
    label = _insight_topic_label()
    assert _gi("explore", "--topic", label, "--output-dir", str(CORPUS), "--limit", "5") == 0
    out = tmp_path / "explore.json"
    rc = _gi(
        "explore",
        "--topic",
        label,
        "--output-dir",
        str(CORPUS),
        "--format",
        "json",
        "--grounded-only",
        "--sort",
        "time",
        "--out",
        str(out),
    )
    assert rc == 0
    assert out.is_file() and json.loads(out.read_text(encoding="utf-8"))


def test_query_answers_a_topic_question(capsys: pytest.CaptureFixture[str]) -> None:
    label = _insight_topic_label()
    rc = _gi("query", "--output-dir", str(CORPUS), "--question", f"What insights about {label}?")
    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_export_writes_ndjson_and_merged(tmp_path: Path) -> None:
    nd = tmp_path / "gi.ndjson"
    merged = tmp_path / "gi.json"
    assert _gi("export", "--output-dir", str(CORPUS), "--format", "ndjson", "--out", str(nd)) == 0
    assert (
        _gi("export", "--output-dir", str(CORPUS), "--format", "merged", "--out", str(merged)) == 0
    )
    assert nd.read_text(encoding="utf-8").strip()
    assert json.loads(merged.read_text(encoding="utf-8"))


def _a_speaker_name() -> str:
    """The name of the person a supporting quote is SPOKEN_BY (explore and query find them)."""
    doc = json.loads(EPISODE.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in doc["nodes"]}
    edges = doc.get("edges") or []
    supporting = {e["to"] for e in edges if str(e.get("type", "")).upper() == "SUPPORTED_BY"}
    for e in edges:
        if str(e.get("type", "")).upper() == "SPOKEN_BY" and e.get("from") in supporting:
            name = ((nodes.get(e["to"]) or {}).get("properties") or {}).get("name")
            if name:
                return str(name)
    raise AssertionError("fixture episode has no named speaker behind a supporting quote")


@pytest.mark.parametrize("sort", ["confidence", "evidence-density"])
def test_explore_sorts_and_filters_a_topics_insights(tmp_path: Path, sort: str) -> None:
    out = tmp_path / "e.json"
    rc = _gi(
        "explore",
        "--topic",
        _insight_topic_label(),
        "--output-dir",
        str(CORPUS),
        "--sort",
        sort,
        "--format",
        "json",
        "--out",
        str(out),
    )
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))


def test_explore_for_an_unknown_speaker_finds_nothing() -> None:
    rc = _gi("explore", "--speaker", "nobody-by-this-name", "--output-dir", str(CORPUS))
    assert rc == 4  # EXIT_NO_RESULTS


def test_explore_strict_and_expand_clusters_still_answer(tmp_path: Path) -> None:
    # One schema-valid episode: the corpus's translated p10_e01 fails strict validation (its Quote
    # carries a ``translation`` property the GIL schema does not allow).
    meta = tmp_path / "feeds" / "p01" / "run_1" / "metadata"
    meta.mkdir(parents=True)
    shutil.copy2(EPISODE, meta / EPISODE.name)
    label = _insight_topic_label()
    rc = _gi(
        "explore", "--topic", label, "--output-dir", str(tmp_path), "--strict", "--expand-clusters"
    )
    assert rc == 0


def test_explore_needs_a_corpus(tmp_path: Path) -> None:
    assert _gi("explore", "--topic", "x", "--output-dir", str(tmp_path / "missing")) != 0


@pytest.mark.parametrize(
    "question", ["Top topics", "Which topics have the most insights?", "What did {speaker} say?"]
)
def test_query_answers_each_supported_shape(
    question: str, capsys: pytest.CaptureFixture[str]
) -> None:
    q = question.format(speaker=_a_speaker_name())
    codes = [
        _gi("query", "--output-dir", str(CORPUS), "--question", q, "--format", fmt)
        for fmt in ("json", "pretty")
    ]
    assert codes[0] == codes[1] == 0
    assert capsys.readouterr().out.strip()


def test_query_refuses_an_unsupported_question() -> None:
    assert _gi("query", "--output-dir", str(CORPUS), "--question", "Why is the sky blue?") != 0


def test_inspect_by_corpus_and_episode_id_with_the_insights_shown() -> None:
    doc = json.loads(EPISODE.read_text(encoding="utf-8"))
    episode_id = doc.get("episode_id")
    assert (
        _gi("inspect", "--output-dir", str(CORPUS), "--episode-id", str(episode_id), "--show") == 0
    )
    assert _gi("inspect", "--output-dir", str(CORPUS), "--episode-id", "no-such-episode") != 0
    assert _gi("inspect") != 0


def test_export_to_stdout_and_its_refusals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gi("export", "--output-dir", str(CORPUS)) == 0
    assert capsys.readouterr().out.strip()
    assert _gi("export", "--output-dir", str(tmp_path / "missing")) != 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _gi("export", "--output-dir", str(empty)) != 0


def test_unknown_subcommand_is_refused() -> None:
    import argparse

    assert cli._run_gi(argparse.Namespace(gi_subcommand="nope"), log=LOG) == 1


@pytest.mark.parametrize(
    ("sub", "target"),
    [
        ("clusters", "run_cluster_browse_cli"),
        ("explore-quotes", "run_explore_quotes_cli"),
        ("topic-insights", "run_topic_insights_cli"),
    ],
)
def test_the_cluster_views_dispatch_to_the_cluster_cli(sub: str, target: str) -> None:
    import argparse
    from unittest.mock import patch

    with patch(f"podcast_scraper.insight_stage.cluster_cli.{target}", return_value=0) as run:
        assert cli._run_gi(argparse.Namespace(gi_subcommand=sub), log=LOG) == 0
    assert run.called
