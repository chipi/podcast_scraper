"""The ``gi`` verbs (validate, inspect, show-insight, explore, query, export) on a real fixture
corpus, in process: the insight stage's CLI after its move out of ``cli.py`` (E04 S7)."""

from __future__ import annotations

import json
import logging
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
