"""CLI wiring for the ``ingest`` subcommand (#1069).

``podcast_scraper ingest <feed-url> --output-dir <corpus>`` routes the pipeline
run through the ingestion primitive (policy seam + dedup + corpus stamp). These
tests cover the routing and the handler adapter with the pipeline **mocked** —
no real transcription/summarization runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper import cli
from podcast_scraper.utils.filesystem import feed_workspace_dirname

pytestmark = [pytest.mark.unit]

_URL = "https://example.com/feed.xml"


def test_parse_args_routes_ingest() -> None:
    args = cli.parse_args(["ingest", _URL, "--output-dir", "/tmp/corpus", "--force"])
    assert args.command == "ingest"
    assert args.rss == _URL  # the positional feed URL
    assert args.force is True
    assert args.single_feed_uses_corpus_layout is True  # forced on for corpus layout


def test_default_run_is_unaffected_by_ingest_branch() -> None:
    # Regression: the extract-method refactor must leave the default run parsing intact.
    args = cli.parse_args([_URL, "--output-dir", "/tmp/corpus"])
    assert not hasattr(args, "command")
    assert args.rss == _URL


def test_run_ingest_command_routes_through_primitive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    cfg = SimpleNamespace(
        rss_url=_URL,
        output_dir=str(corpus / "feeds" / feed_workspace_dirname(_URL)),
        single_feed_uses_corpus_layout=True,
    )
    monkeypatch.setattr(cli, "_build_config", lambda args: cfg)

    import podcast_scraper.service as service

    monkeypatch.setattr(
        service,
        "run",
        lambda c: SimpleNamespace(success=True, episodes_processed=2, summary="ok", error=None),
    )

    rc = cli._run_ingest_command(argparse.Namespace(force=False))

    assert rc == 0
    out = capsys.readouterr().out
    assert "[ingest] ingested" in out
    assert "episodes_added=2" in out


def test_parse_args_routes_enrich_passthrough() -> None:
    # #1069 consistency: enrich is a main-CLI verb; its args pass through untouched.
    args = cli.parse_args(["enrich", "--output-dir", "/tmp/c", "--only", "grounding_rate"])
    assert args.command == "enrich"
    assert args.enrich_argv == ["--output-dir", "/tmp/c", "--only", "grounding_rate"]


def test_enrich_subcommand_delegates_to_enrichment_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_enrich_main(argv: list[str]) -> int:
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("podcast_scraper.enrichment.cli.main", _fake_enrich_main)
    rc = cli.main(["enrich", "--output-dir", "/tmp/c", "--corpus-only"])

    assert rc == 0
    assert seen == [["--output-dir", "/tmp/c", "--corpus-only"]]


def test_run_ingest_command_failed_pipeline_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    cfg = SimpleNamespace(
        rss_url=_URL,
        output_dir=str(corpus / "feeds" / feed_workspace_dirname(_URL)),
        single_feed_uses_corpus_layout=True,
    )
    monkeypatch.setattr(cli, "_build_config", lambda args: cfg)

    import podcast_scraper.service as service

    monkeypatch.setattr(
        service,
        "run",
        lambda c: SimpleNamespace(success=False, episodes_processed=0, summary=None, error="boom"),
    )

    rc = cli._run_ingest_command(argparse.Namespace(force=False))

    assert rc == 1
    assert "error: boom" in capsys.readouterr().out
