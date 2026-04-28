"""Unit tests for ``pipeline_jobs.build_pipeline_argv``."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.rss.feeds_spec import FEEDS_SPEC_DEFAULT_BASENAME
from podcast_scraper.server.pipeline_jobs import build_pipeline_argv

pytestmark = pytest.mark.integration


def test_argv_includes_feeds_spec_when_present(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    spec_path = corpus / FEEDS_SPEC_DEFAULT_BASENAME
    spec_path.write_text(
        "feeds:\n  - https://example.com/podcast.xml\n",
        encoding="utf-8",
    )
    argv = build_pipeline_argv(corpus, op)
    idx = argv.index("--feeds-spec")
    assert argv[idx + 1] == str(spec_path.resolve())


def test_argv_multi_feed_still_passes_feeds_spec(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    (corpus / FEEDS_SPEC_DEFAULT_BASENAME).write_text(
        "feeds:\n  - https://a.example/1.xml\n  - https://b.example/2.xml\n",
        encoding="utf-8",
    )
    argv = build_pipeline_argv(corpus, op)
    assert "--feeds-spec" in argv


def test_argv_omits_feeds_spec_when_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    assert "--feeds-spec" not in argv


def test_argv_includes_profile_before_config_when_present(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\nmax_episodes: 1\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    i_prof = argv.index("--profile")
    i_cfg = argv.index("--config")
    assert argv[i_prof + 1] == "cloud_balanced"
    assert argv[i_cfg + 1] == str(op)
    assert i_prof < i_cfg


def test_argv_omits_profile_when_operator_has_none(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("max_episodes: 2\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    assert "--profile" not in argv
    assert "--config" in argv
