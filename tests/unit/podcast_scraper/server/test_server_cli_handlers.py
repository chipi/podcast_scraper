"""Unit tests for ``podcast_scraper.server.cli_handlers`` (serve CLI surface)."""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from pathlib import Path

import pytest

from podcast_scraper.server import cli_handlers

_log = logging.getLogger("test_server_cli_handlers")


def test_parse_serve_argv_minimal() -> None:
    ns = cli_handlers.parse_serve_argv(["--output-dir", "/tmp/corpus-x"])
    assert ns.output_dir == "/tmp/corpus-x"
    assert ns.host == "127.0.0.1"
    assert ns.port == 8000
    assert ns.command == "serve"


def test_parse_serve_argv_feature_flags() -> None:
    ns = cli_handlers.parse_serve_argv(
        [
            "--output-dir",
            "/tmp/x",
            "--enable-feeds-api",
            "--enable-operator-config-api",
            "--enable-jobs-api",
            "--config-file",
            "/tmp/op.yaml",
        ],
    )
    assert ns.enable_feeds_api is True
    assert ns.enable_operator_config_api is True
    assert ns.enable_jobs_api is True
    assert ns.config_file == "/tmp/op.yaml"


def test_env_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_SERVE_TEST_FLAG", raising=False)
    assert cli_handlers._env_truthy("PODCAST_SERVE_TEST_FLAG") is False
    monkeypatch.setenv("PODCAST_SERVE_TEST_FLAG", "yes")
    assert cli_handlers._env_truthy("PODCAST_SERVE_TEST_FLAG") is True


def test_merged_bool_flag_cli_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_SERVE_ENABLE_JOBS_API", "0")
    ns = Namespace(enable_jobs_api=True)
    assert (
        cli_handlers._merged_bool_flag(
            getattr(ns, "enable_jobs_api", False),
            "PODCAST_SERVE_ENABLE_JOBS_API",
        )
        is True
    )


def test_sync_reload_environ_sets_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PODCAST_SERVE_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("PODCAST_SERVE_ENABLE_FEEDS_API", raising=False)
    monkeypatch.delenv("PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API", raising=False)
    monkeypatch.delenv("PODCAST_SERVE_ENABLE_JOBS_API", raising=False)
    ns = Namespace(
        enable_feeds_api=False,
        enable_operator_config_api=False,
        enable_jobs_api=False,
        config_file=None,
    )
    cli_handlers._sync_reload_environ(ns, tmp_path)
    assert Path(os.environ["PODCAST_SERVE_OUTPUT_DIR"]).resolve() == tmp_path.resolve()


def test_run_serve_missing_output_dir_returns_2() -> None:
    ns = Namespace(output_dir="/no/such/dir/podcast-serve-test-xyz")
    assert cli_handlers.run_serve(ns, _log) == 2
