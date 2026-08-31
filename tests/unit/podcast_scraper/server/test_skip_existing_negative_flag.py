"""``skip_existing=false`` must actually reach the pipeline (#1888 daylight batch).

``--config`` values become argparse DEFAULTS (``_load_and_merge_config``). With only a
``store_true`` flag, a caller who wanted skip_existing=false could do nothing but omit the
flag — which inherits ``skip_existing: true`` from the operator YAML instead. The API
accepted the parameter, reported success, and re-skipped every episode. That is how an A/B
run was launched against episodes it had already ingested.

The subtle half is the DEFAULT: argparse resolves a shared ``dest`` to the last registered
action's default, so a bare ``store_false`` would flip ``--skip-existing``'s default to True
and invert the behaviour of every caller that passes neither flag. Case 2 guards that.
"""

from __future__ import annotations

import pytest

from podcast_scraper import cli
from podcast_scraper.server.jobs import build_pipeline_argv


@pytest.fixture
def cfg_true(tmp_path):
    p = tmp_path / "viewer_operator.yaml"
    p.write_text("skip_existing: true\n", encoding="utf-8")
    return p


@pytest.fixture
def cfg_silent(tmp_path):
    p = tmp_path / "silent.yaml"
    p.write_text("rss_url: https://example.com/f.rss\n", encoding="utf-8")
    return p


def _resolve(extra, cfg_path):
    argv = [
        "https://example.com/f.rss",
        "--config",
        str(cfg_path),
        "--output-dir",
        "/tmp/x",
    ] + extra
    return cli._parse_pipeline_argv(argv).skip_existing


def test_config_true_is_honoured_when_no_flag_given(cfg_true):
    assert _resolve([], cfg_true) is True


def test_default_stays_false_when_nothing_says_otherwise(cfg_silent):
    """Guards the argparse shared-dest trap — a bare store_false would make this True."""
    assert _resolve([], cfg_silent) is False


def test_explicit_positive_flag_still_works(cfg_silent):
    assert _resolve(["--skip-existing"], cfg_silent) is True


def test_negative_flag_overrides_config_true(cfg_true):
    """THE FIX: the caller can now say false against an operator YAML that says true."""
    assert _resolve(["--no-skip-existing"], cfg_true) is False


@pytest.mark.parametrize(
    "skip,expected", [(True, "--skip-existing"), (False, "--no-skip-existing")]
)
def test_argv_builder_always_states_the_choice(tmp_path, skip, expected):
    """Omission is not 'false' — the builder must emit one flag or the other."""
    argv = build_pipeline_argv(
        tmp_path,
        tmp_path / "viewer_operator.yaml",
        run_id="r1",
        feed_url="https://example.com/f.rss",
        skip_existing=skip,
    )
    assert expected in argv
    other = "--no-skip-existing" if skip else "--skip-existing"
    assert other not in argv
