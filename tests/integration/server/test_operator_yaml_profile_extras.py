"""pipeline_install_extras parsing for Docker job path (#660)."""

from podcast_scraper.server.operator_yaml_profile import parse_pipeline_install_extras
import pytest

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def test_parse_pipeline_install_extras_ml() -> None:
    text = "pipeline_install_extras: ml\nmax_episodes: 1\n"
    assert parse_pipeline_install_extras(text) == "ml"


def test_parse_pipeline_install_extras_llm_quoted() -> None:
    text = 'pipeline_install_extras: "llm"\n'
    assert parse_pipeline_install_extras(text) == "llm"


def test_parse_pipeline_install_extras_missing() -> None:
    assert parse_pipeline_install_extras("max_episodes: 1\n") is None


def test_parse_pipeline_install_extras_empty_value_then_valid() -> None:
    """Empty quoted token is skipped; later line wins (branch ``if v`` false → continue)."""
    text = 'pipeline_install_extras: ""\npipeline_install_extras: ml\n'
    assert parse_pipeline_install_extras(text) == "ml"
