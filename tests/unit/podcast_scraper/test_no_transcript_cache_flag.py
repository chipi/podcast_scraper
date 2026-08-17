"""``--no-transcript-cache`` must disable the cache without becoming un-disable-able (#35).

WHY THE FLAG EXISTS
``transcript_cache_enabled`` has been a Config field since the transcript cache was added, but had
no CLI flag: the only way to bypass the cache was to set it in YAML. ``config/profiles/*.yaml``
are a GENERATED view of the model registry (ADR-112) and must never be hand-edited, so for any
profile that does not already disable it, an operator had no sanctioned way to force a fresh
transcription. Three reprocess profiles (``reprocess_dgx_no_llm``, ``reprocess_dgx_turbo``,
``reprocess_v23_turbo``) do set it false; this flag makes the same choice available with ANY
profile, which is what a one-off repair run needs.

WHY THE TEST IS MORE THAN "the flag sets the field"
This exact field is the one ADR-122 (#1253) names as having shipped BROKEN: a CLI-side default of
True beat a config file's False, making the cache un-disable-able from YAML. Adding an argparse
flag to a field that previously had none is precisely the change that can reintroduce that, so the
config-file direction is pinned here alongside the flag itself.
"""

from __future__ import annotations

import textwrap

import pytest

from podcast_scraper.cli import _build_config, parse_args

pytestmark = [pytest.mark.unit]

_BASE = ["https://example.com/feed.xml", "--output-dir", "/tmp/_transcript_cache_flag_test"]


def _cfg(argv):
    return _build_config(parse_args([*argv]))


def test_flag_disables_the_cache():
    assert _cfg(["--no-transcript-cache", *_BASE]).transcript_cache_enabled is False


def test_absent_flag_leaves_the_cache_on():
    """The default must not change: normal runs still benefit from the cache."""
    assert _cfg(_BASE).transcript_cache_enabled is True


def test_config_file_can_still_disable_it_without_the_flag(tmp_path):
    """THE ADR-122 (#1253) regression, on the field that ADR names.

    A CLI default must never beat an explicit config value. Before that fix this field shipped
    un-disable-able from YAML; adding a flag is exactly the change that could bring it back.
    """
    cfg_file = tmp_path / "no_cache.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
            transcript_cache_enabled: false
            """),
        encoding="utf-8",
    )
    cfg = _cfg(["--config", str(cfg_file), *_BASE])
    assert cfg.transcript_cache_enabled is False


def test_config_file_can_still_enable_it_explicitly(tmp_path):
    """The other direction, so the assertion above is not passing by accident on a default."""
    cfg_file = tmp_path / "with_cache.yaml"
    cfg_file.write_text("transcript_cache_enabled: true\n", encoding="utf-8")
    assert _cfg(["--config", str(cfg_file), *_BASE]).transcript_cache_enabled is True


def test_flag_beats_a_config_file_that_enables_it(tmp_path):
    """An explicit CLI flag is the operator's last word — it must win over the file."""
    cfg_file = tmp_path / "with_cache.yaml"
    cfg_file.write_text("transcript_cache_enabled: true\n", encoding="utf-8")
    cfg = _cfg(["--config", str(cfg_file), "--no-transcript-cache", *_BASE])
    assert cfg.transcript_cache_enabled is False
