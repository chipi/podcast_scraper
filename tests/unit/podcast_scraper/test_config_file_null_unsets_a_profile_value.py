"""An explicit ``null`` in a ``--config`` file unsets what the profile set (D20).

Measured 2026-10-09 on the non-English arc: a run config with ``profile: prod_dgx_full`` and
``translate_api_base: null`` still translated. ``_load_and_merge_config`` carries only non-None
values onto ``args``, so the null never reached ``_build_config``'s payload, and the profile, which
is resolved again there, put its own value back. The operator got no error and no effect.
"""

from __future__ import annotations

import textwrap

import pytest

from podcast_scraper.cli import _build_config, parse_args

pytestmark = [pytest.mark.unit]


def _cfg(tmp_path, body: str, *argv: str):
    path = tmp_path / "run.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return _build_config(parse_args(["--config", str(path), *argv]))


def test_a_null_switches_off_the_profile_translator(tmp_path) -> None:
    cfg = _cfg(
        tmp_path,
        """
        profile: prod_dgx_full
        rss: https://example.com/feed.xml
        output_dir: /tmp/_d20
        translate_api_base: null
        """,
    )
    assert cfg.translate_api_base is None


def test_a_key_left_out_keeps_the_profile_value(tmp_path) -> None:
    cfg = _cfg(
        tmp_path,
        """
        profile: prod_dgx_full
        rss: https://example.com/feed.xml
        output_dir: /tmp/_d20
        """,
    )
    assert cfg.translate_api_base


def test_a_value_still_overrides_the_profile(tmp_path) -> None:
    cfg = _cfg(
        tmp_path,
        """
        profile: prod_dgx_full
        rss: https://example.com/feed.xml
        output_dir: /tmp/_d20
        translate_api_base: http://translator.invalid/v1
        """,
    )
    assert cfg.translate_api_base == "http://translator.invalid/v1"
