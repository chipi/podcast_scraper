"""``--litellm-api-base`` must override the profile without clobbering it when absent (#1676).

WHY THE FLAG EXISTS
``litellm_api_base`` is set by the PROFILE — ``config/profiles/cloud_balanced.yaml`` pins
``http://homelab:4001/v1`` — and LiteLLM was the ONE provider namespace with no ``--*-api-base``
flag while eight siblings had one (openai / gemini / anthropic / mistral / deepgram / deepseek /
grok / ollama). Its only override layer was the box's ``viewer_operator.yaml``.

That gap was load-bearing, not cosmetic. ``reprocess-prod.yml`` invokes the CLI as
``--config <profile>``, which bypasses ``viewer_operator.yaml`` entirely — so a production
reprocess had NO way to reach the prod-VPS gateway (ADR-142) and would silently route every LLM
call through the homelab gateway instead. Profiles are a generated view of the model registry
(ADR-112) and must never be hand-edited, so editing the pin was never a sanctioned fix either.

WHY THIS TEST IS MORE THAN "the flag sets the field"
ADR-122 (#1253) records a field that shipped un-disable-able because a CLI-side default beat an
explicit config value, and adding a flag to a field that previously had none is exactly the change
that can reintroduce that. So the absent-flag direction is pinned here alongside the override.

Note on the mechanism, measured rather than assumed: ``_load_and_merge_config`` calls
``parser.set_defaults(**config_dump)`` BEFORE ``parse_args``, so a config-file value arrives
already on ``args``. The neighbouring unguarded write —
``payload["ollama_api_base"] = getattr(args, "ollama_api_base", None)`` — therefore stores that
value rather than ``None``, and a profile's ``ollama_api_base`` does survive today. The guarded
form used for ``litellm_api_base`` buys ordering-independence, not a fix to a live bug: a key that
is never written cannot lose to a config file under any future ordering of merge-then-parse.
"""

from __future__ import annotations

import textwrap

import pytest

from podcast_scraper.cli import _build_config, parse_args

pytestmark = [pytest.mark.unit]

_BASE = ["https://example.com/feed.xml", "--output-dir", "/tmp/_litellm_api_base_flag_test"]

_HOMELAB = "http://homelab:4001/v1"
_PROD_GATEWAY = "http://127.0.0.1:4001/v1"


def _cfg(argv):
    return _build_config(parse_args([*argv]))


def _profile(tmp_path, base: str):
    """A config file that pins a gateway, standing in for ``cloud_balanced.yaml``."""
    path = tmp_path / "profile.yaml"
    path.write_text(
        textwrap.dedent(f"""\
            litellm_api_base: {base}
            """),
        encoding="utf-8",
    )
    return str(path)


def test_flag_sets_the_gateway():
    assert _cfg(["--litellm-api-base", _PROD_GATEWAY, *_BASE]).litellm_api_base == _PROD_GATEWAY


def test_flag_overrides_the_profile(tmp_path):
    """THE prod-repair case: the profile pins homelab, the run must reach the prod gateway."""
    cfg = _cfg(
        ["--config", _profile(tmp_path, _HOMELAB), "--litellm-api-base", _PROD_GATEWAY, *_BASE]
    )
    assert cfg.litellm_api_base == _PROD_GATEWAY


def test_absent_flag_does_not_clobber_the_profile(tmp_path):
    """THE ADR-122 (#1253) direction: a flag's absence must never strip a config-supplied value.

    Every existing run omits this flag, so a regression here would silently remove the gateway
    from all of them. This currently holds via two independent mechanisms — the config file
    reaching ``args`` through ``set_defaults``, and the guarded payload write — and this assertion
    is what fails if either one is later removed.
    """
    cfg = _cfg(["--config", _profile(tmp_path, _HOMELAB), *_BASE])
    assert cfg.litellm_api_base == _HOMELAB


def test_absent_flag_and_no_profile_value_stays_none():
    """No flag, no profile value: the field must stay unset rather than acquire a CLI default."""
    assert _cfg(_BASE).litellm_api_base is None


def test_flag_is_registered_on_the_parser():
    """Guards the wiring itself: the arg group must actually be attached to the parser.

    ``_add_litellm_arguments`` defined but never called would leave every assertion above
    passing only because ``parse_args`` errors out — so pin the attribute's existence directly.
    """
    args = parse_args([*_BASE])
    assert hasattr(args, "litellm_api_base")
    assert args.litellm_api_base is None
