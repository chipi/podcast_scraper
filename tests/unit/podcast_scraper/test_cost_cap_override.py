"""A repair must be able to state its own budget, and the env var cannot do it.

`COST_SOFT_CAP_USD_PER_RUN` looks like an override and is not one. `_load_float_env_var` only
fills a field the profile left unset:

    if field_name not in data or data.get(field_name) is None:

Every deployed profile sets `cost_soft_cap_usd_per_run`, so the env var is silently ignored in
exactly the situation someone would reach for it. A run that "set the cap" and was quietly
governed by a different number is the same class of failure as the incident these tests exist
for: a safeguard believed to be active that was not.

Hence `--cost-soft-cap-usd-per-run`, matching the precedent of `--litellm-api-base` (#1676),
added because LiteLLM was the one provider namespace lacking the per-run flag its siblings had.
"""

# mypy: disable-error-code="call-arg"
# Config(rss_url=...) — the field declares alias="rss", so mypy's pydantic plugin only
# knows the alias while populate-by-name accepts either at runtime. Same as
# test_reprocess_episode_ids.py.

from __future__ import annotations

import pytest

from podcast_scraper import config
from podcast_scraper.cli import parse_args

pytestmark = [pytest.mark.unit]


def test_the_flag_parses_as_a_number() -> None:
    args = parse_args(["--rss", "https://example.com/f.xml", "--cost-soft-cap-usd-per-run", "13.5"])
    assert args.cost_soft_cap_usd_per_run == 13.5


def test_omitting_it_leaves_the_profile_in_charge() -> None:
    args = parse_args(["--rss", "https://example.com/f.xml"])
    assert args.cost_soft_cap_usd_per_run is None


def test_the_ENV_VAR_alone_cannot_override_a_profile_value(monkeypatch) -> None:
    """The reason the flag had to exist. Pinned so nobody 'simplifies' it back to an env var."""
    monkeypatch.setenv("COST_SOFT_CAP_USD_PER_RUN", "99.0")
    cfg = config.Config(rss_url="https://example.com/f.xml", cost_soft_cap_usd_per_run=25.0)
    assert cfg.cost_soft_cap_usd_per_run == 25.0, (
        "the env var overrode a set profile value — if this ever becomes true, the flag is "
        "redundant; while it is false, an env-var-only override is silently ignored"
    )


def test_the_env_var_still_works_when_nothing_else_sets_the_cap(monkeypatch) -> None:
    """It fills a gap, which is its actual contract — worth pinning so the flag doesn't break it."""
    monkeypatch.setenv("COST_SOFT_CAP_USD_PER_RUN", "7.5")
    cfg = config.Config(rss_url="https://example.com/f.xml")
    assert cfg.cost_soft_cap_usd_per_run == 7.5


def test_an_explicit_cap_reaches_the_config_and_wins() -> None:
    cfg = config.Config(rss_url="https://example.com/f.xml", cost_soft_cap_usd_per_run=13.0)
    assert cfg.cost_soft_cap_usd_per_run == 13.0


def test_the_prod_profile_cap_is_sized_above_a_normal_nightly() -> None:
    """Guards the regression I nearly shipped.

    Measured 2026-08-19 against the 678-episode corpus: $0.238/episode all-in, 14 feeds, and a
    p90 episode of 91 min. A nightly at 2 new episodes per feed is $6.65 mean / $11.55 at p90.
    The cap was $5 — so enforcing it at whole-run scope would have HALTED a normal nightly, and
    every feed after the break would have got nothing at all.

    The number is free to change as feeds grow; what must stay true is that it clears a p90
    nightly with room, because the thing protecting against runaways is the selection gate, not
    this cap.
    """
    import yaml

    with open("config/profiles/cloud_balanced.yaml", encoding="utf-8") as fh:
        profile = yaml.safe_load(fh)

    cap = float(profile["cost_soft_cap_usd_per_run"])
    p90_nightly = 14 * 2 * (91 * 0.0043 + 0.020)  # feeds x eps x (ASR + LLM)
    assert cap > p90_nightly, (
        f"cap ${cap:.2f} is below a p90 nightly (${p90_nightly:.2f}) — a normal run would abort "
        "partway and the remaining feeds would be skipped entirely"
    )
    assert profile["cost_soft_cap_action"] == "abort"
    # and the alert must fire BEFORE the cap, or it tells you nothing you didn't already learn
    assert float(profile["cost_daily_alert_usd"]) < cap
