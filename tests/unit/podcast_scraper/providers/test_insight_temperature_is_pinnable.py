"""Can an eval PIN the sampler? For a long time: no — and nothing said so.

Every arm YAML in the bake-off carried ``temperature: 0.0``. That key mapped to no ``Config`` field,
so insight extraction sampled at the provider default of **0.3** in every run we have ever scored.
The line looked like a control and was decoration.

This matters more than a stray default, because it corrupts the ONE metric the bake-off exists to
produce. At t=0.3 a model disagrees with ITSELF between runs, and that self-disagreement is
indistinguishable from "the other model found knowledge this one missed". The 18-episode
head-to-head credited gemini with 9.3 unique insights/episode over qwen — and an unknown share of
that was the random number generator, not the model. We were within one step of spending ten vendor
arms measuring sampling noise and calling it a leaderboard.

The bug is ADR-111's allowlist, exactly: a key nobody copies does not error, it takes its default,
and the default is "off". So these tests do NOT check that a field exists. They check that a pinned
value SURVIVES the trip to the thing that actually samples — through both hand-maintained allowlists
(the eval's ``params:`` mapping and the CLI's profile-forwarding tuple), for every provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from podcast_scraper import config as cfgmod
from podcast_scraper.evaluation.eval_gi_kg_runtime import merge_eval_task_into_summarizer_config
from podcast_scraper.providers import insight_salvage

pytestmark = pytest.mark.unit

LLM_PROVIDERS = ["anthropic", "deepseek", "gemini", "grok", "mistral", "ollama", "openai"]

# The profiles that run an LLM insight stage. A bake-off winner that does not REPRODUCE in
# production is not a winner, so eval and prod must pin the same sampler.
GI_PROFILES = [
    "cloud_balanced",
    "cloud_quality",
    "cloud_thin",
    "cloud_with_dgx_primary",
    "eval_default",
    "preprod_local_whisper",
    "prod_dgx_balanced",
    "prod_dgx_full_with_fallback",
]


def _cfg(**kw: object) -> cfgmod.Config:
    kw.setdefault("gemini_api_key", "test-key")
    return cfgmod.Config(rss="https://example.com/feed.xml", **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
def test_a_pinned_temperature_reaches_extraction(provider: str) -> None:
    """THE REGRESSION. Not "the field parses" — the value the SAMPLER is handed."""
    cfg = _cfg(gi_insight_temperature=0.0)
    assert insight_salvage.resolve_insight_temperature(cfg, provider) == 0.0, (
        f"{provider} ignores gi_insight_temperature and samples at its own default. A head-to-head "
        f"run this way disagrees with itself between runs, and reports the disagreement as model "
        f"quality."
    )


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
def test_unpinned_still_falls_back_to_the_provider_default(provider: str) -> None:
    """The knob is opt-in. Absent, behaviour is unchanged — this is a new control, not a new
    default that silently re-tunes every existing caller."""
    assert insight_salvage.resolve_insight_temperature(_cfg(), provider) == pytest.approx(0.3)


def test_pinning_insights_does_NOT_re_tune_summarisation_or_speaker_detection() -> None:
    """Why this is its own knob rather than a reuse of ``<provider>_temperature``.

    That field also drives summarisation and speaker detection. Pinning the insight sampler through
    it would have silently changed two unrelated stages mid-bake-off — and the resulting "model
    difference" would have included a summariser we quietly re-tuned."""
    cfg = _cfg(gi_insight_temperature=0.0, gemini_temperature=0.3)
    assert insight_salvage.resolve_insight_temperature(cfg, "gemini") == 0.0
    assert cfg.gemini_temperature == pytest.approx(0.3), "the insight pin leaked into other stages"


def test_the_eval_forwards_the_pin_to_the_run() -> None:
    """ALLOWLIST #1 — the eval's ``params:`` mapping.

    This is the one that was missing. ``merge_eval_task_into_summarizer_config`` copies keys BY
    HAND; an unmapped key does not raise, it just never arrives."""
    cfg = merge_eval_task_into_summarizer_config(
        _cfg(summary_provider="gemini"),
        "grounded_insights",
        {"gi_insight_source": "provider", "gi_insight_temperature": 0.0},
    )
    assert insight_salvage.resolve_insight_temperature(cfg, "gemini") == 0.0, (
        "the eval dropped gi_insight_temperature on the floor — an arm cannot pin its sampler, "
        "which is precisely the bug: every scored run sampled at 0.3 while its YAML said 0.0."
    )


def test_the_cli_forwards_the_pin_from_a_profile() -> None:
    """ALLOWLIST #2 — the CLI's profile-forwarding tuple. A key missing from it is dropped by
    ``--config profile.yaml`` in production, so eval and prod would drift apart."""
    from podcast_scraper.cli import GIL_TUNING_KEYS

    assert "gi_insight_temperature" in GIL_TUNING_KEYS, (
        "the profile's pin never reaches production Config; prod would keep sampling at 0.3 while "
        "the eval ran at 0.0, so the bake-off winner would not reproduce when we reprocess."
    )


@pytest.mark.parametrize("profile", GI_PROFILES)
def test_every_llm_profile_pins_the_sampler(profile: str) -> None:
    """Production must be REPRODUCIBLE: the same episode, reprocessed, yields the same insights.
    Unpinned, it does not — and the 100-episode corpus would be unrepeatable by construction."""
    y = yaml.safe_load(Path(f"config/profiles/{profile}.yaml").read_text())
    assert y.get("gi_insight_temperature") == 0.0, (
        f"{profile} leaves insight extraction sampling at 0.3, so a reprocess of the same episode "
        f"produces different knowledge each time and cannot be compared to the eval that chose the "
        f"model."
    )
