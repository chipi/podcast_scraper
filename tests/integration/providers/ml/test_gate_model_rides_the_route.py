"""The value-gate rater must ride the summariser's route — asserted on the CONFIG THAT RUNS.

Why this file exists separately from test_registry_is_the_source_of_truth.py: that suite asserts on
``resolve_profile_to_settings``, the registry's own view. During this change the registry was already
returning the right answer while ``Config`` still resolved to a direct Anthropic call, because
``gi_value_gate_provider``/``_model`` were not in REGISTRY_GOVERNED_FIELDS and a hand-authored YAML
literal silently won. The registry-level test passed throughout. Only building a real ``Config``
catches that, so these tests deliberately pay the cost of full profile resolution.

The rule under test (2026-08-29): route consistency for PRODUCTION, vendor-disjointness for
EVALUATION. Bake-off profiles are exempt and that exemption is itself asserted, so nobody "fixes"
them into route-consistency and quietly re-introduces the #939 same-vendor bias into a cohort.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

REPO = Path(__file__).resolve().parents[4]

for _k in (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPGRAM_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "GROK_API_KEY",
    "MISTRAL_API_KEY",
    "LITELLM_API_KEY",
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
):
    os.environ.setdefault(_k, "dummy-for-validation")


def _profile_names() -> list[str]:
    return sorted(
        Path(p).stem
        for p in glob.glob(str(REPO / "config" / "profiles" / "*.yaml"))
        if not p.endswith(".example.yaml")
    )


def _resolved(name: str):
    from podcast_scraper.config import Config

    # model_validate, not kwargs: ``rss_url`` is a pydantic alias mypy cannot see, and
    # ``rss_urls`` is typed as List[RssFeedEntry], so neither kwarg form type-checks.
    return Config.model_validate({"rss_url": "https://route-check.example/f.xml", "profile": name})


def _summariser_wire_model(cfg) -> str | None:
    provider = cfg.summary_provider
    return getattr(cfg, f"{provider}_summary_model", None) or getattr(cfg, "summary_model", None)


@pytest.mark.parametrize("name", _profile_names())
def test_production_profile_rates_on_the_summarisers_route(name: str) -> None:
    """No production profile may rate insights through a provider it does not summarise with.

    An off-route rater means a second vendor, a second credential, and spend that never reaches the
    gateway's SpendLogs — which is how a litellm-routed pipeline came to call the Anthropic API
    directly, mid-run, while every cost report showed only gateway traffic.
    """
    if name.startswith("bakeoff_"):
        pytest.skip("bake-off profiles are evaluation cohorts — see the exemption test below")
    cfg = _resolved(name)
    rater = getattr(cfg, "gi_value_gate_provider", None)
    if rater is None:
        # An LLM summariser + gate ENABLED + no rater is not "inapplicable" — it is a silent
        # self-grade at half strictness, taking gi/value_gate.py:90-91 where nothing is logged and
        # nothing is measured. cloud_split_dgx_down sat in exactly this state: the DGX-outage
        # failover profile quietly halving gate strictness during the window its output is compared
        # against baseline. Returning early here is what let it through the first time.
        from podcast_scraper.providers.ml.model_registry import _LLM_PROVIDERS

        if getattr(cfg, "gi_value_gate_enabled", False) and cfg.summary_provider in _LLM_PROVIDERS:
            pytest.fail(
                f"{name}: gate is ENABLED with LLM summariser {cfg.summary_provider!r} but no rater "
                f"resolves, so it self-grades with no log and no metric. Pin an on-route "
                f"gi_value_gate_provider/_model, or promote the profile to a registry preset."
            )
        return  # genuinely no LLM on this path
    assert rater == cfg.summary_provider, (
        f"{name}: summarises via {cfg.summary_provider!r} but rates insights via {rater!r}. The "
        f"rater must ride the summariser's route."
    )


@pytest.mark.parametrize("name", _profile_names())
def test_a_curated_route_does_not_self_grade(name: str) -> None:
    """Same route, DIFFERENT model, wherever we have curated a sibling.

    Self-grading is measurably lenient — ~10% of insights dropped against ~25% for a distinct rater.
    cloud_balanced did exactly this (podcast-flash-0731 rating podcast-flash-0731) for all 765
    episodes before 2026-08-29, so this is a regression guard, not a hypothetical.
    """
    from podcast_scraper.providers.ml.model_registry import (
        _PREFERRED_GATE_MODEL,
        _PROFILE_PRESETS,
    )

    if name.startswith("bakeoff_"):
        pytest.skip("bake-off profiles are evaluation cohorts")
    cfg = _resolved(name)
    provider = cfg.summary_provider
    rater_model = getattr(cfg, "gi_value_gate_model", None)
    if rater_model is None or provider not in _PREFERRED_GATE_MODEL:
        # Gate not configured on this profile, or an uncurated route where self-grade is the
        # accepted outcome (value_gate logs it at WARNING).
        return

    # NOT self-grading is the invariant, and it holds for every profile. The exact alias is only
    # dictated for REGISTRY-GOVERNED profiles: `litellm` names a gateway, and profiles pointed at a
    # different gateway instance (homelab_balanced -> http://homelab:4001) have their own alias
    # namespace, where podcast-pro-0829 does not exist. Asserting the literal there would demand an
    # alias the gateway cannot serve.
    assert rater_model != _summariser_wire_model(cfg), (
        f"{name}: {provider} rates its own output with {rater_model!r} — self-grading, roughly "
        f"half as strict as a distinct rater."
    )
    if name in _PROFILE_PRESETS:
        assert rater_model == _PREFERRED_GATE_MODEL[provider], (
            f"{name} is registry-governed, so curated route {provider!r} must rate with "
            f"{_PREFERRED_GATE_MODEL[provider]!r}, got {rater_model!r}"
        )


def test_a_litellm_rater_alias_is_advertised_by_the_gateway() -> None:
    """The gateway must serve the alias before a profile names it.

    The pipeline runs with ``litellm_verify_served_model=true``: an alias the gateway does not
    advertise fails the RUN, not the config load. That makes this a deploy-ORDERING constraint —
    infra/litellm/config.yaml ships before the profiles that reference it — and a test is the only
    place that ordering is written down where someone will trip over it.
    """
    from podcast_scraper.providers.ml.model_registry import _PREFERRED_GATE_MODEL

    alias = _PREFERRED_GATE_MODEL.get("litellm")
    gateway = (REPO / "infra" / "litellm" / "config.yaml").read_text(encoding="utf-8")
    assert f"model_name: {alias}" in gateway, (
        f"the litellm gate alias {alias!r} is not advertised in infra/litellm/config.yaml. With "
        f"litellm_verify_served_model=true every GI run on a litellm profile would fail."
    )


def test_governed_gate_fields_cannot_be_overridden_by_a_yaml_literal() -> None:
    """The omission that made the whole fix inert must not come back.

    The resolver derived the rater correctly for months while hand-authored YAML literals overrode
    it, because these two fields were absent from REGISTRY_GOVERNED_FIELDS. Ungoverned routing
    fields do not stay correct; they stay whatever someone last typed.
    """
    from podcast_scraper.providers.ml.model_registry import REGISTRY_GOVERNED_FIELDS

    for field in ("gi_value_gate_provider", "gi_value_gate_model"):
        assert field in REGISTRY_GOVERNED_FIELDS, (
            f"{field} is not registry-governed, so a profile YAML can silently override the "
            f"derived rater and materialisation will not correct it."
        )


@pytest.mark.parametrize("name", [n for n in _profile_names() if n.startswith("bakeoff_")])
def test_bakeoff_profiles_keep_a_vendor_disjoint_judge(name: str) -> None:
    """The EXEMPTION, asserted — so route-consistency is never "fixed" into a bake-off.

    A cohort judge that shares a vendor with the arm it grades hands that arm a free pass, and the
    scoreboard then reports our judge assignment as model quality (#939). Production wants one
    route; evaluation wants two vendors. Both rules are real; they govern different things.
    """
    cfg = _resolved(name)
    judge = getattr(cfg, "gi_value_gate_provider", None)
    if judge is None:
        return
    assert judge != cfg.summary_provider, (
        f"{name} is an evaluation cohort: its judge ({judge!r}) must NOT share a vendor with the "
        f"arm it grades ({cfg.summary_provider!r}) — that is #939 self-grading bias."
    )
