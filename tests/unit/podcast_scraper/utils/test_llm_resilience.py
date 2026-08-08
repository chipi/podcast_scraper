"""Tests for the per-(provider, model) resilience profile + cfg override (#1482 RFC-111 follow-up).

A homelab LiteLLM gateway outage (APIConnectionError) exhausted retry_with_metrics' short default
window (~12s) before the operator could react. The fix is a config-driven retry window: short
defaults (unchanged behaviour, unchanged test duration) with prod profiles opting into a long
hold via ``llm_retry_*`` Config fields. These tests assert the RESOLVED profile values only —
never wall-clock sleep — so they stay fast regardless of how large a profile sets the retry window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcast_scraper.utils.llm_resilience import (
    DEFAULT_PROFILE,
    resolve_resilience,
)


@pytest.mark.unit
class TestResolveResilienceNoCfg:
    """Back-compat: calling without cfg (or with cfg=None) is unchanged."""

    def test_unknown_provider_model_gets_default(self) -> None:
        profile = resolve_resilience("openai", "gpt-4o-mini")
        assert profile == DEFAULT_PROFILE

    def test_known_override_still_wins_without_cfg(self) -> None:
        profile = resolve_resilience("gemini", "gemini-2.5-flash-lite")
        assert profile.max_retries == 6
        assert profile.initial_delay == 2.0
        assert profile.max_delay == 60.0


@pytest.mark.unit
class TestResolveResilienceCfgOverride:
    """cfg's general llm_retry_* knobs override the resolved profile, field-by-field."""

    def test_default_cfg_leaves_profile_unchanged(self) -> None:
        """A cfg carrying today's defaults (3 / 1.0 / 30.0) must not perturb the resolved profile —
        this is the "nothing changes unless a profile opts in" contract."""
        cfg = SimpleNamespace(
            llm_retry_max_retries=3,
            llm_retry_initial_delay_seconds=1.0,
            llm_retry_max_delay_seconds=30.0,
        )
        profile = resolve_resilience("litellm", "deepseek-v4-flash", cfg)
        assert profile == DEFAULT_PROFILE
        assert profile.max_retries == 3
        assert profile.max_delay == 30.0

    def test_prod_cfg_produces_long_hold_window(self) -> None:
        """A prod profile (cloud_balanced-shaped) that opts into a long retry window resolves to
        that window — this is the "hold for minutes" contract for a gateway outage."""
        cfg = SimpleNamespace(
            llm_retry_max_retries=12,
            llm_retry_initial_delay_seconds=2.0,
            llm_retry_max_delay_seconds=120.0,
        )
        profile = resolve_resilience("litellm", "deepseek-v4-flash", cfg)
        assert profile.max_retries == 12
        assert profile.initial_delay == 2.0
        assert profile.max_delay == 120.0

    def test_partial_override_composes_with_per_model_profile(self) -> None:
        """Setting only max_retries on cfg must not clobber flash-lite's own longer backoff —
        the override is per-field, not "cfg replaces the whole profile"."""
        cfg = SimpleNamespace(
            llm_retry_max_retries=20,
            llm_retry_initial_delay_seconds=1.0,  # cfg default -> no override
            llm_retry_max_delay_seconds=30.0,  # cfg default -> no override
        )
        profile = resolve_resilience("gemini", "gemini-2.5-flash-lite", cfg)
        assert profile.max_retries == 20  # cfg override
        assert profile.initial_delay == 2.0  # flash-lite's own value, untouched
        assert profile.max_delay == 60.0  # flash-lite's own value, untouched

    def test_missing_llm_retry_attrs_is_a_noop(self) -> None:
        """A cfg-like object without the llm_retry_* attributes (e.g. an older stub in a test)
        must not raise — getattr falls back to None and no override applies."""
        cfg = SimpleNamespace()
        profile = resolve_resilience("openai", "gpt-4o-mini", cfg)
        assert profile == DEFAULT_PROFILE
