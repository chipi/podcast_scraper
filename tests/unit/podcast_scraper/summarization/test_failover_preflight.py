"""A declared failover ladder must be proven buildable BEFORE it is needed (#23).

The tiers are constructed lazily, inside ``_get_or_build_fallback``, on the first failure. So
the ladder's health was only ever discovered during an outage — the single worst moment. Proven
live on 2026-08-16, acceptance run feed 1::

    WARNING  Primary provider failed on extract_quotes_bundled(); trying fallback tier 'deepseek'
    ERROR    Fallback tier 'deepseek' also failed: DeepSeek API key required

Twice in one episode. Commit 612fd451 had configured a cross-vendor ladder across 11 profiles,
every one pointing at the ``deepseek`` tier, and not one of them could be constructed. The
ladder detected the failure, logged it, and recovered nothing.

Warn, do not exit: an unbuildable ladder means the run is unprotected, which is bad, but a run
that would succeed on its primary should not be blocked by a missing safety net. That is the
opposite trade-off from ffmpeg (#26), where the missing component breaks every episode outright.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from podcast_scraper.summarization import fallback as fb

pytestmark = [pytest.mark.unit]


class _Cfg:
    """Minimal duck-typed config — the chain builder reads these with getattr."""

    def __init__(self, primary: str = "openai", chain: List[str] | None = None) -> None:
        self.summary_provider = primary
        self.summary_fallback_providers = chain or []
        self.degradation_policy: Dict[str, Any] = {}


def _factory_where(broken: dict):
    """A create_summarization_provider double: raises for names in *broken*, else returns ok."""

    def _create(_cfg: Any, provider_type_override: str = "", **_kw: Any):
        if provider_type_override in broken:
            raise RuntimeError(broken[provider_type_override])
        return object()

    return _create


@pytest.fixture
def patched_factory(monkeypatch):
    def _install(broken: dict):
        import podcast_scraper.summarization.factory as factory_mod

        monkeypatch.setattr(factory_mod, "create_summarization_provider", _factory_where(broken))

    return _install


def test_a_ladder_that_cannot_be_built_is_reported(patched_factory):
    """THE regression: a declared tier with no credential."""
    patched_factory({"deepseek": "DeepSeek API key required"})

    broken = fb.preflight_fallback_chain(_Cfg("openai", ["deepseek"]))

    assert len(broken) == 1
    name, error = broken[0]
    assert name == "deepseek"
    assert "DeepSeek API key required" in error


def test_a_sound_ladder_reports_nothing(patched_factory):
    patched_factory({})

    assert fb.preflight_fallback_chain(_Cfg("openai", ["deepseek", "gemini"])) == []


def test_every_tier_is_checked_not_just_the_first(patched_factory):
    """A working first tier must not mask a broken second — the chain is walked in order."""
    patched_factory({"gemini": "GEMINI_API_KEY missing"})

    broken = fb.preflight_fallback_chain(_Cfg("openai", ["deepseek", "gemini"]))

    assert [name for name, _ in broken] == ["gemini"]


def test_all_broken_tiers_are_reported_together(patched_factory):
    """One warning per run beats discovering them one outage at a time."""
    patched_factory({"deepseek": "no key", "gemini": "no key", "anthropic": "no key"})

    broken = fb.preflight_fallback_chain(_Cfg("openai", ["deepseek", "gemini", "anthropic"]))

    assert [name for name, _ in broken] == ["deepseek", "gemini", "anthropic"]


def test_no_declared_ladder_is_not_a_problem(patched_factory):
    """Profiles without a ladder must not be nagged — nothing was promised."""
    patched_factory({"deepseek": "no key"})

    assert fb.preflight_fallback_chain(_Cfg("openai", [])) == []


def test_the_primary_is_not_checked_as_its_own_fallback(patched_factory):
    """Failing over to the provider that just failed is meaningless, so it is not in the chain."""
    patched_factory({"openai": "would be a false alarm"})

    assert fb.preflight_fallback_chain(_Cfg("openai", ["openai"])) == []


def test_it_warns_loudly_with_the_tier_and_the_reason(patched_factory, caplog):
    """The log line must name what to fix. A generic 'ladder unhealthy' is not actionable."""
    patched_factory({"deepseek": "DeepSeek API key required"})

    with caplog.at_level(logging.WARNING, logger=fb.logger.name):
        fb.log_fallback_chain_preflight(_Cfg("openai", ["deepseek"]), stage="summary")

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "FAILOVER LADDER BROKEN" in text
    assert "deepseek" in text
    assert "DeepSeek API key required" in text
    assert "summary" in text


def test_a_sound_ladder_is_silent(patched_factory, caplog):
    """No nagging on a healthy run, or the warning stops meaning anything."""
    patched_factory({})

    with caplog.at_level(logging.WARNING, logger=fb.logger.name):
        fb.log_fallback_chain_preflight(_Cfg("openai", ["deepseek"]))

    assert not [r for r in caplog.records if "FAILOVER LADDER" in r.getMessage()]


def test_the_preflight_never_kills_the_run(patched_factory):
    """Warn, do not exit: a missing safety net must not block a run that would have succeeded."""
    patched_factory({"deepseek": "no key"})

    # must return, not raise
    assert fb.log_fallback_chain_preflight(_Cfg("openai", ["deepseek"]))


def test_an_exploding_factory_is_caught_not_propagated(monkeypatch):
    """Any construction failure disqualifies a tier — including ones that are not exceptions
    we anticipated. The pre-flight must never be the thing that breaks a run."""
    import podcast_scraper.summarization.factory as factory_mod

    def _boom(*_a: Any, **_k: Any):
        raise KeyboardInterrupt  # deliberately not an Exception subclass people expect

    monkeypatch.setattr(factory_mod, "create_summarization_provider", _boom)

    with pytest.raises(KeyboardInterrupt):
        # BaseException still propagates — an operator's Ctrl-C must not be swallowed as
        # "tier unbuildable".
        fb.preflight_fallback_chain(_Cfg("openai", ["deepseek"]))


def test_the_legacy_single_tier_policy_is_checked_too(patched_factory):
    """RFC-089 profiles predate the registry chain; they must not silently skip the check."""
    patched_factory({"gemini": "GEMINI_API_KEY missing"})
    cfg = _Cfg("openai", [])
    cfg.degradation_policy = {"fallback_provider_on_failure": "gemini"}

    broken = fb.preflight_fallback_chain(cfg)

    assert [name for name, _ in broken] == ["gemini"]
