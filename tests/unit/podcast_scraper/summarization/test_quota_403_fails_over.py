"""#1634-#1639 — an over-quota 403 must fail over, not hard-stop the run.

Six glitchtip issues, one incident: on 2026-08-13 23:09-23:12 the OpenRouter key passed its
weekly limit and returned 403 ``Key limit exceeded (weekly limit)``. litellm surfaced it, the
taxonomy classified it TERMINAL, and the run hard-stopped. It surfaced from summarisation
(#1634/#1635/#1636/#1639) and from transcript cleaning (#1637/#1638).

Two separate mechanisms produced that outcome, and only one of them is fixed:

1. HALF THE LLM SURFACE HAD NO LADDER. The fallback wrapper used an allowlist of eight method
   names, so ``clean_transcript``, ``generate_insights``, ``detect_speakers`` and five others
   were forwarded straight to the broken primary while summarisation quietly failed over. That
   is fixed — the allowlist became ``_NEVER_WRAPPED``, a denylist, in f6c77fcd (2026-08-17).
   ``test_failover_covers_every_llm_call.py`` pins it. The incident build, sha-1c6b3de, is dated
   2026-08-11 and therefore predates the fix.

2. THE CLASSIFIER STILL READS 403 AS AN ACCESS FAILURE. ``classify_llm_error`` matches none of
   ``_TERMINAL_SIGNALS`` against "Key limit exceeded (weekly limit)"; it reaches
   ``if _auth_status(status)`` and returns TERMINAL because OpenRouter uses 403 for a BUDGET
   condition. That is unchanged, and it is correct only because TERMINAL means "do not retry" —
   it must NOT also mean "do not fail over".

What no test covered until now is the two together: the REAL provider payload, through the REAL
taxonomy, into the REAL ladder. The existing failover tests raise a bare ``RuntimeError``, which
would still pass if the taxonomy started short-circuiting the ladder. These tests use the
verbatim 403 body from the #1636 event so a regression in either half is caught here.

Operator policy (2026-08-18): fail over to the next tier; hard-stop ONLY when every configured
tier is exhausted.
"""

# mypy: disable-error-code="arg-type"
# _Cfg is a deliberate stand-in; the provider only ever getattr()s off it.

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.summarization.fallback import FallbackAwareSummarizationProvider
from podcast_scraper.utils.llm_error_taxonomy import (
    classify_llm_error,
    LLMErrorClass,
    LLMTerminalError,
    terminal_message,
)

pytestmark = pytest.mark.unit


# The verbatim body from the #1636 / #1639 glitchtip events.
OPENROUTER_WEEKLY_LIMIT_403 = (
    "Error code: 403 - {'error': {'message': 'litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Key limit exceeded (weekly limit). Manage it using '
    'https://openrouter.ai/workspaces/podcast","code":403}}\'}}'
)


class _QuotaError(RuntimeError):
    """What the provider SDK raises: a 403 carrying the quota body."""

    def __init__(self) -> None:
        super().__init__(OPENROUTER_WEEKLY_LIMIT_403)
        self.status_code = 403


class TestTheTaxonomyStillCallsThisTerminal:
    """Pins the classification itself, so a change to it is a deliberate act."""

    def test_the_weekly_limit_403_is_terminal(self) -> None:
        assert classify_llm_error(_QuotaError()) is LLMErrorClass.TERMINAL

    def test_it_is_terminal_because_of_the_STATUS_not_the_message(self) -> None:
        """The distinction that matters: no signal string matches — only the 403 does.

        If someone adds "key limit exceeded" to _TERMINAL_SIGNALS this still passes, but the
        message-only case below documents which half is load-bearing today.
        """

        class _MessageOnly(RuntimeError):
            pass  # same text, no status_code attribute and no "403" token

        message_only = _MessageOnly("Key limit exceeded (weekly limit).")
        assert classify_llm_error(message_only) is not LLMErrorClass.TERMINAL

    def test_a_revoked_key_stays_terminal(self) -> None:
        """The genuine access failure must not be softened by any quota handling."""

        class _Revoked(RuntimeError):
            def __init__(self) -> None:
                super().__init__("Error code: 401 - Incorrect API key provided")
                self.status_code = 401

        assert classify_llm_error(_Revoked()) is LLMErrorClass.TERMINAL

    def test_the_operator_message_names_the_budget_not_the_key(self) -> None:
        msg = terminal_message("litellm", _QuotaError())
        assert "no budget/credit left on this key" in msg


class _Cfg:
    def __init__(self, chain: List[str]) -> None:
        self.summary_provider = "litellm"
        self.speaker_detector_provider = "litellm"
        self.summary_fallback_providers = list(chain)
        self.degradation_policy: Dict[str, Any] = {}


class _OverQuotaPrimary:
    """Every LLM call raises the terminal error the retry layer produces for a 403.

    provider_metrics.py:632 converts the classified 403 into LLMTerminalError before the ladder
    ever sees it, so that — not the raw SDK error — is what the wrapper must cope with.
    """

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        def boom(*_a: Any, **_k: Any) -> Any:
            self.calls.append(name)
            raise LLMTerminalError(terminal_message("litellm", _QuotaError()))

        return boom

    def initialize(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


class _HealthyTier:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        def ok(*_a: Any, **_k: Any) -> str:
            self.calls.append(name)
            return f"{name}-from-deepseek"

        return ok

    def initialize(self) -> None:
        return None


# The stages the six issues actually named, plus the ones that were unprotected alongside them.
STAGES = [
    "summarize",
    "clean_transcript",
    "generate_insights",
    "detect_speakers",
    "extract_kg_graph",
    "classify_insights",
]


class TestAnOverQuotaProviderFailsOverInsteadOfStopping:
    @staticmethod
    def _wrapped(monkeypatch: pytest.MonkeyPatch) -> Any:
        primary, tier = _OverQuotaPrimary(), _HealthyTier()
        w = FallbackAwareSummarizationProvider(primary, ["deepseek"], _Cfg(["deepseek"]))
        monkeypatch.setattr(w, "_get_or_build_fallback", lambda _n: tier)
        w._primary_probe, w._tier_probe = primary, tier  # type: ignore[attr-defined]
        return w

    @pytest.mark.parametrize("stage", STAGES)
    def test_the_stage_completes_on_the_next_tier(
        self, monkeypatch: pytest.MonkeyPatch, stage: str
    ) -> None:
        """A terminal quota error is not allowed to end the run while a tier is left."""
        w = self._wrapped(monkeypatch)
        assert getattr(w, stage)("x") == f"{stage}-from-deepseek"
        assert w._primary_probe.calls == [stage], "the primary must be tried first"
        assert w._tier_probe.calls == [stage], "the fallback tier must actually run"

    def test_cleaning_specifically_fails_over(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """#1637/#1638 died here: clean_transcript was outside the old allowlist."""
        w = self._wrapped(monkeypatch)
        assert w.clean_transcript("transcript text") == "clean_transcript-from-deepseek"

    def test_the_run_stops_only_once_every_tier_is_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operator policy: hard-stop is the LAST resort, not the first response."""
        primary = _OverQuotaPrimary()
        w = FallbackAwareSummarizationProvider(primary, ["deepseek"], _Cfg(["deepseek"]))
        monkeypatch.setattr(w, "_get_or_build_fallback", lambda _n: _OverQuotaPrimary())

        with pytest.raises(LLMTerminalError) as excinfo:
            w.summarize("x")
        # The operator must still be told WHY the run stopped.
        assert "no budget/credit left on this key" in str(excinfo.value)

    def test_a_healthy_second_tier_rescues_a_dead_first_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two tiers, first also over quota: the ladder must keep walking."""
        healthy = _HealthyTier()
        built: List[str] = []

        def _build(name: str) -> Any:
            built.append(name)
            return _OverQuotaPrimary() if name == "openai" else healthy

        w = FallbackAwareSummarizationProvider(
            _OverQuotaPrimary(), ["openai", "deepseek"], _Cfg(["openai", "deepseek"])
        )
        monkeypatch.setattr(w, "_get_or_build_fallback", _build)

        assert w.summarize("x") == "summarize-from-deepseek"
        assert built == ["openai", "deepseek"], "tiers must be tried in configured order"
