"""E2E: per-cloud-provider circuit breaker trips on a burst of 503s.

ADR-100 follow-up — cloud LLM providers were previously relying on the
cloud SDK's own retry + the FallbackAware swap; nothing tracked cross-call
upstream-overload state. The ``LLMCircuitBreakerConfig`` substrate (per
#697) has existed for a while but no provider had it wired. This test
proves the wiring: when ``llm_circuit_breaker_enabled=True`` and a burst
of 503s exceeds the failure threshold, subsequent calls within the
cooldown window sleep through the cooldown (wait-and-resume semantics,
NOT raise-and-fail-fast).

Single test case is enough to prove the wiring works through the real
SDK; per-provider replication would just verify the same retry decorator
fires identically (which it does, by construction). Repeat per-provider
in a follow-up if a single provider's SDK proves to surface different
exception shapes.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.e2e]

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config  # noqa: E402
from podcast_scraper.exceptions import ProviderRuntimeError  # noqa: E402
from podcast_scraper.utils import llm_circuit_breaker  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state(e2e_server):
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_all_error_behaviors()
    llm_circuit_breaker.reset_for_test()
    yield
    E2EHTTPRequestHandler.clear_all_error_behaviors()
    llm_circuit_breaker.reset_for_test()


@pytest.fixture(autouse=True)
def _skip_retry_backoff():
    """Skip retry sleeps and breaker cooldowns so the test stays fast."""
    with (
        patch("podcast_scraper.utils.provider_metrics.time.sleep"),
        patch("podcast_scraper.utils.llm_circuit_breaker.time.sleep"),
    ):
        yield


def _openai_provider_with_breaker(e2e_server):
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "openai_api_key": "sk-test",
            "openai_api_base": e2e_server.urls.openai_api_base(),
            "summary_provider": "openai",
            "generate_summaries": True,
            "llm_circuit_breaker_enabled": True,
            "llm_circuit_breaker_failure_threshold": 3,
            "llm_circuit_breaker_window_seconds": 60.0,
            "llm_circuit_breaker_cooldown_seconds": 1.0,
        }
    )
    p = OpenAIProvider(cfg)
    p.initialize()
    return p


class TestPerProviderCircuitBreakerE2E:
    def test_breaker_trips_after_burst_of_503s(self, e2e_server):
        """Burst of 503s ≥ failure_threshold within the window trips the
        breaker; ``stats()`` reflects ``trips_total >= 1``."""
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        # Permanent 503 — each summarize call exhausts retries, records the
        # failures into the breaker, and eventually surfaces as
        # ProviderRuntimeError. After enough calls (threshold=3 here), the
        # breaker trips and subsequent calls wait the cooldown.
        E2EHTTPRequestHandler.set_error_behavior("/v1/chat/completions", status=503)
        provider = _openai_provider_with_breaker(e2e_server)

        # Pre-condition: breaker has no recorded trips.
        assert llm_circuit_breaker.stats("openai")["trips_total"] == 0

        # Make calls until the breaker observably trips. Each summarize
        # internally retries 3x, so each call records up to 3 failures into
        # the breaker; with threshold=3 the first call alone should trip it.
        for _ in range(2):
            with pytest.raises(ProviderRuntimeError):
                provider.summarize("Sample transcript for circuit-breaker E2E. " * 4)

        # Post-condition: breaker has tripped at least once.
        s = llm_circuit_breaker.stats("openai")
        assert s["trips_total"] >= 1
        assert s["cooldown_seconds_total"] > 0

    def test_breaker_disabled_by_default(self, e2e_server):
        """Without ``llm_circuit_breaker_enabled=True``, the breaker stays
        unwired — the existing soft retry/backoff path is unchanged."""
        from podcast_scraper.providers.openai.openai_provider import OpenAIProvider
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior("/v1/chat/completions", status=503)
        cfg = Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "openai_api_key": "sk-test",
                "openai_api_base": e2e_server.urls.openai_api_base(),
                "summary_provider": "openai",
                "generate_summaries": True,
                # breaker NOT enabled
            }
        )
        p = OpenAIProvider(cfg)
        p.initialize()
        with pytest.raises(ProviderRuntimeError):
            p.summarize("Sample transcript. " * 4)

        # No trips recorded — breaker wasn't wired.
        assert llm_circuit_breaker.stats("openai")["trips_total"] == 0
