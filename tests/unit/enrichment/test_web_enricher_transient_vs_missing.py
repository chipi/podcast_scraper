"""A failed fetch must never become a negative-cache miss.

On 2026-09-16 the web enrichers issued ~911 requests in 65 seconds (~43/s), Wikimedia's
robot policy blocked every one with 403, and because ``_get_json`` collapsed *every*
failure to ``None``, the caller recorded all 911 entities — Keir Starmer, Katie Couric,
Allbirds — as authoritative misses with a 30-day TTL. A transient block silently became a
month of missing data that would then read as "nothing left to enrich".

Two properties are load-bearing and tested here:

1. **404 means absent** → ``fetch_raw`` returns ``None`` → caller may record a miss.
2. **Anything else means unknown** → ``TransientFetchError`` → caller records NOTHING.

Plus the throttle that prevents the block in the first place. No network: every test
drives an ``httpx.MockTransport``.
"""

from __future__ import annotations

import httpx
import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _WEB_MIN_INTERVAL_S,
    TransientFetchError,
    WikipediaProvider,
)


def _provider(handler) -> WikipediaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WikipediaProvider(client=client)


class TestAbsentVersusUnknown:
    def test_404_returns_none_so_the_caller_may_record_a_miss(self):
        """The one case where a miss is legitimate: the source looked and had nothing."""
        p = _provider(lambda req: httpx.Response(404, json={"detail": "not found"}))

        assert p.fetch_raw("nobody", "Nobody At All") is None

    @pytest.mark.parametrize("status", [403, 429, 500, 502, 503])
    def test_error_status_raises_rather_than_returning_none(self, status):
        """403 is the exact status that caused the incident. 429/5xx are the same class:
        we could not ask, so we know nothing about this entity."""
        p = _provider(lambda req: httpx.Response(status, text="blocked"))

        with pytest.raises(TransientFetchError):
            p.fetch_raw("keir-starmer", "Keir Starmer")

    def test_connection_error_raises_rather_than_returning_none(self):
        def boom(req):
            raise httpx.ConnectError("no route to host")

        with pytest.raises(TransientFetchError):
            _provider(boom).fetch_raw("katie-couric", "Katie Couric")

    def test_unparsable_body_raises_rather_than_returning_none(self):
        """A 200 with garbage is still 'we don't know', not 'there is nothing'."""
        p = _provider(lambda req: httpx.Response(200, text="<html>not json</html>"))

        with pytest.raises(TransientFetchError):
            p.fetch_raw("allbirds", "Allbirds")

    def test_a_real_hit_still_works(self):
        """Guard against the fix breaking the happy path."""
        p = _provider(lambda req: httpx.Response(200, json={"title": "Katie Couric"}))

        assert p.fetch_raw("katie-couric", "Katie Couric") == {"title": "Katie Couric"}


class TestThrottle:
    """~43 req/s is what got us blocked. The limiter is what keeps us under the policy."""

    def test_consecutive_requests_are_spaced(self):
        slept: list[float] = []
        p = _provider(lambda req: httpx.Response(200, json={"ok": True}))
        p._limiter._sleep = slept.append  # type: ignore[attr-defined]
        clock = iter([0.0, 0.0, 0.01, 0.01, 0.02, 0.02])
        p._limiter._clock = lambda: next(clock)  # type: ignore[attr-defined]

        p.fetch_raw("a", "A")
        p.fetch_raw("b", "B")

        assert slept, "second request to the same host must be delayed"
        assert slept[0] <= _WEB_MIN_INTERVAL_S

    def test_interval_is_at_least_one_second(self):
        """Wikimedia's policy is the constraint; do not quietly lower this."""
        assert _WEB_MIN_INTERVAL_S >= 1.0
