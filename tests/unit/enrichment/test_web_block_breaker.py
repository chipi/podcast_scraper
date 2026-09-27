"""Block-aware circuit breaker for the web enrichers (#2163).

The gap this closes: on 2026-09-16 and again on 2026-09-27 Wikimedia 403-blocked the egress IP and
NOTHING detected it. 403 is not in the retry forcelist, so each call surfaced as a
``TransientFetchError`` — which the per-entity loop swallows and skips. A 200-entity pass would
walk its entire budget into a wall at ~1 req/s and report ``status: ok`` with zero rows fetched.

Everything here drives ``httpx.MockTransport``. Nothing contacts Wikimedia; nothing runs real
enrichment.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _BLOCK_TRIP_AFTER,
    ClusterRateLimiter,
    set_web_block_breaker,
    web_block_breaker,
    WebBlockBreaker,
    WikipediaProvider,
)
from podcast_scraper.enrichment.resilience import (
    classify_failure,
    RetryClass,
    UpstreamBlockedError,
)

BLOCK_BODY = (
    "Please respect our robot policy https://w.wiki/4wJS when crawling us. "
    "Contact bot-traffic@wikimedia.org if you need higher volumes."
)


@pytest.fixture(autouse=True)
def _fresh_breaker():
    set_web_block_breaker(None)
    yield
    set_web_block_breaker(None)


def _provider(handler) -> WikipediaProvider:
    """A provider on a mock transport with a zero-interval limiter (no real sleeping)."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WikipediaProvider(client=client, limiter=ClusterRateLimiter(0.0))


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def test_a_recognised_block_body_trips_immediately(caplog: pytest.LogCaptureFixture) -> None:
    """The body is the upstream telling us in words — no need to wait for a streak."""
    breaker = WebBlockBreaker()

    with caplog.at_level(logging.ERROR):
        breaker.note_response(403, BLOCK_BODY, "https://www.wikidata.org/w/api.php")

    assert breaker.is_open
    assert "robot policy" in (breaker.reason or "")
    assert "STOPPING" in caplog.text


def test_an_unrecognised_403_needs_a_streak() -> None:
    """One 403 can be a single bad URL; three in a row is a block."""
    breaker = WebBlockBreaker()

    for _ in range(_BLOCK_TRIP_AFTER - 1):
        breaker.note_response(403, "weird", "https://www.wikidata.org/w/api.php")
        assert not breaker.is_open

    breaker.note_response(403, "weird", "https://www.wikidata.org/w/api.php")
    assert breaker.is_open


def test_a_success_between_403s_resets_the_streak() -> None:
    """Intermittent 403s are not a block, and must not abort a healthy run."""
    breaker = WebBlockBreaker()

    for _ in range(10):
        breaker.note_response(403, "weird", "https://www.wikidata.org/w/api.php")
        breaker.note_success()

    assert not breaker.is_open


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_throttling_and_5xx_never_trip_the_breaker(status: int) -> None:
    """429 is ordinary throttling handled by the limiter + RetryTransport.

    If these counted, a normal rate-limited run would abort itself.
    """
    breaker = WebBlockBreaker()

    for _ in range(20):
        breaker.note_response(status, "Too Many Requests", "https://www.wikidata.org/w/api.php")

    assert not breaker.is_open


def test_a_stray_success_does_not_reopen_a_latched_breaker() -> None:
    """A cached edge response mid-block must not undo the decision to stop."""
    breaker = WebBlockBreaker()
    breaker.note_response(403, BLOCK_BODY, "https://www.wikidata.org/w/api.php")
    assert breaker.is_open

    breaker.note_success()

    assert breaker.is_open


# --------------------------------------------------------------------------- #
# run scoping
# --------------------------------------------------------------------------- #


def test_a_new_run_clears_the_breaker() -> None:
    """Latching per PROCESS would be worse than the bug: the API container is long-lived, so one
    block would fail every future run until a restart."""
    breaker = WebBlockBreaker()
    breaker.reset_for_run("run-1")
    breaker.note_response(403, BLOCK_BODY, "https://www.wikidata.org/w/api.php")
    assert breaker.is_open

    breaker.reset_for_run("run-2")

    assert not breaker.is_open
    assert breaker.reason is None


def test_reset_is_idempotent_within_one_run() -> None:
    """Both enrichers reset with the same run_id; the second must not wipe the first's trip."""
    breaker = WebBlockBreaker()
    breaker.reset_for_run("run-1")
    breaker.note_response(403, BLOCK_BODY, "https://www.wikidata.org/w/api.php")

    breaker.reset_for_run("run-1")  # org_web starting after person_web already tripped

    assert breaker.is_open


def test_both_enrichers_share_one_breaker() -> None:
    """A block is per-IP, so org_web must stop when person_web is blocked."""
    assert web_block_breaker() is web_block_breaker()


# --------------------------------------------------------------------------- #
# provider behaviour
# --------------------------------------------------------------------------- #


def test_provider_raises_blocked_and_stops_issuing_requests() -> None:
    """THE POINT: after the block is detected, no further requests go out."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(403, text=BLOCK_BODY)

    p = _provider(handler)

    with pytest.raises(UpstreamBlockedError):
        p.fetch_raw("a", "A")
    first = len(calls)

    # Every later entity fails instantly, WITHOUT touching the network.
    for name in ("B", "C", "D", "E"):
        with pytest.raises(UpstreamBlockedError):
            p.fetch_raw(name.lower(), name)

    assert len(calls) == first == 1, f"expected exactly one request, got {calls}"


def test_blocked_is_not_a_transient_fetch_error() -> None:
    """The per-entity loop swallows TransientFetchError and continues.

    If the block raised that type, the run would silently skip all 200 entities and report ok —
    which is exactly what happened twice. The type must be distinct for the run to abort.
    """
    from podcast_scraper.enrichment.enrichers.person_web import TransientFetchError

    assert not issubclass(UpstreamBlockedError, TransientFetchError)


def test_a_single_403_still_raises_transient_not_blocked() -> None:
    """Below the streak threshold the old behaviour holds: transient, caller records nothing."""
    from podcast_scraper.enrichment.enrichers.person_web import TransientFetchError

    p = _provider(lambda req: httpx.Response(403, text="odd"))

    with pytest.raises(TransientFetchError):
        p.fetch_raw("a", "A")


def test_404_counts_as_the_upstream_answering() -> None:
    """A 404 is an authoritative answer, so it proves we are not blocked."""
    p = _provider(lambda req: httpx.Response(404, json={"detail": "not found"}))
    breaker = web_block_breaker()
    breaker.note_response(403, "weird", "https://x/1")
    breaker.note_response(403, "weird", "https://x/2")

    assert p.fetch_raw("nobody", "Nobody") is None
    # The streak was cleared, so the next lone 403 cannot trip it.
    breaker.note_response(403, "weird", "https://x/3")
    assert not breaker.is_open


# --------------------------------------------------------------------------- #
# executor contract
# --------------------------------------------------------------------------- #


def test_blocked_is_classified_non_retryable() -> None:
    """Retrying is the wrong response to being blocked — and it escalates the block."""
    assert classify_failure(UpstreamBlockedError("blocked")) is RetryClass.NON_RETRYABLE
