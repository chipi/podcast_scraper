"""Cluster-keyed shared throttle + throttle forensics for the web enrichers (#2163).

The bug these guard: ``_WEB_MIN_INTERVAL_S = 1.0`` was honoured per ``(host, provider instance)``.
``person_web`` and ``org_web`` run concurrently (``EnricherTier.WEB`` has ``concurrency=2``) and
each built its own ``HostRateLimiter``, while the limiter keyed on hostname — so wikidata,
wikipedia and commons each drew an independent 1/s budget, twice over. Wikimedia meters those
hosts as ONE per-IP bucket at its text edge, which is what produced 118 × 429 in a 76-minute run
and, twice, an outright 403 block of the egress IP.

Every case here uses a fake clock + fake sleep, or ``httpx.MockTransport``. NOTHING contacts
Wikimedia and nothing runs enrichment — that constraint is the whole point of this file.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _log_retry_after,
    _WIKIMEDIA_TEXT_BUCKET,
    ClusterRateLimiter,
    rate_limit_bucket,
    set_shared_web_limiter,
    shared_web_limiter,
)


class FakeClock:
    """Monotonic clock advanced only by the limiter's own sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _reset_shared_limiter() -> Any:
    """Never leak a limiter between tests — it is process-wide by design."""
    set_shared_web_limiter(None)
    yield
    set_shared_web_limiter(None)


# --------------------------------------------------------------------------- #
# bucket mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "https://www.wikidata.org/w/api.php?action=wbsearchentities",
        "https://en.wikipedia.org/api/rest_v1/page/summary/Foo",
        "https://commons.wikimedia.org/w/api.php?action=query",
        "https://commons.wikimedia.org/wiki/Special:FilePath/x.svg",
    ],
)
def test_text_edge_hosts_share_one_bucket(url: str) -> None:
    """The three hosts that 429'd are metered together, because the edge meters them together."""
    assert rate_limit_bucket(url) == _WIKIMEDIA_TEXT_BUCKET


def test_upload_cluster_keeps_its_own_bucket() -> None:
    """``upload.wikimedia.org`` carried sustained image traffic with zero 429s — a separate
    cluster, so it must not be starved behind the text bucket."""
    assert rate_limit_bucket("https://upload.wikimedia.org/wikipedia/commons/3/3f/x.svg") != (
        _WIKIMEDIA_TEXT_BUCKET
    )


def test_unrelated_hosts_are_still_metered_per_host() -> None:
    assert rate_limit_bucket("https://example.org/a") == "example.org"
    assert rate_limit_bucket("https://other.test/b") == "other.test"


# --------------------------------------------------------------------------- #
# the pacing guarantee
# --------------------------------------------------------------------------- #


def test_calls_across_different_text_hosts_are_spaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE BUG: three different text hosts used to emit 3 req/s while "honouring" 1 req/s each."""
    clock = FakeClock()
    limiter = ClusterRateLimiter(1.0, sleep=clock.sleep, clock=clock)

    limiter.wait("https://www.wikidata.org/w/api.php?a=1")
    limiter.wait("https://en.wikipedia.org/api/rest_v1/page/summary/Foo")
    limiter.wait("https://commons.wikimedia.org/w/api.php?a=2")

    # First call is free; the next two must each have waited ~1s.
    assert len(clock.sleeps) == 2, f"expected 2 waits, got {clock.sleeps}"
    assert all(s == pytest.approx(1.0, abs=0.01) for s in clock.sleeps)


def test_two_provider_instances_share_the_process_limiter() -> None:
    """Two enrichers must not each get their own budget — that doubled the emitted rate."""
    assert shared_web_limiter() is shared_web_limiter()

    clock = FakeClock()
    injected = ClusterRateLimiter(1.0, sleep=clock.sleep, clock=clock)
    set_shared_web_limiter(injected)

    # Stand-ins for the two enrichers, both resolving the limiter the production way.
    person_side = shared_web_limiter()
    org_side = shared_web_limiter()
    assert person_side is org_side is injected

    person_side.wait("https://www.wikidata.org/w/api.php?a=1")
    org_side.wait("https://www.wikidata.org/w/api.php?a=2")

    assert clock.sleeps == [pytest.approx(1.0, abs=0.01)]


def test_upload_bucket_does_not_wait_behind_the_text_bucket() -> None:
    """Image bytes must not be throttled by unrelated API chatter."""
    clock = FakeClock()
    limiter = ClusterRateLimiter(1.0, sleep=clock.sleep, clock=clock)

    limiter.wait("https://www.wikidata.org/w/api.php?a=1")
    limiter.wait("https://upload.wikimedia.org/wikipedia/commons/3/3f/x.svg")

    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# throttle forensics
# --------------------------------------------------------------------------- #


def test_retry_after_hook_logs_the_evidence(caplog: pytest.LogCaptureFixture) -> None:
    """A 76-minute run logged 118 throttles and recorded NONE of their Retry-After values.

    Diagnosis had to infer the wait from timestamp gaps, and inferred it wrong twice.
    """
    response = httpx.Response(
        429,
        headers={
            "Retry-After": "55",
            "x-cache": "cp3073 int",
            "server": "HAProxy",
        },
        text="Please respect our robot policy https://w.wiki/4wJS when crawling us.",
        request=httpx.Request("GET", "https://www.wikidata.org/w/api.php?action=wbgetentities"),
    )

    with caplog.at_level(logging.WARNING):
        _log_retry_after(response, "https://www.wikidata.org/w/api.php?action=wbgetentities")

    assert "retry_after='55'" in caplog.text
    assert "cp3073 int" in caplog.text
    assert "HAProxy" in caplog.text
    assert "robot policy" in caplog.text


def test_retry_after_hook_survives_an_unreadable_body(caplog: pytest.LogCaptureFixture) -> None:
    """Logging must never break the retry path — the hook runs inside the transport."""

    response = httpx.Response(
        503,
        headers={"Retry-After": "2"},
        request=httpx.Request("GET", "https://en.wikipedia.org/api/rest_v1/page/summary/Foo"),
    )

    def _boom() -> bytes:
        raise RuntimeError("stream gone")

    # Patched on the instance AFTER construction: httpx.Response.__init__ reads the body itself,
    # so a subclass override would explode before the hook under test ever runs.
    setattr(response, "read", _boom)

    with caplog.at_level(logging.WARNING):
        _log_retry_after(response, "https://en.wikipedia.org/api/rest_v1/page/summary/Foo")

    assert "retry_after='2'" in caplog.text
    assert "<unreadable>" in caplog.text


def test_client_wires_the_forensic_hook_into_the_transport() -> None:
    """Regression guard: the hook is useless unless ``_build_web_client`` actually passes it.

    ``RetryTransport``'s default is a silent no-op, which is exactly how the evidence was lost.
    """
    from podcast_scraper.enrichment.enrichers.person_web import _build_web_client

    client = _build_web_client()
    try:
        transport = client._transport
        assert getattr(transport, "_on_retry_after", None) is _log_retry_after
    finally:
        client.close()
