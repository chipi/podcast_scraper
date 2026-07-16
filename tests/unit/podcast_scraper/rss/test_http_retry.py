"""Unit tests for the httpx ``RetryTransport`` (#1194).

Covers the urllib3-parity behaviour the downloader used to get from
``requests.Session`` + urllib3 ``Retry``:

- status-forcelist retries fire only for allowed methods (GET/HEAD/OPTIONS)
- non-forcelist statuses pass through untouched
- exponential backoff math + 120s cap
- ``Retry-After`` header overrides the computed backoff
- connection-level exceptions retry against the same budget
- ``total=0`` disables retries
- both callbacks (``on_retry`` counter, ``on_retry_after`` hook) fire the right number of times

All against ``httpx.MockTransport`` — no real network, no sleep.
"""

from __future__ import annotations

from typing import List, Union

import httpx
import pytest

from podcast_scraper.rss.http_retry import RetryTransport

pytestmark = pytest.mark.unit


_STATUS_FORCELIST = (408, 429, 500, 502, 503, 504)
_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@pytest.fixture(autouse=True)
def _skip_sleep(monkeypatch: pytest.MonkeyPatch):
    """Neuter ``time.sleep`` in the RetryTransport module so tests run fast."""
    monkeypatch.setattr("podcast_scraper.rss.http_retry.time.sleep", lambda _s: None)


def _make_transport(
    responses: List[Union[httpx.Response, BaseException]],
    *,
    total: int = 3,
    backoff_factor: float = 0.1,
    status_forcelist=_STATUS_FORCELIST,
    allowed_methods=_ALLOWED_METHODS,
):
    """Build a RetryTransport whose base pops the next response from the list."""
    retry_calls: List[None] = []
    retry_after_calls: List[tuple] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if not responses:
            raise AssertionError("MockTransport ran out of responses")
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    base = httpx.MockTransport(_handler)
    transport = RetryTransport(
        base,
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        on_retry=lambda: retry_calls.append(None),
        on_retry_after=lambda resp, url: retry_after_calls.append((resp.status_code, url)),
    )
    return transport, retry_calls, retry_after_calls


def _get(transport: RetryTransport, url: str = "https://example.com/") -> httpx.Response:
    """Drive one GET through the transport (bypasses Client so we test transport in isolation)."""
    req = httpx.Request("GET", url)
    return transport.handle_request(req)


# ---- status-forcelist behaviour --------------------------------------------


def test_transient_500_then_200_returns_200_and_counts_one_retry() -> None:
    """One retry on 500 → success on second try. Retry counter fires once."""
    transport, retries, _ = _make_transport(
        [httpx.Response(500), httpx.Response(200, text="ok")], total=3
    )
    resp = _get(transport)
    assert resp.status_code == 200
    assert resp.text == "ok"
    assert len(retries) == 1


def test_all_attempts_return_forcelisted_status_final_response_surfaces() -> None:
    """``raise_on_status=False`` parity: last attempt's response is returned
    even when the status is still on the forcelist. Retry counter fires ``total`` times.
    """
    transport, retries, _ = _make_transport([httpx.Response(503) for _ in range(5)], total=3)
    resp = _get(transport)
    assert resp.status_code == 503
    # total=3 → first call + 3 retries = 4 attempts total, retry counter fires 3 times
    assert len(retries) == 3


def test_non_forcelist_status_passes_through_without_retry() -> None:
    """404 is NOT in the forcelist → return immediately, no retry."""
    transport, retries, _ = _make_transport([httpx.Response(404), httpx.Response(200)], total=3)
    resp = _get(transport)
    assert resp.status_code == 404
    assert len(retries) == 0


@pytest.mark.parametrize("status", _STATUS_FORCELIST)
def test_every_forcelist_status_triggers_retry(status: int) -> None:
    """Explicit sweep of every status in the forcelist to guard against
    accidentally shrinking the set during a refactor.
    """
    transport, retries, _ = _make_transport([httpx.Response(status), httpx.Response(200)], total=2)
    resp = _get(transport)
    assert resp.status_code == 200
    assert len(retries) == 1


# ---- method allowlist --------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_allowed_methods_retry_on_forcelist_status(method: str) -> None:
    transport, retries, _ = _make_transport([httpx.Response(503), httpx.Response(200)], total=2)
    req = httpx.Request(method, "https://example.com/")
    resp = transport.handle_request(req)
    assert resp.status_code == 200
    assert len(retries) == 1


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_disallowed_methods_do_not_retry_on_forcelist_status(method: str) -> None:
    """POST/PUT/DELETE/PATCH surface the forcelist response after one call —
    urllib3 parity (allowed_methods=GET/HEAD/OPTIONS only).
    """
    transport, retries, _ = _make_transport([httpx.Response(503), httpx.Response(200)], total=3)
    req = httpx.Request(method, "https://example.com/")
    resp = transport.handle_request(req)
    assert resp.status_code == 503
    assert len(retries) == 0


# ---- connection-level errors -------------------------------------------


def test_connect_error_counts_toward_retry_budget() -> None:
    """Transient ConnectError → retry, then success."""
    transport, retries, _ = _make_transport(
        [httpx.ConnectError("boom"), httpx.Response(200)], total=3
    )
    resp = _get(transport)
    assert resp.status_code == 200
    assert len(retries) == 1


def test_read_error_counts_toward_retry_budget() -> None:
    transport, retries, _ = _make_transport([httpx.ReadError("boom"), httpx.Response(200)], total=3)
    resp = _get(transport)
    assert resp.status_code == 200
    assert len(retries) == 1


def test_timeout_error_counts_toward_retry_budget() -> None:
    transport, retries, _ = _make_transport(
        [httpx.ConnectTimeout("slow"), httpx.Response(200)], total=3
    )
    resp = _get(transport)
    assert resp.status_code == 200
    assert len(retries) == 1


def test_exhausted_retries_on_connect_error_raises() -> None:
    """After ``total`` retries all failing, the last exception is raised."""
    transport, retries, _ = _make_transport(
        [
            httpx.ConnectError("boom-1"),
            httpx.ConnectError("boom-2"),
            httpx.ConnectError("boom-3"),
            httpx.ConnectError("boom-4"),
        ],
        total=3,
    )
    with pytest.raises(httpx.ConnectError):
        _get(transport)
    # total=3 → 3 retries fired before the final raise
    assert len(retries) == 3


# ---- total=0 boundary --------------------------------------------------


def test_total_zero_disables_retries() -> None:
    """``total=0`` → one attempt only; forcelist status is returned."""
    transport, retries, _ = _make_transport([httpx.Response(503), httpx.Response(200)], total=0)
    resp = _get(transport)
    assert resp.status_code == 503
    assert len(retries) == 0


# ---- backoff math ------------------------------------------------------


def test_backoff_is_capped_at_120_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """``backoff_factor * 2 ** (attempt-1)`` never exceeds 120 s (urllib3 parity).

    Captures the actual sleep durations by recording every ``time.sleep`` call
    on the RetryTransport module namespace.
    """
    sleeps: List[float] = []
    monkeypatch.setattr(
        "podcast_scraper.rss.http_retry.time.sleep",
        lambda s: sleeps.append(s),
    )
    transport, _, _ = _make_transport(
        [httpx.Response(503) for _ in range(6)],
        total=5,
        backoff_factor=1000.0,  # blow the cap on every attempt
    )
    _get(transport)
    # Every recorded sleep must be at or below the cap.
    assert sleeps, "expected at least one retry sleep"
    assert all(s <= 120.0 for s in sleeps), f"sleeps exceeded cap: {sleeps}"


def test_zero_backoff_factor_means_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(
        "podcast_scraper.rss.http_retry.time.sleep",
        lambda s: sleeps.append(s),
    )
    transport, _, _ = _make_transport(
        [httpx.Response(503), httpx.Response(200)],
        total=3,
        backoff_factor=0.0,
    )
    _get(transport)
    assert sleeps == [] or all(s == 0 for s in sleeps)


# ---- Retry-After header -----------------------------------------------


def test_retry_after_numeric_seconds_overrides_computed_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(
        "podcast_scraper.rss.http_retry.time.sleep",
        lambda s: sleeps.append(s),
    )
    transport, retries, retry_after_hits = _make_transport(
        [
            httpx.Response(429, headers={"Retry-After": "42"}),
            httpx.Response(200),
        ],
        total=3,
        backoff_factor=1.0,
    )
    resp = _get(transport)
    assert resp.status_code == 200
    assert len(retries) == 1
    assert sleeps == [42.0], f"expected [42.0], got {sleeps}"
    # on_retry_after callback fires exactly once with the 429 response + URL.
    assert retry_after_hits == [(429, "https://example.com/")]


def test_retry_after_non_numeric_falls_back_to_computed_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP-date form (unsupported) → fall back to exponential backoff."""
    sleeps: List[float] = []
    monkeypatch.setattr(
        "podcast_scraper.rss.http_retry.time.sleep",
        lambda s: sleeps.append(s),
    )
    transport, _, _ = _make_transport(
        [
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
            httpx.Response(200),
        ],
        total=3,
        backoff_factor=0.5,
    )
    _get(transport)
    # Attempt 0 → backoff = 0.5 * 2**0 = 0.5s (not 0 for the HTTP-date fallback path).
    assert sleeps == [0.5], f"expected [0.5], got {sleeps}"


def test_retry_after_over_120_seconds_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-directed Retry-After bypasses the 120s cap (urllib3 parity)."""
    sleeps: List[float] = []
    monkeypatch.setattr(
        "podcast_scraper.rss.http_retry.time.sleep",
        lambda s: sleeps.append(s),
    )
    transport, _, _ = _make_transport(
        [
            httpx.Response(503, headers={"Retry-After": "300"}),
            httpx.Response(200),
        ],
        total=2,
    )
    _get(transport)
    assert sleeps == [300.0]


# ---- callback contract -------------------------------------------------


def test_on_retry_fires_exactly_once_per_retry() -> None:
    """3 forcelist responses + 1 OK = 3 on_retry calls."""
    transport, retries, _ = _make_transport(
        [
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200),
        ],
        total=5,
    )
    resp = _get(transport)
    assert resp.status_code == 200
    assert len(retries) == 3


def test_on_retry_after_only_fires_for_status_retries_not_exceptions() -> None:
    """The Retry-After hook is scoped to response-based retries. Exception
    retries (ConnectError etc.) do not have a response to inspect.
    """
    transport, _, retry_after_hits = _make_transport(
        [httpx.ConnectError("boom"), httpx.Response(200)], total=3
    )
    _get(transport)
    assert retry_after_hits == []


def test_on_retry_after_fires_on_forcelist_response_without_header() -> None:
    """The hook fires whenever we retry a status-forcelisted response — even
    when there's no ``Retry-After`` header. The caller decides what to do
    with the missing header (typically: nothing).
    """
    transport, _, retry_after_hits = _make_transport(
        [httpx.Response(503), httpx.Response(200)], total=3
    )
    _get(transport)
    assert len(retry_after_hits) == 1
    assert retry_after_hits[0] == (503, "https://example.com/")


# ---- close() delegates ------------------------------------------------


def test_close_delegates_to_base() -> None:
    """Closing the RetryTransport must close the wrapped transport too, so we
    don't leak the connection pool the factory built.
    """

    closed: List[bool] = []

    class RecordingBase(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(200)

        def close(self) -> None:
            closed.append(True)

    transport = RetryTransport(
        RecordingBase(),
        total=1,
        backoff_factor=0.0,
        status_forcelist=_STATUS_FORCELIST,
        allowed_methods=_ALLOWED_METHODS,
    )
    transport.close()
    assert closed == [True]
