"""The re-check's page fetch (CRFC-006 Part B, B.2): a fake network serving two versions of a page
per URL, as B.5's proof names (same page, whitespace change, edit, 404, paywall appears), plus the
answers that must not count as gone (a timeout, a 5xx, robots.txt) and the conditional GET.

The extractor is faked (a page's ``<p>`` text): it is a ``[news]`` extra, and what is under test
here is the fetch and how its answers reach the compare, not the extraction."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import pytest

from podcast_scraper.news import extract, recheck, recheck_fetch
from podcast_scraper.rss import downloader

pytestmark = pytest.mark.unit

URL = "https://news.test/world/2026/nov/24/bridge"
SOURCE = SimpleNamespace(user_agent="close-listening-test")
_P = [
    "The council voted on Tuesday to close the bridge for repairs from next month, after a "
    "survey found the deck could no longer carry buses safely through the winter.",
    "Engineers said the steel showed fatigue cracks that could not wait for spring, and that "
    "the repairs would take at least eleven weeks if the weather allowed work at night.",
    "Residents will be offered a free shuttle across the river while the work runs, the "
    "council said, with buses every ten minutes at peak hours and every half hour otherwise.",
    "Shopkeepers on the north bank said they feared losing trade, and asked the council to "
    "publish the timetable for the works before the end of the month so they could plan.",
]


def _page(paragraphs: List[str], *, spacing: str = " ") -> str:
    body = "".join(f"<p>{spacing.join(p.split(' '))}</p>" for p in paragraphs)
    return (
        "<html lang='en'><head><title>Bridge to close</title></head><body><article>"
        f"<h1>Bridge to close for repairs</h1>{body}</article></body></html>"
    )


ORIGINAL = _page(_P)


def _fake_main_text(
    html: str, url: str, feed_html: Optional[str] = None, *, text_from: str = "page"
) -> Optional[extract.Extracted]:
    assert text_from == "page", "the source's own text_from reaches the extractor"
    paragraphs = [" ".join(p.split()) for p in re.findall(r"<p>(.*?)</p>", html, re.S)]
    return extract.Extracted("\n\n".join(paragraphs), "fake", "en") if paragraphs else None


@pytest.fixture(autouse=True)
def _extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "main_text", _fake_main_text)


def _served() -> str:
    found = _fake_main_text(ORIGINAL, URL)
    assert found is not None
    return str(found.text)


class _Net:
    """Answers queued per call; records each request's headers."""

    def __init__(self) -> None:
        self.answers: List[Optional[httpx.Response]] = []
        self.sent: List[Dict[str, str]] = []
        self.robots = True

    def get(
        self, url: str, agent: str, timeout: int, *, extra_headers: Optional[Dict[str, str]] = None
    ) -> Optional[httpx.Response]:
        self.sent.append(dict(extra_headers or {}))
        return self.answers.pop(0)


def _resp(status: int, html: str = "", headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return httpx.Response(
        status, text=html, headers=headers or {}, request=httpx.Request("GET", URL)
    )


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> _Net:
    fake = _Net()
    monkeypatch.setattr(downloader, "http_get_response", fake.get)
    monkeypatch.setattr(recheck_fetch, "robots_allow", lambda url, agent: fake.robots)
    return fake


def _check(net: _Net, answer: Optional[httpx.Response], **kw: Any) -> recheck_fetch.Check:
    net.answers.append(answer)
    return recheck_fetch.check(_served(), URL, SOURCE, **kw)


def test_the_same_page_is_the_same(net: _Net) -> None:
    got = _check(net, _resp(200, ORIGINAL, {"ETag": '"v1"'}))
    assert got.status == recheck.SAME
    assert got.validators == recheck_fetch.Validators(etag='"v1"')
    assert got.html == ORIGINAL


def test_a_whitespace_change_is_the_same(net: _Net) -> None:
    assert _check(net, _resp(200, _page(_P, spacing="  \n "))).status == recheck.SAME


def test_an_edit_is_a_new_version(net: _Net) -> None:
    edited = list(_P)
    edited[1] = edited[1].replace("eleven weeks", "fourteen weeks")
    got = _check(net, _resp(200, _page(edited)))
    assert got.status == recheck.CHANGED
    assert (
        got.comparison is not None and got.comparison.served_hash != got.comparison.candidate_hash
    )


def test_a_paywall_in_the_articles_place_is_held(net: _Net) -> None:
    teaser = _page([_P[0][:120] + " Subscribe to read the full story."])
    got = _check(net, _resp(200, teaser))
    assert got.status in (recheck.HELD_SHORT, recheck.HELD_NOT_FULL)


def test_a_page_with_no_text_is_held_never_written(net: _Net) -> None:
    got = _check(net, _resp(200, "<html><body><nav>Home</nav></body></html>"))
    assert got.status == recheck.HELD_SHORT


@pytest.mark.parametrize("status", [404, 410])
def test_404_and_410_are_gone(net: _Net, status: int) -> None:
    assert _check(net, _resp(status)).status == recheck.GONE


@pytest.mark.parametrize("answer", [None, 500, 503, 403, 429, 301])
def test_a_timeout_a_5xx_or_a_refusal_is_failed_not_gone(net: _Net, answer: Optional[int]) -> None:
    got = _check(net, None if answer is None else _resp(answer))
    assert got.status == recheck.FAILED
    assert got.http_status == (answer or 0)


def test_robots_refusing_is_failed_and_fetches_nothing(net: _Net) -> None:
    net.robots = False
    assert recheck_fetch.check(_served(), URL, SOURCE).status == recheck.FAILED
    assert net.sent == []


def test_validators_make_the_get_conditional_and_not_modified_is_same(net: _Net) -> None:
    old = recheck_fetch.Validators(etag='"v1"', last_modified="Mon, 24 Nov 2026 06:00:00 GMT")
    got = _check(net, _resp(304, headers={"ETag": '"v1"'}), validators=old)
    assert net.sent == [
        {"If-None-Match": '"v1"', "If-Modified-Since": "Mon, 24 Nov 2026 06:00:00 GMT"}
    ]
    assert got.status == recheck.SAME and got.comparison is None
    assert got.validators == old, "a 304 that repeats only the ETag keeps the Last-Modified"


def test_no_validators_sends_no_conditional_headers(net: _Net) -> None:
    _check(net, _resp(200, ORIGINAL))
    assert net.sent == [{}]


def test_two_gone_checks_in_a_row_withdraw_and_a_failure_between_does_not(net: _Net) -> None:
    from datetime import datetime, timezone

    state = recheck.start(datetime(2026, 11, 24, tzinfo=timezone.utc))
    for answer in (_resp(404), None, _resp(404)):
        state, action = recheck.advance(state, _check(net, answer).status)
        assert action == recheck.RECORD
    state, action = recheck.advance(state, _check(net, _resp(410)).status)
    assert action == recheck.WITHDRAW and state.done
