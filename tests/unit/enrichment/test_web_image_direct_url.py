"""Image downloads must skip the ``Special:FilePath`` redirect chain (#2163, item 2).

What it cost. Downloading a Commons image via ``Special:FilePath`` is a redirect service; measured
on prod for one logo:

    commons.wikimedia.org/wiki/Special:FilePath/Blue%20Origin%20new%20logo.svg  -> 301
    commons.wikimedia.org/wiki/Special:FilePath/Blue_Origin_new_logo.svg        -> 302
    commons.wikimedia.org/wiki/Special:Redirect/file/Blue_Origin_new_logo.svg   -> 301
    upload.wikimedia.org/wikipedia/commons/3/3f/Blue_Origin_new_logo.svg        -> 200

Four physical requests for one image, THREE of them against ``commons.wikimedia.org`` — the
rate-metered text cluster. And because ``follow_redirects=True`` is handled inside
``httpx.Client`` above our transport, one ``limiter.wait()`` covered all four, so the hops went out
unthrottled. The imageinfo call we already make can return the final ``upload.wikimedia.org`` URL
in the same response, so the metered cost of an image drops from 4 requests to 1.

No network: everything drives ``httpx.MockTransport``. Real PNG bytes come from PIL, which the
downscale path already depends on — nothing here hand-builds an image format.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _COMMONS_THUMB_WIDTH,
    ClusterRateLimiter,
    FetchedImage,
    IMAGE_SKIP,
    set_web_block_breaker,
    WikipediaProvider,
)

FILEPATH_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Example.png?width=640"
THUMB_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Example.png/640px-Example.png"
)
FULL_URL = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.png"


@pytest.fixture(autouse=True)
def _fresh_breaker():
    set_web_block_breaker(None)
    yield
    set_web_block_breaker(None)


def _png_bytes() -> bytes:
    """A REAL, decodable PNG — the download path sniffs magic bytes then re-encodes via PIL."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _imageinfo_body(*, thumburl: str | None, url: str | None = FULL_URL) -> dict:
    info: dict = {
        "extmetadata": {
            "LicenseShortName": {"value": "CC BY-SA 4.0"},
            "Artist": {"value": "Somebody"},
        }
    }
    if thumburl is not None:
        info["thumburl"] = thumburl
    if url is not None:
        info["url"] = url
    return {"query": {"pages": {"1": {"imageinfo": [info]}}}}


def _provider(handler) -> WikipediaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WikipediaProvider(client=client, limiter=ClusterRateLimiter(0.0))


def _run(thumburl: str | None, url: str | None = FULL_URL) -> tuple[object, list[str]]:
    seen: list[str] = []
    png = _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "action=query" in str(request.url):
            return httpx.Response(200, json=_imageinfo_body(thumburl=thumburl, url=url))
        return httpx.Response(200, content=png, headers={"Content-Type": "image/png"})

    return _provider(handler).fetch_image(FILEPATH_URL), seen


def test_the_download_goes_straight_to_the_direct_url() -> None:
    """THE FIX: two requests total — imageinfo, then the upload.wikimedia.org rendition."""
    result, seen = _run(thumburl=THUMB_URL)

    assert isinstance(result, FetchedImage)
    assert len(seen) == 2, f"expected imageinfo + one download, got {seen}"
    assert "action=query" in seen[0]
    assert seen[1] == THUMB_URL
    assert not any(
        "Special:FilePath" in u for u in seen[1:]
    ), f"the redirect chain must not be used for the download, got {seen}"


def test_only_one_request_touches_the_metered_text_cluster() -> None:
    """The whole point is the load on commons/wikidata/wikipedia, not the total request count."""
    _, seen = _run(thumburl=THUMB_URL)

    metered = [u for u in seen if "commons.wikimedia.org" in u or "wikipedia.org" in u]

    assert len(metered) == 1, f"expected exactly one metered request, got {metered}"


def test_imageinfo_asks_for_the_url_and_the_thumb_width() -> None:
    """Guard the query itself — without these params there is no thumburl to use."""
    _, seen = _run(thumburl=THUMB_URL)

    query = seen[0]
    assert "iiprop=extmetadata%7Curl" in query
    assert f"iiurlwidth={_COMMONS_THUMB_WIDTH}" in query


def test_falls_back_to_the_full_url_when_there_is_no_thumb() -> None:
    """Not every file can be thumbnailed; ``url`` is still direct, so still one hop."""
    result, seen = _run(thumburl=None)

    assert isinstance(result, FetchedImage)
    assert seen[1] == FULL_URL


def test_falls_back_to_the_callers_url_when_imageinfo_gives_neither() -> None:
    """No regression: behave exactly as before when the API returns no URL at all."""
    result, seen = _run(thumburl=None, url=None)

    assert isinstance(result, FetchedImage)
    assert seen[1] == FILEPATH_URL


def test_an_off_allowlist_direct_url_is_refused() -> None:
    """SSRF: ``thumburl`` is external payload, so the guard must judge the URL actually fetched.

    Checking the ORIGINAL url and then downloading a different one would leave the real request
    unvalidated — the bytes get re-served to users.
    """
    result, seen = _run(thumburl="https://evil.example.com/payload.png")

    assert result is IMAGE_SKIP
    assert not any(
        "evil.example.com" in u for u in seen
    ), f"an off-allowlist host must never be requested, got {seen}"


def test_a_transient_imageinfo_failure_is_still_retryable() -> None:
    """Unchanged contract: None means retry next run, never a cached permanent skip."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    assert _provider(handler).fetch_image(FILEPATH_URL) is None


def test_missing_license_is_still_a_permanent_skip() -> None:
    """Unchanged contract: never host what we cannot attribute."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = {"query": {"pages": {"1": {"imageinfo": [{"thumburl": THUMB_URL}]}}}}
        return httpx.Response(200, json=json.loads(json.dumps(body)))

    assert _provider(handler).fetch_image(FILEPATH_URL) is IMAGE_SKIP
