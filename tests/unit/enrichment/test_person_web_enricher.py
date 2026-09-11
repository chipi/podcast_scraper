"""Unit tests for the person_web enricher (wave-G, EnricherTier.WEB) — fixtured, no network."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from podcast_scraper.enrichment.enrichers import person_web
from podcast_scraper.enrichment.enrichers.person_web import (
    PersonWebEnricher,
    WikipediaProvider,
)
from podcast_scraper.enrichment.protocol import (
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
)

pytestmark = pytest.mark.unit


def _bundle(stem: str) -> EpisodeArtifactBundle:
    p = Path(f"/unused/{stem}.gi.json")
    return EpisodeArtifactBundle(
        metadata_path=p, gi_path=p, kg_path=None, bridge_path=None, episode_id=stem, stem=stem
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="r",
        parent_run_id=None,
        enricher_id="person_web",
        enricher_version="0.1.0",
        tier="web",
        attempt=1,
        job_id="r",
        cancel_event=asyncio.Event(),
    )


class _FakeProvider:
    """A raw payload per found person; counts fetches so cache-reuse is observable."""

    name = "fake"

    def __init__(self, found: set[str]) -> None:
        self._found = found
        self.fetch_calls = 0

    def fetch_raw(self, person_id, display_name):
        self.fetch_calls += 1
        if person_id not in self._found:
            return None
        return {"type": "standard", "extract": f"{display_name} is a person."}

    def derive(self, person_id, display_name, raw):
        extract = raw.get("extract")
        if not extract:
            return None
        return person_web.PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio=extract,
            description="A fake person.",
            image_url=None,
            source="fake",
            source_url=None,
            license="CC-BY-SA 4.0",
        )


_GI = {
    "nodes": [
        {"id": "person:jane", "type": "Person", "properties": {"name": "Jane Doe"}},
        {"id": "person:john", "type": "Person", "properties": {"name": "John Roe"}},
    ]
}


def _run(enricher, tmp_path, config=None):
    return asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=[_bundle("a")],
            config=config or {},
            ctx=_ctx(),
        )
    )


class _ImageProvider:
    """Derives an image_url; fetch_image is scripted (a FetchedImage / IMAGE_SKIP / None) and counts
    calls so the skip-cache (no re-fetch) is observable."""

    name = "fake"

    def __init__(self, image_result) -> None:
        self._image_result = image_result
        self.image_calls = 0

    def fetch_raw(self, person_id, display_name):
        return {"type": "standard", "extract": f"{display_name} bio.", "image": "x"}

    def derive(self, person_id, display_name, raw):
        return person_web.PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio=raw["extract"],
            description="Fake descriptor.",
            image_url="https://img.example/x.png",
            source="fake",
            source_url=None,
            license="CC-BY-SA 4.0",
        )

    def fetch_image(self, image_url):
        self.image_calls += 1
        return self._image_result


def test_image_hosted_when_provider_returns_a_validated_image(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    img = person_web.FetchedImage(
        data=b"\x89PNG\r\n\x1a\n", ext="png", license="CC BY-SA 4.0", artist="A"
    )
    result = _run(PersonWebEnricher(provider=_ImageProvider(img)), tmp_path)
    row = result.data["persons"][0]
    assert row["image_hosted"] is True and row["image_ext"] == "png"
    assert row["image_license"] == "CC BY-SA 4.0"
    assert person_web.person_image_path(tmp_path, "person:jane") is not None


def test_transient_image_error_is_not_cached_and_retries(monkeypatch, tmp_path: Path) -> None:
    # fetch_image → None means a TRANSIENT error: not hosted, and NO skip sidecar (so the next run
    # re-attempts). Both persons in _GI are tried each run.
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    prov = _ImageProvider(None)
    _run(PersonWebEnricher(provider=prov), tmp_path)
    assert prov.image_calls == 2  # jane + john
    assert person_web.person_image_path(tmp_path, "person:jane") is None
    assert person_web._read_image_meta(tmp_path, "person:jane") is None  # nothing cached
    prov2 = _ImageProvider(None)
    _run(PersonWebEnricher(provider=prov2), tmp_path)
    assert prov2.image_calls == 2  # retried, not suppressed


def test_unhostable_image_caches_a_skip_and_never_refetches(monkeypatch, tmp_path: Path) -> None:
    # IMAGE_SKIP means PERMANENTLY un-hostable (no license / bad bytes): not hosted, skip cached,
    # and a second run does NOT call fetch_image again.
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    prov = _ImageProvider(person_web.IMAGE_SKIP)
    _run(PersonWebEnricher(provider=prov), tmp_path)
    assert prov.image_calls == 2
    assert person_web.person_image_path(tmp_path, "person:jane") is None
    assert person_web._read_image_meta(tmp_path, "person:jane") == {"skip": True}
    prov2 = _ImageProvider(person_web.IMAGE_SKIP)
    _run(PersonWebEnricher(provider=prov2), tmp_path)
    assert prov2.image_calls == 0  # skip sidecar suppressed the re-fetch entirely


def test_hosted_image_refetched_when_file_vanished(monkeypatch, tmp_path: Path) -> None:
    # A hosted sidecar whose image FILE was deleted must be treated as a miss and re-fetched
    # (sidecar↔file consistency), not served as a phantom hosted photo.
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    img = person_web.FetchedImage(
        data=b"\x89PNG\r\n\x1a\n", ext="png", license="CC BY-SA 4.0", artist="A"
    )
    prov = _ImageProvider(img)
    _run(PersonWebEnricher(provider=prov), tmp_path)
    stored = person_web.person_image_path(tmp_path, "person:jane")
    assert stored is not None
    stored[0].unlink()  # the file vanishes but the sidecar remains
    prov2 = _ImageProvider(img)
    result = _run(PersonWebEnricher(provider=prov2), tmp_path)
    assert prov2.image_calls == 1  # only jane re-fetched (file gone); john reused from its sidecar
    assert result.data["persons"][0]["image_hosted"] is True
    assert person_web.person_image_path(tmp_path, "person:jane") is not None


def test_manifest_is_web_tier_corpus_scope() -> None:
    m = PersonWebEnricher().manifest
    assert m.tier is EnricherTier.WEB and m.scope is EnricherScope.CORPUS
    assert m.writes == "person_web.json"
    assert m.requires_opt_in is False  # ON by default in cloud/prod; airgap held by profile


def test_distinct_persons_dedups_across_bundles(monkeypatch) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    persons = person_web._distinct_persons([_bundle("a"), _bundle("b")])
    assert persons == [("person:jane", "Jane Doe"), ("person:john", "John Roe")]


def test_enrich_writes_rows_and_persists_raw(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    enricher = PersonWebEnricher(provider=_FakeProvider(found={"person:jane"}))
    result = _run(enricher, tmp_path)
    assert result.status == "ok" and result.data is not None
    rows = result.data["persons"]
    assert [r["person_id"] for r in rows] == ["person:jane"]  # the miss is dropped, not nulled
    assert rows[0]["bio"] == "Jane Doe is a person."
    # RAW is persisted per person for later re-derivation / mining.
    raw = person_web._read_raw_cache(tmp_path, "person:jane")
    assert raw is not None and raw["extract"] == "Jane Doe is a person."


def test_cached_raw_is_reused_without_refetch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    first = _FakeProvider(found={"person:jane"})
    _run(PersonWebEnricher(provider=first), tmp_path)
    assert first.fetch_calls == 2  # jane + john both attempted on the first pass

    # A second run with a NEW provider that would fetch again — but cached raw means derive-only.
    second = _FakeProvider(found={"person:jane"})
    result = _run(PersonWebEnricher(provider=second), tmp_path)
    assert second.fetch_calls == 1  # only john (uncached miss) re-attempted; jane from cache
    assert [r["person_id"] for r in result.data["persons"]] == ["person:jane"]


def test_refresh_bypasses_the_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    _run(PersonWebEnricher(provider=_FakeProvider(found={"person:jane"})), tmp_path)
    refetch = _FakeProvider(found={"person:jane"})
    _run(PersonWebEnricher(provider=refetch), tmp_path, config={"refresh": True})
    assert refetch.fetch_calls == 2  # both re-fetched despite the cache


def test_enrich_respects_max_persons(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    result = _run(
        PersonWebEnricher(provider=_FakeProvider(found={"person:jane", "person:john"})),
        tmp_path,
        config={"max_persons": 1},
    )
    assert result.records_written == 1  # capped before fetch


def _json_client(payload: dict) -> httpx.Client:
    """An httpx client whose MockTransport answers every GET with ``payload`` — no network."""

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(_handler))


def _boom_client() -> httpx.Client:
    """An httpx client that raises a transport error on every request (simulates no network)."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    return httpx.Client(transport=httpx.MockTransport(_handler))


def test_wikipedia_provider_fetch_then_derive() -> None:
    p = WikipediaProvider(
        client=_json_client(
            {
                "type": "standard",
                "extract": "Jane Doe is a researcher.",
                "thumbnail": {"source": "https://img/jane.jpg"},
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Jane_Doe"}},
            }
        )
    )
    raw = p.fetch_raw("person:jane", "Jane Doe")
    assert raw is not None and raw["extract"].startswith("Jane Doe")
    info = p.derive("person:jane", "Jane Doe", raw)
    assert info is not None
    assert info.bio == "Jane Doe is a researcher."
    assert info.image_url == "https://img/jane.jpg"
    assert info.source == "wikipedia"
    assert info.source_url is not None and info.source_url.endswith("Jane_Doe")


def test_wikipedia_derive_none_on_disambiguation() -> None:
    p = WikipediaProvider(client=_boom_client())
    assert p.derive("person:x", "X", {"type": "disambiguation", "extract": "many"}) is None


def test_wikipedia_fetch_raw_swallows_network_error() -> None:
    assert WikipediaProvider(client=_boom_client()).fetch_raw("person:x", "X") is None


def test_fetch_image_skips_when_imageinfo_resolves_without_license() -> None:
    # imageinfo answered but carries no LicenseShortName → resolved-but-unlicensed → PERMANENT skip
    # (never host what we cannot attribute). No download is attempted.
    p = WikipediaProvider(
        client=_json_client({"query": {"pages": {"-1": {"imageinfo": [{"extmetadata": {}}]}}}})
    )
    assert p.fetch_image("https://img.example/x.png") is person_web.IMAGE_SKIP


def test_fetch_image_none_on_transient_imageinfo_error() -> None:
    # A network failure resolving the license is TRANSIENT → None (retry next run), not a skip.
    assert WikipediaProvider(client=_boom_client()).fetch_image("https://img.example/x.png") is None


@pytest.mark.parametrize(
    "url,expected",
    [
        # REST thumbnail: the real file is the pre-rendition segment; the query must be dropped.
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/"
            "President_Barack_Obama.jpg/330px-President_Barack_Obama.jpg"
            "?utm_source=en.wikipedia.org",
            "President_Barack_Obama.jpg",
        ),
        # Non-thumb / original: the last segment is the file.
        ("https://upload.wikimedia.org/wikipedia/commons/8/8d/Jane_Doe.jpg", "Jane_Doe.jpg"),
        # Percent-escapes are decoded so the caller re-encodes exactly once.
        ("https://x/thumb/a/ab/Jos%C3%A9_Mour.jpg/50px-Jos%C3%A9_Mour.jpg", "José_Mour.jpg"),
    ],
)
def test_wiki_file_title(url: str, expected: str) -> None:
    assert person_web._wiki_file_title(url) == expected


def test_fetch_image_resolves_license_via_correct_title_for_a_thumb_url() -> None:
    # H1 regression: the imageinfo title must be the real File name, not the thumb rendition. The
    # mock answers extmetadata ONLY for the correct title and "missing" otherwise — so a wrong title
    # would fail to host (and cache a skip). A valid PNG then downloads.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    thumb = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/"
        "President_Barack_Obama.jpg/330px-President_Barack_Obama.jpg?utm_source=x"
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        u = str(request.url)
        if "action=query" in u:
            if "File:President_Barack_Obama.jpg" in u and "330px" not in u:
                return httpx.Response(
                    200,
                    json={
                        "query": {
                            "pages": {
                                "-1": {
                                    "imageinfo": [
                                        {
                                            "extmetadata": {
                                                "LicenseShortName": {"value": "CC BY 2.0"}
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    },
                )
            return httpx.Response(200, json={"query": {"pages": {"-1": {"missing": ""}}}})
        return httpx.Response(200, content=png, headers={"Content-Type": "image/png"})

    p = WikipediaProvider(client=httpx.Client(transport=httpx.MockTransport(_handler)))
    result = p.fetch_image(thumb)
    assert isinstance(result, person_web.FetchedImage) and result.license == "CC BY 2.0"


def test_imageinfo_200_with_error_is_transient_not_skip() -> None:
    # H2 regression: MediaWiki reports rate-limits as 200 + {"error":…}; that must be a retry
    # (None), never a permanent skip.
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": "ratelimited", "info": "slow down"}})

    p = WikipediaProvider(client=httpx.Client(transport=httpx.MockTransport(_handler)))
    assert p.fetch_image("https://upload.wikimedia.org/wikipedia/commons/8/8d/X.jpg") is None


def test_fetch_image_skips_off_allowlist_host() -> None:
    # M2 (SSRF guard): even if a license resolves, an image on a non-Wikimedia / non-summary host is
    # never downloaded — permanent skip.
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "-1": {
                            "imageinfo": [{"extmetadata": {"LicenseShortName": {"value": "CC"}}}]
                        }
                    }
                }
            },
        )

    p = WikipediaProvider(client=httpx.Client(transport=httpx.MockTransport(_handler)))
    assert p.fetch_image("https://evil.example.com/steal.png") is person_web.IMAGE_SKIP


def test_fetch_image_404_is_permanent_skip() -> None:
    # L1: a gone image (404/410) is permanent — don't re-attempt every run.
    def _handler(request: httpx.Request) -> httpx.Response:
        if "action=query" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": {
                            "-1": {
                                "imageinfo": [
                                    {"extmetadata": {"LicenseShortName": {"value": "CC"}}}
                                ]
                            }
                        }
                    }
                },
            )
        return httpx.Response(404)

    p = WikipediaProvider(client=httpx.Client(transport=httpx.MockTransport(_handler)))
    assert (
        p.fetch_image("https://upload.wikimedia.org/wikipedia/commons/8/8d/Gone.jpg")
        is person_web.IMAGE_SKIP
    )


def test_write_skip_meta_removes_a_previously_stored_photo(tmp_path: Path) -> None:
    # M3: a later skip (e.g. a refresh whose license no longer resolves) must un-host the old photo,
    # not leave a phantom that person_image_path keeps serving.
    img = person_web.FetchedImage(data=b"\x89PNG\r\n\x1a\n", ext="png", license="CC", artist=None)
    person_web._store_image(tmp_path, "person:jane", img)
    assert person_web.person_image_path(tmp_path, "person:jane") is not None
    person_web._write_skip_meta(tmp_path, "person:jane")
    assert person_web.person_image_path(tmp_path, "person:jane") is None
    assert person_web._read_image_meta(tmp_path, "person:jane") == {"skip": True}


def test_fetch_image_downloads_validates_and_attributes(monkeypatch) -> None:
    # The full success path over a MockTransport: imageinfo carries a license, the image bytes are a
    # valid PNG under the cap → a FetchedImage with the resolved license/artist (routed through the
    # retry-wrapped client, no network).
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def _handler(request: httpx.Request) -> httpx.Response:
        if "api.php" in request.url.path or "action=query" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": {
                            "-1": {
                                "imageinfo": [
                                    {
                                        "extmetadata": {
                                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                            "Artist": {"value": "A Photographer"},
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        return httpx.Response(200, content=png, headers={"Content-Type": "image/png"})

    p = WikipediaProvider(client=httpx.Client(transport=httpx.MockTransport(_handler)))
    result = p.fetch_image("https://upload.wikimedia.org/wikipedia/commons/8/8d/jane.png")
    assert isinstance(result, person_web.FetchedImage)
    assert result.ext == "png" and result.license == "CC BY-SA 4.0"
    assert result.artist == "A Photographer"


def test_fetch_image_skips_oversize_download() -> None:
    # Over the 2 MB cap → PERMANENT skip (we never host it), and the stream is bounded.
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (person_web._IMAGE_MAX_BYTES + 16)

    def _handler(request: httpx.Request) -> httpx.Response:
        if "action=query" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": {
                            "-1": {
                                "imageinfo": [
                                    {"extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}}}
                                ]
                            }
                        }
                    }
                },
            )
        return httpx.Response(200, content=big, headers={"Content-Type": "image/png"})

    p = WikipediaProvider(client=httpx.Client(transport=httpx.MockTransport(_handler)))
    assert (
        p.fetch_image("https://upload.wikimedia.org/wikipedia/commons/8/8d/huge.png")
        is person_web.IMAGE_SKIP
    )


def test_web_wiring_registers_and_returns_ids() -> None:
    from podcast_scraper.enrichment.registry import EnricherRegistry
    from podcast_scraper.enrichment.web_wiring import register_web_enrichers

    reg = EnricherRegistry()
    ids = register_web_enrichers(reg)
    assert ids == ["person_web", "org_web"]  # org_web joined the WEB tier (#2035)
    assert reg.get("person_web").manifest.id == "person_web"
    assert reg.get("org_web").manifest.id == "org_web"
