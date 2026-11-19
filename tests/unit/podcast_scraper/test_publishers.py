"""The publisher registry (ADR-180 rule 6)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from podcast_scraper.publishers import DEFAULT_PATH, load_registry, PublisherRegistryError

pytestmark = pytest.mark.unit

_BBC = {
    "id": "org:bbc",
    "name": "BBC",
    "domains": ["bbc.co.uk", "www.bbc.com"],
    "external_ids": {"wikidata": "Q9531"},
    "sources": ["podcast:global-news-podcast", "source:bbc-world"],
}
_BBC_SPORT = {"id": "org:bbc-sport", "name": "BBC Sport", "domains": ["sport.bbc.co.uk"]}


def _write(tmp_path: Path, publishers: list) -> Path:
    path = tmp_path / "publishers.yaml"
    path.write_text(yaml.safe_dump({"publishers": publishers}), encoding="utf-8")
    return path


def test_the_shipped_registry_loads() -> None:
    load_registry(DEFAULT_PATH)


def test_a_missing_file_is_an_empty_registry(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "none.yaml").publishers == ()


def test_one_organisation_across_podcasts_and_news(tmp_path: Path) -> None:
    reg = load_registry(_write(tmp_path, [_BBC]))
    assert reg.for_source("podcast:global-news-podcast") is reg.for_source("source:bbc-world")
    assert reg.by_id("org:bbc") is reg.for_source("source:bbc-world")
    assert reg.for_source("source:unlisted") is None


def test_a_domain_matches_itself_and_its_subdomains_longest_first(tmp_path: Path) -> None:
    reg = load_registry(_write(tmp_path, [_BBC, _BBC_SPORT]))
    assert reg.for_domain("news.bbc.co.uk") is reg.by_id("org:bbc")
    assert reg.for_domain("M.BBC.COM") is reg.by_id("org:bbc")
    assert reg.for_domain("live.sport.bbc.co.uk") is reg.by_id("org:bbc-sport")
    assert reg.for_domain("notbbc.co.uk") is None


def test_the_served_shape_names_the_matched_domain(tmp_path: Path) -> None:
    bbc = load_registry(_write(tmp_path, [_BBC])).by_id("org:bbc")
    assert bbc is not None
    assert bbc.served("www.bbc.com") == {
        "id": "org:bbc",
        "name": "BBC",
        "domain": "bbc.com",
    }
    assert bbc.served() == {"id": "org:bbc", "name": "BBC", "domain": "bbc.co.uk"}


@pytest.mark.parametrize(
    "second",
    [
        {"id": "org:bbc", "name": "BBC again"},
        {"id": "org:other", "name": "Other", "domains": ["bbc.co.uk"]},
        {"id": "org:other", "name": "Other", "sources": ["source:bbc-world"]},
        {"id": "org:other", "name": "Other", "external_ids": {"wikidata": "Q9531"}},
    ],
)
def test_a_key_claimed_twice_is_refused(tmp_path: Path, second: dict) -> None:
    with pytest.raises(PublisherRegistryError, match="claimed by both"):
        load_registry(_write(tmp_path, [_BBC, second]))


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "publisher:bbc", "name": "BBC"},
        {"id": "org:", "name": "BBC"},
        {"id": "org:bbc"},
        {"id": "org:bbc", "name": "BBC", "sources": ["rss_anchor.fm_37c73efb"]},
    ],
)
def test_an_entry_outside_the_rules_is_refused(tmp_path: Path, bad: dict) -> None:
    with pytest.raises(PublisherRegistryError):
        load_registry(_write(tmp_path, [bad]))


def test_the_env_names_another_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_PUBLISHERS_FILE", str(_write(tmp_path, [_BBC])))
    assert [p.id for p in load_registry().publishers] == ["org:bbc"]
    monkeypatch.delenv("PODCAST_PUBLISHERS_FILE")
    assert load_registry().publishers == load_registry(DEFAULT_PATH).publishers
